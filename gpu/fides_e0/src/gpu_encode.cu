// Device half of the GPU encoder (see gpu_encode.hpp).
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <variant>

#include <cuda_runtime.h>

#include "CKKS/Context.cuh"
#include "CKKS/Limb.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/RNSPoly.cuh"

#include "gpu_encode.hpp"

using namespace FIDESlib::CKKS;

#define CK(x)                                                                                    \
	do {                                                                                         \
		cudaError_t err_ = (x);                                                                  \
		if (err_ != cudaSuccess) {                                                               \
			std::fprintf(stderr, "gpu_encode: %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
			             __LINE__);                                                              \
			throw std::runtime_error("gpu_encode CUDA error");                                   \
		}                                                                                        \
	} while (0)

// out[i] = coef[i / gap] mod q at multiples of gap, 0 elsewhere -- what
// CKKSPackedEncoding::FitToNativeVector writes in COEFFICIENT format.
__global__ static void scatter_limb(uint64_t* out, const int64_t* coef, uint32_t N, uint32_t gap,
                                    uint64_t q) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N)
		return;
	uint64_t v = 0;
	if (i % gap == 0) {
		const int64_t c = coef[i / gap];
		v = c >= 0 ? uint64_t(c) % q : (q - uint64_t(-(c + 1)) % q - 1) % q;  // no overflow at INT64_MIN
	}
	out[i] = v;
}

namespace e0 {

struct GpuEncoder {
	Context ctx;
	ContextData* cc = nullptr;
	int level       = 0;
	uint32_t N = 0, D = 0, gap = 0;
	double sf = 0;
	std::unique_ptr<RNSPoly> scratch;
	int64_t* d_coef = nullptr;
	size_t n_sets   = 0;
};

static Limb<uint64_t>& limb_at(ContextData& cc, RNSPoly& p, int i) {
	auto& part = p.GPU.at(cc.limbGPUid.at(i).x);
	return std::get<Limb<uint64_t>>(part.limb.at(cc.limbGPUid.at(i).y));
}

GpuEncoder* gpuenc_create(std::any& gpu_ctx, std::shared_ptr<void>& ref_dev_pt, uint32_t D) {
	auto* e = new GpuEncoder;
	e->ctx  = std::any_cast<Context&>(gpu_ctx);
	e->cc   = e->ctx.get();
	SetCurrentContext(e->ctx);
	auto& ref = *std::static_pointer_cast<Plaintext>(ref_dev_pt);
	e->level  = ref.c0.getLevel();
	e->sf     = ref.NoiseFactor;
	e->N      = uint32_t(e->cc->N);
	e->D      = D;
	e->gap    = e->N / (2 * D);  // FitToNativeVector: ringDim / (2 * slots)
	// Non-constant limbs: these carry the aux buffer the NTT kernels write through.
	e->scratch = std::make_unique<RNSPoly>(*e->cc, e->level);
	CK(cudaDeviceSynchronize());
	return e;
}

double gpuenc_scaling_factor(const GpuEncoder* e) { return e->sf; }

double gpuenc_upload(GpuEncoder* e, const std::vector<std::vector<int64_t>>& coeffs) {
	auto t0         = std::chrono::steady_clock::now();
	const size_t sz = 2 * size_t(e->D);
	std::vector<int64_t> flat(coeffs.size() * sz);
	for (size_t k = 0; k < coeffs.size(); ++k) {
		if (coeffs[k].size() != sz)
			throw std::runtime_error("gpuenc_upload: coefficient set has the wrong size");
		std::copy(coeffs[k].begin(), coeffs[k].end(), flat.begin() + k * sz);
	}
	if (e->d_coef)
		CK(cudaFree(e->d_coef));
	CK(cudaMalloc(&e->d_coef, flat.size() * sizeof(int64_t)));
	CK(cudaMemcpy(e->d_coef, flat.data(), flat.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
	e->n_sets = coeffs.size();
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void gpuenc_encode(GpuEncoder* e, size_t idx, std::shared_ptr<void>& dst_dev_pt) {
	if (idx >= e->n_sets)
		throw std::runtime_error("gpuenc_encode: index out of range");
	auto& dst = *std::static_pointer_cast<Plaintext>(dst_dev_pt);
	if (dst.c0.getLevel() != e->level)
		throw std::runtime_error("gpuenc_encode: destination plaintext is at a different level");
	ContextData& cc        = *e->cc;
	const int64_t* coef    = e->d_coef + idx * 2 * size_t(e->D);
	constexpr int block    = 256;
	const uint32_t grid    = (e->N + block - 1) / block;
	CK(cudaDeviceSynchronize());  // previous users of dst / scratch are done
	for (int i = 0; i <= e->level; ++i) {
		auto& l = limb_at(cc, *e->scratch, i);
		CK(cudaSetDevice(e->scratch->GPU.at(cc.limbGPUid.at(i).x).device));
		scatter_limb<<<grid, block, 0, l.stream.ptr()>>>(l.v.data, coef, e->N, e->gap, cc.prime.at(l.primeid).p);
	}
	CK(cudaGetLastError());
	CK(cudaDeviceSynchronize());
	e->scratch->NTT(cc.batch, true);
	CK(cudaDeviceSynchronize());
	for (int i = 0; i <= e->level; ++i) {
		auto& s = limb_at(cc, *e->scratch, i);
		auto& d = limb_at(cc, dst.c0, i);
		CK(cudaMemcpyAsync(d.v.data, s.v.data, size_t(e->N) * sizeof(uint64_t), cudaMemcpyDeviceToDevice,
		                   d.stream.ptr()));
	}
	CK(cudaDeviceSynchronize());
	dst.NoiseFactor = e->sf;
	dst.NoiseLevel  = 1;
}

uint64_t gpuenc_count_diff(GpuEncoder* e, std::shared_ptr<void>& a, std::shared_ptr<void>& b) {
	CK(cudaDeviceSynchronize());
	auto& pa = *std::static_pointer_cast<Plaintext>(a);
	auto& pb = *std::static_pointer_cast<Plaintext>(b);
	if (pa.c0.getLevel() != pb.c0.getLevel())
		return ~uint64_t(0);
	ContextData& cc = *e->cc;
	uint64_t diff   = 0;
	std::vector<uint64_t> ha(e->N), hb(e->N);
	for (int i = 0; i <= pa.c0.getLevel(); ++i) {
		CK(cudaMemcpy(ha.data(), limb_at(cc, pa.c0, i).v.data, ha.size() * 8, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(hb.data(), limb_at(cc, pb.c0, i).v.data, hb.size() * 8, cudaMemcpyDeviceToHost));
		for (size_t k = 0; k < ha.size(); ++k)
			diff += ha[k] != hb[k];
	}
	return diff;
}

void gpuenc_destroy(GpuEncoder* e) {
	if (!e)
		return;
	if (e->d_coef)
		cudaFree(e->d_coef);
	delete e;
}

}  // namespace e0
