// Device half of the GPU encoder (see gpu_encode.hpp).
#include <chrono>
#include <cstdio>
#include <map>
#include <set>
#include <unordered_map>
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

// ---- sparse encode (FIDES_VM_SPARSE) ----------------------------------------------------------
// p(X) = sum_k c_k X^(k*gap) with M = 2D coefficients, gap = N/M. At every root w of X^N + 1, w^gap is a
// root of Y^M + 1, so p's N evaluations are the M evaluations of q(Y) = sum_k c_k Y^k, each repeated gap
// times. One block per (limb, plaintext): an M-point negacyclic NTT in shared memory (twist by zeta^k,
// cyclic radix-2 DIT with omega = zeta^2), then out[r] = Q[map[r]]. zeta, map and the representation
// scale s come from FIDESlib's own NTT of 1 and X^gap at startup, so its output order and Montgomery-
// ness need not be known here (sparse_setup); a bit-exact check against the full path follows.
static constexpr int SP_MAXB = 16, SP_MAXL = 24;
struct SpLimb {
	uint64_t p;
	const uint64_t *tw, *twS, *W, *WS;   // tw[k] = s*zeta^k (k < M), W[m] = zeta^(2m) (m < M/2); *S = Shoup
	const uint16_t* map;                 // N entries: output r <- Q[map[r]]
};
struct SpArgs {
	int nl, nb;
	const SpLimb* tabs;   // nl entries (this device's limbs)
	const int64_t* coef;
	uint64_t coef_off[SP_MAXB];
	uint64_t* out[SP_MAXB][SP_MAXL];
};

__device__ __forceinline__ uint64_t mul_shoup(uint64_t x, uint64_t w, uint64_t ws, uint64_t p) {
	const uint64_t q = __umul64hi(x, ws);
	uint64_t r       = x * w - q * p;
	return r >= p ? r - p : r;
}

__global__ void __launch_bounds__(1024) sparse_encode(SpArgs a, uint32_t M, uint32_t logM, uint32_t N) {
	extern __shared__ uint64_t sm[];
	const SpLimb t   = a.tabs[blockIdx.x];
	const uint64_t p = t.p;
	const int64_t* c = a.coef + a.coef_off[blockIdx.y];
	for (uint32_t k = threadIdx.x; k < M; k += blockDim.x) {
		const int64_t ck = c[k];
		uint64_t x = ck >= 0 ? uint64_t(ck) % p : (p - uint64_t(-(ck + 1)) % p - 1) % p;
		sm[__brev(k) >> (32 - logM)] = mul_shoup(x, t.tw[k], t.twS[k], p);
	}
	__syncthreads();
	for (uint32_t len = 2; len <= M; len <<= 1) {
		const uint32_t half = len >> 1, step = M / len;
		for (uint32_t u = threadIdx.x; u < M / 2; u += blockDim.x) {
			const uint32_t pos = u % half, i = (u / half) * len + pos, wi = pos * step;
			const uint64_t x = sm[i], y = mul_shoup(sm[i + half], t.W[wi], t.WS[wi], p);
			const uint64_t s = x + y;
			sm[i]        = s >= p ? s - p : s;
			sm[i + half] = x >= y ? x - y : x + p - y;
		}
		__syncthreads();
	}
	uint64_t* o = a.out[blockIdx.y][blockIdx.x];
	for (uint32_t r = threadIdx.x; r < N; r += blockDim.x) o[r] = sm[t.map[r]];
}

namespace e0 {

struct GpuEncoder {
	Context ctx;
	ContextData* cc = nullptr;
	int level       = 0;
	uint32_t N = 0, D = 0, gap = 0;
	double sf = 0;
	std::unique_ptr<RNSPoly> scratch;
	std::vector<std::unique_ptr<RNSPoly>> scr;   // gpuenc_encode_batch: one scratch per batch slot
	// One copy of the coefficient sets per device: with FIDES_E0_GPUS > 1 the
	// limbs (and so the scatter kernels) are spread over the cards.
	std::map<int, int64_t*> d_coef;
	size_t n_sets = 0;
	// sparse path: per partition, the device table of its limbs (in limb order) and their limb indices
	bool sparse = false;
	uint32_t M = 0, logM = 0;
	std::map<int, SpLimb*> d_tabs;          // partition -> device array
	std::map<int, std::vector<int>> plimbs; // partition -> limb indices i
	std::vector<void*> d_alloc;             // (device, ptr) of every table allocation, for destroy
	std::vector<int> d_alloc_dev;
};

static void sync_all(ContextData& cc) {
	int cur = 0;
	cudaGetDevice(&cur);
	for (int dev : std::set<int>(cc.GPUid.begin(), cc.GPUid.end())) {
		CK(cudaSetDevice(dev));
		CK(cudaDeviceSynchronize());
	}
	CK(cudaSetDevice(cur));
}

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
	sync_all(*e->cc);
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
	std::set<int> devs;
	for (auto& part : e->scratch->GPU) devs.insert(part.device);
	for (int dev : devs) {
		CK(cudaSetDevice(dev));
		if (e->d_coef.count(dev))
			CK(cudaFree(e->d_coef[dev]));
		CK(cudaMalloc(&e->d_coef[dev], flat.size() * sizeof(int64_t)));
		CK(cudaMemcpy(e->d_coef[dev], flat.data(), flat.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
	}
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
	const size_t coef_off  = idx * 2 * size_t(e->D);
	constexpr int block    = 256;
	const uint32_t grid    = (e->N + block - 1) / block;
	if (e->sparse) {
		sync_all(*e->cc);
		sparse_launch(e, { idx }, { &dst.c0 });
		sync_all(*e->cc);
		dst.NoiseFactor = e->sf;
		dst.NoiseLevel  = 1;
		return;
	}
	sync_all(*e->cc);  // previous users of dst / scratch are done
	for (int i = 0; i <= e->level; ++i) {
		auto& l       = limb_at(cc, *e->scratch, i);
		const int dev = e->scratch->GPU.at(cc.limbGPUid.at(i).x).device;
		CK(cudaSetDevice(dev));
		scatter_limb<<<grid, block, 0, l.stream.ptr()>>>(l.v.data, e->d_coef.at(dev) + coef_off, e->N, e->gap,
		                                                 cc.prime.at(l.primeid).p);
	}
	CK(cudaGetLastError());
	sync_all(*e->cc);
	e->scratch->NTT(cc.batch, true);
	sync_all(*e->cc);
	for (int i = 0; i <= e->level; ++i) {
		auto& s = limb_at(cc, *e->scratch, i);
		auto& d = limb_at(cc, dst.c0, i);
		CK(cudaSetDevice(e->scratch->GPU.at(cc.limbGPUid.at(i).x).device));
		CK(cudaMemcpyAsync(d.v.data, s.v.data, size_t(e->N) * sizeof(uint64_t), cudaMemcpyDeviceToDevice,
		                   d.stream.ptr()));
	}
	sync_all(*e->cc);
	dst.NoiseFactor = e->sf;
	dst.NoiseLevel  = 1;
}

void gpuenc_encode_batch(GpuEncoder* e, const std::vector<size_t>& idx, const std::vector<std::shared_ptr<void>*>& dst) {
	if (idx.size() != dst.size())
		throw std::runtime_error("gpuenc_encode_batch: idx and dst differ in length");
	ContextData& cc     = *e->cc;
	constexpr int block = 256;
	const uint32_t grid = (e->N + block - 1) / block;
	if (e->sparse) {
		std::vector<RNSPoly*> polys;
		for (size_t j = 0; j < idx.size(); ++j) {
			auto& d = *std::static_pointer_cast<Plaintext>(*dst[j]);
			if (d.c0.getLevel() != e->level)
				throw std::runtime_error("gpuenc_encode_batch: destination plaintext is at a different level");
			polys.push_back(&d.c0);
			d.NoiseFactor = e->sf;
			d.NoiseLevel  = 1;
		}
		sync_all(cc);  // the previous batch's EvalMults have read their plaintexts
		sparse_launch(e, idx, polys);
		sync_all(cc);
		return;
	}
	while (e->scr.size() < idx.size())
		e->scr.push_back(std::make_unique<RNSPoly>(cc, e->level));
	sync_all(cc);  // the previous batch's EvalMults have read their plaintexts
	for (size_t j = 0; j < idx.size(); ++j) {
		if (idx[j] >= e->n_sets)
			throw std::runtime_error("gpuenc_encode_batch: index out of range");
		auto& d = *std::static_pointer_cast<Plaintext>(*dst[j]);
		if (d.c0.getLevel() != e->level)
			throw std::runtime_error("gpuenc_encode_batch: destination plaintext is at a different level");
		RNSPoly& sp           = *e->scr[j];
		const size_t coef_off = idx[j] * 2 * size_t(e->D);
		for (int i = 0; i <= e->level; ++i) {   // scatter on the partition stream: NTT(sync) waits on it
			auto& part = sp.GPU.at(cc.limbGPUid.at(i).x);
			auto& l    = limb_at(cc, sp, i);
			CK(cudaSetDevice(part.device));
			scatter_limb<<<grid, block, 0, part.s.ptr()>>>(l.v.data, e->d_coef.at(part.device) + coef_off, e->N, e->gap,
			                                               cc.prime.at(l.primeid).p);
		}
		CK(cudaGetLastError());
		sp.NTT(cc.batch, true);                  // limb streams wait on part.s, then part.s waits on them
		for (int i = 0; i <= e->level; ++i) {
			auto& part = sp.GPU.at(cc.limbGPUid.at(i).x);
			auto& s    = limb_at(cc, sp, i);
			auto& dd   = limb_at(cc, d.c0, i);
			CK(cudaSetDevice(part.device));
			CK(cudaMemcpyAsync(dd.v.data, s.v.data, size_t(e->N) * sizeof(uint64_t), cudaMemcpyDeviceToDevice, part.s.ptr()));
		}
		d.NoiseFactor = e->sf;
		d.NoiseLevel  = 1;
	}
	sync_all(cc);  // every dst is ready for EvalMult on whatever streams it uses
}

// ---- sparse path -------------------------------------------------------------------------------
using u128 = unsigned __int128;
static uint64_t mulmod(uint64_t a, uint64_t b, uint64_t p) { return uint64_t(u128(a) * b % p); }
static uint64_t powmod(uint64_t a, uint64_t n, uint64_t p) {
	uint64_t r = 1 % p;
	for (; n; n >>= 1, a = mulmod(a, a, p))
		if (n & 1) r = mulmod(r, a, p);
	return r;
}
static uint64_t shoup(uint64_t w, uint64_t p) { return uint64_t((u128(w) << 64) / p); }

// Full path (scatter + FIDESlib NTT) of coefficient set idx into the scratch polynomial; limbs 0..level to host.
static std::vector<std::vector<uint64_t>> full_to_host(GpuEncoder* e, size_t idx, RNSPoly& sp) {
	ContextData& cc     = *e->cc;
	constexpr int block = 256;
	const uint32_t grid = (e->N + block - 1) / block;
	sync_all(cc);
	for (int i = 0; i <= e->level; ++i) {
		auto& part = sp.GPU.at(cc.limbGPUid.at(i).x);
		auto& l    = limb_at(cc, sp, i);
		CK(cudaSetDevice(part.device));
		scatter_limb<<<grid, block, 0, part.s.ptr()>>>(l.v.data, e->d_coef.at(part.device) + idx * 2 * size_t(e->D), e->N,
		                                               e->gap, cc.prime.at(l.primeid).p);
	}
	CK(cudaGetLastError());
	sp.NTT(cc.batch, true);
	sync_all(cc);
	std::vector<std::vector<uint64_t>> h(e->level + 1, std::vector<uint64_t>(e->N));
	for (int i = 0; i <= e->level; ++i) {
		CK(cudaSetDevice(sp.GPU.at(cc.limbGPUid.at(i).x).device));
		CK(cudaMemcpy(h[i].data(), limb_at(cc, sp, i).v.data, size_t(e->N) * 8, cudaMemcpyDeviceToHost));
	}
	return h;
}

template <class T>
static T* dev_upload(GpuEncoder* e, int dev, const std::vector<T>& v) {
	T* d = nullptr;
	CK(cudaSetDevice(dev));
	CK(cudaMalloc(&d, v.size() * sizeof(T)));
	CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
	e->d_alloc.push_back(d);
	e->d_alloc_dev.push_back(dev);
	return d;
}

// One launch per partition: blocks (its limbs) x (plaintexts). out[j][l] = limb l of dst j's c0.
static void sparse_launch(GpuEncoder* e, const std::vector<size_t>& idx, const std::vector<RNSPoly*>& dst) {
	ContextData& cc = *e->cc;
	if (idx.size() > size_t(SP_MAXB)) throw std::runtime_error("sparse encode: batch larger than 16");
	for (auto& [x, limbs] : e->plimbs) {
		SpArgs a{};
		a.nl   = int(limbs.size());
		a.nb   = int(idx.size());
		a.tabs = e->d_tabs.at(x);
		auto& part = dst[0]->GPU.at(x);
		a.coef = e->d_coef.at(part.device);
		for (size_t j = 0; j < idx.size(); ++j) {
			if (idx[j] >= e->n_sets) throw std::runtime_error("sparse encode: index out of range");
			a.coef_off[j] = idx[j] * 2 * size_t(e->D);
			for (size_t l = 0; l < limbs.size(); ++l) a.out[j][l] = limb_at(cc, *dst[j], limbs[l]).v.data;
		}
		CK(cudaSetDevice(part.device));
		sparse_encode<<<dim3(a.nl, a.nb), 1024, e->M * sizeof(uint64_t), part.s.ptr()>>>(a, e->M, e->logM, e->N);
	}
	CK(cudaGetLastError());
}

bool gpuenc_enable_sparse(GpuEncoder* e) {
	ContextData& cc = *e->cc;
	const uint32_t M = 2 * e->D, N = e->N;
	uint32_t logM = 0;
	while ((1u << logM) < M) ++logM;
	if ((1u << logM) != M || M > 4096) throw std::runtime_error("sparse encode: 2D must be a power of two <= 4096");
	e->M = M;
	e->logM = logM;
	// calibration sets: 1, X^gap, and a random one for the bit-exact check
	std::vector<std::vector<int64_t>> sets(3, std::vector<int64_t>(M, 0));
	sets[0][0] = 1;
	sets[1][1] = 1;
	uint64_t st = 0x9e3779b97f4a7c15ull;
	for (auto& v : sets[2]) {
		st ^= st << 13; st ^= st >> 7; st ^= st << 17;
		v = int64_t(st >> 20) - (int64_t(1) << 43);
	}
	gpuenc_upload(e, sets);
	auto one = full_to_host(e, 0, *e->scratch);
	auto xg  = full_to_host(e, 1, *e->scratch);
	std::map<int, std::vector<SpLimb>> tabs;
	for (int i = 0; i <= e->level; ++i) {
		const int x      = cc.limbGPUid.at(i).x;
		const int dev    = e->scratch->GPU.at(x).device;
		const uint64_t p = cc.prime.at(limb_at(cc, *e->scratch, i).primeid).p;
		const uint64_t sc = one[i][0];
		for (uint32_t r = 0; r < N; ++r)
			if (one[i][r] != sc) throw std::runtime_error("sparse encode: NTT(1) is not constant");
		const uint64_t sinv = powmod(sc, p - 2, p);
		const uint64_t zeta = mulmod(xg[i][0], sinv, p);
		if (powmod(zeta, M, p) != p - 1) throw std::runtime_error("sparse encode: zeta is not a primitive 2M-th root");
		std::unordered_map<uint64_t, uint16_t> where;   // zeta^(2j+1) -> j
		const uint64_t z2 = mulmod(zeta, zeta, p);
		uint64_t pw = zeta;
		for (uint32_t j = 0; j < M; ++j, pw = mulmod(pw, z2, p)) where[pw] = uint16_t(j);
		std::vector<uint16_t> map(N);
		for (uint32_t r = 0; r < N; ++r) {
			auto it = where.find(mulmod(xg[i][r], sinv, p));
			if (it == where.end()) throw std::runtime_error("sparse encode: NTT(X^gap) value is not a root of Y^M+1");
			map[r] = it->second;
		}
		std::vector<uint64_t> tw(M), twS(M), W(M / 2), WS(M / 2);
		uint64_t zk = sc;
		for (uint32_t k = 0; k < M; ++k, zk = mulmod(zk, zeta, p)) { tw[k] = zk; twS[k] = shoup(zk, p); }
		uint64_t wm = 1;
		for (uint32_t m = 0; m < M / 2; ++m, wm = mulmod(wm, z2, p)) { W[m] = wm; WS[m] = shoup(wm, p); }
		SpLimb L{ p, dev_upload(e, dev, tw), dev_upload(e, dev, twS), dev_upload(e, dev, W), dev_upload(e, dev, WS),
		          dev_upload(e, dev, map) };
		tabs[x].push_back(L);
		e->plimbs[x].push_back(i);
		if (e->plimbs[x].size() > size_t(SP_MAXL)) throw std::runtime_error("sparse encode: too many limbs on one card");
	}
	for (auto& [x, v] : tabs) e->d_tabs[x] = dev_upload(e, e->scratch->GPU.at(x).device, v);
	// bit-exact check: random set through both paths
	auto ref = full_to_host(e, 2, *e->scratch);
	RNSPoly chk(cc, e->level);
	sync_all(cc);
	sparse_launch(e, { 2 }, { &chk });
	sync_all(cc);
	uint64_t diff = 0;
	std::vector<uint64_t> h(N);
	for (int i = 0; i <= e->level; ++i) {
		CK(cudaSetDevice(chk.GPU.at(cc.limbGPUid.at(i).x).device));
		CK(cudaMemcpy(h.data(), limb_at(cc, chk, i).v.data, size_t(N) * 8, cudaMemcpyDeviceToHost));
		for (uint32_t r = 0; r < N; ++r) diff += h[r] != ref[i][r];
	}
	std::fprintf(stderr, "gpu_encode: sparse encode level %d: scale %s, %llu of %llu residues differ from the full NTT\n",
	             e->level, one[0][0] == 1 ? "1" : "!=1 (Montgomery?)", (unsigned long long)diff,
	             (unsigned long long)(uint64_t(N) * (e->level + 1)));
	e->sparse = diff == 0;
	return e->sparse;
}

uint64_t gpuenc_count_diff(GpuEncoder* e, std::shared_ptr<void>& a, std::shared_ptr<void>& b) {
	sync_all(*e->cc);
	auto& pa = *std::static_pointer_cast<Plaintext>(a);
	auto& pb = *std::static_pointer_cast<Plaintext>(b);
	if (pa.c0.getLevel() != pb.c0.getLevel())
		return ~uint64_t(0);
	ContextData& cc = *e->cc;
	uint64_t diff   = 0;
	std::vector<uint64_t> ha(e->N), hb(e->N);
	for (int i = 0; i <= pa.c0.getLevel(); ++i) {
		// Limbs come from per-device memory pools: copy with the owning device current.
		CK(cudaSetDevice(pa.c0.GPU.at(cc.limbGPUid.at(i).x).device));
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
	for (auto& [dev, ptr] : e->d_coef) {
		cudaSetDevice(dev);
		cudaFree(ptr);
	}
	for (size_t k = 0; k < e->d_alloc.size(); ++k) {
		cudaSetDevice(e->d_alloc_dev[k]);
		cudaFree(e->d_alloc[k]);
	}
	delete e;
}

}  // namespace e0
