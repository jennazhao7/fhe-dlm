// E0 GPU feasibility (HANDOFF §6.1): does the E5 parameter set run on one
// 24 GB Quadro RTX 6000 (sm_75) under FIDESlib, and how fast?
//
// Same CKKS parameters as the CPU sweep (scripts/e0_microbench.py,
// scripts/e0_matvec_bench.py): N = 2^17, depth 43, scaling/first 59/60,
// FLEXIBLEAUTO, HEStd_128_classic, UNIFORM_TERNARY, HYBRID key switching.
//
//   fides_e0 boot <slots> [budget]   [budget,budget] bootstrap, default 4
//   fides_e0 matvec                  1024x1024 BSGS matvec (1->1) at level 21
//
// One mode per process: a CUDA OOM inside FIDESlib is not reliably
// recoverable, and the job script runs each mode separately so one failure
// does not hide the others. Each mode prints exactly one line
//   RESULT {json}
// that cluster/job_gpu_fides.sh collects. GPU memory is read with
// cudaMemGetInfo after each phase (includes FIDESlib's memory-pool cache,
// i.e. what the process actually holds); the job also samples nvidia-smi.
//
// FIDESlib rule: every key (mult, rotation, bootstrap) must be generated
// BEFORE LoadContext, which uploads all of them to the device at once --
// LoadContext is therefore where a 24 GB card either fits or does not.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <sys/resource.h>

#include <fideslib.hpp>

using namespace fideslib;

static constexpr uint32_t DEPTH = 43;
// Overrides for a CPU dry run of the code paths (FIDESlib falls back to
// OpenFHE when no device is set):  FIDES_E0_CPU=1 FIDES_E0_LOGN=12 fides_e0 matvec
static const bool CPU_ONLY = std::getenv("FIDES_E0_CPU") != nullptr;
static const uint32_t RING = 1u << (std::getenv("FIDES_E0_LOGN") ? std::atoi(std::getenv("FIDES_E0_LOGN")) : 17);
static constexpr uint32_t D = 1024, N1 = 32, N2 = D / N1;
static constexpr uint32_t MV_LEVEL = 21;  // post-bootstrap level, as in e0_matvec_bench.py

using clk = std::chrono::steady_clock;
static double since(clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); }

static double gpu_used_gb() {
	size_t free_b = 0, total_b = 0;
	if (CPU_ONLY || cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return -1;
	return double(total_b - free_b) / (1ull << 30);
}
static double host_peak_gb() {
	rusage u{};
	getrusage(RUSAGE_SELF, &u);
	return double(u.ru_maxrss) / (1 << 20);
}

struct Json {
	std::ostringstream s;
	bool first = true;
	template <class T> Json& kv(const std::string& k, const T& v) {
		s << (first ? "{" : ", ") << '"' << k << "\": " << v;
		first = false;
		return *this;
	}
	Json& ks(const std::string& k, const std::string& v) { return kv(k, '"' + v + '"'); }
	std::string str() const { return s.str() + "}"; }
};

static void phase(Json& j, const std::string& name, clk::time_point t0) {
	double s = since(t0), g = gpu_used_gb();
	j.kv(name + "_s", s).kv(name + "_gpu_gb", g);
	std::cerr << "[fides_e0] " << name << ": " << s << " s, gpu used " << g << " GB, host peak "
	          << host_peak_gb() << " GB" << std::endl;
}

static CryptoContext<DCRTPoly> make_context() {
	CCParams<CryptoContextCKKSRNS> p;
	p.SetSecretKeyDist(UNIFORM_TERNARY);
	p.SetSecurityLevel(RING < (1u << 17) ? HEStd_NotSet : HEStd_128_classic);
	p.SetRingDim(RING);
	p.SetScalingModSize(59);
	p.SetFirstModSize(60);
	p.SetScalingTechnique(FLEXIBLEAUTO);
	p.SetKeySwitchTechnique(HYBRID);
	p.SetMultiplicativeDepth(DEPTH);
	if (!CPU_ONLY) p.SetDevices(std::vector<int>{ 0 });
	auto cc = GenCryptoContext(p);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	cc->Enable(FHE);
	return cc;
}

static int run_boot(uint32_t slots, uint32_t budget) {
	Json j;
	j.ks("mode", "boot").kv("ring", RING).kv("depth", DEPTH).kv("slots", slots).kv("level_budget", budget);
	auto t0 = clk::now();
	auto cc = make_context();
	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	phase(j, "keygen", t0);

	t0 = clk::now();
	cc->EvalBootstrapSetup({ budget, budget }, { 0, 0 }, slots, 0);
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);
	phase(j, "boot_keygen", t0);
	j.kv("host_peak_gb_after_keygen", host_peak_gb());

	t0 = clk::now();
	cc->LoadContext(keys.publicKey);  // uploads every key: the 24 GB question
	cc->Synchronize();
	phase(j, "load_context", t0);

	std::mt19937_64 rng(0);
	std::uniform_real_distribution<double> U(-1.0, 1.0);  // same input range as e0_microbench.py
	std::vector<double> x(slots);
	for (auto& v : x) v = U(rng);
	auto pt = cc->MakeCKKSPackedPlaintext(x, 1, DEPTH - 1, nullptr, slots);
	auto ct = cc->Encrypt(keys.publicKey, pt);

	// 1 warm-up (first-touch allocations, lazy uploads), then 3 timed.
	auto out = cc->EvalBootstrap(ct);
	cc->Synchronize();
	j.kv("gpu_gb_after_warmup", gpu_used_gb());
	std::vector<double> ts;
	for (int r = 0; r < 3; ++r) {
		t0 = clk::now();
		out = cc->EvalBootstrap(ct);
		cc->Synchronize();
		ts.push_back(since(t0));
	}
	std::ostringstream tv;
	tv << "[" << ts[0] << ", " << ts[1] << ", " << ts[2] << "]";
	j.kv("bootstrap_s", tv.str()).kv("gpu_gb_peak_seen", gpu_used_gb());

	// Same definitions as e0_microbench.py: levels_left = depth - level,
	// precision = -log2(max abs error).
	j.kv("levels_left_after", int(DEPTH) - int(out->GetLevel()));
	Plaintext dec;
	cc->Decrypt(keys.secretKey, out, &dec);
	dec->SetLength(slots);
	auto got = dec->GetRealPackedValue();
	double err = 0;
	for (uint32_t i = 0; i < slots; ++i) err = std::max(err, std::abs(got[i] - x[i]));
	j.kv("max_abs_error", err).kv("precision_bits", err > 0 ? -std::log2(err) : 99.0);
	j.kv("host_peak_gb", host_peak_gb());
	std::cout << "RESULT " << j.str() << std::endl;
	return 0;
}

// Mirrors scripts/e0_matvec_bench.py: interleaved slot = dim*T + token,
// BSGS over the D diagonals, hoisted baby steps, diagonals encoded on the fly
// (pre-encoding 1024 diagonals at level 21 is ~24 GB: does not fit either).
static int run_matvec() {
	Json j;
	j.ks("mode", "matvec").kv("ring", RING).kv("depth", DEPTH).kv("level", MV_LEVEL).kv("D", D);
	auto t0 = clk::now();
	auto cc = make_context();
	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	const uint32_t T = RING / 2 / D;
	std::vector<int32_t> baby, rots;
	for (uint32_t b = 1; b < N1; ++b) baby.push_back(int32_t(b * T));
	rots = baby;
	for (uint32_t g = 1; g < N2; ++g) rots.push_back(int32_t(g * N1 * T));
	cc->EvalRotateKeyGen(keys.secretKey, rots);
	phase(j, "keygen", t0);
	t0 = clk::now();
	cc->LoadContext(keys.publicKey);
	cc->Synchronize();
	phase(j, "load_context", t0);

	std::mt19937_64 rng(0);
	std::uniform_real_distribution<double> U(-1.0, 1.0);
	std::vector<double> X(T * D), W(D * D);  // X[t*D+d], W[out*D+in]
	for (auto& v : X) v = U(rng);
	for (auto& v : W) v = U(rng) / std::sqrt(double(D));
	std::vector<double> xs(D * T);
	for (uint32_t d = 0; d < D; ++d)
		for (uint32_t t = 0; t < T; ++t) xs[d * T + t] = X[t * D + d];
	auto ptx = cc->MakeCKKSPackedPlaintext(xs, 1, MV_LEVEL);
	auto ct = cc->Encrypt(keys.publicKey, ptx);

	// diag_slots(W, i, shift): diagonal i (out j <- in j+i), rolled by +shift dims.
	auto diag = [&](uint32_t i, uint32_t shift) {
		std::vector<double> v(D * T);
		for (uint32_t j = 0; j < D; ++j) {
			uint32_t src = (j + D - shift) % D;  // np.roll(d, shift)[j] = d[j - shift]
			double w = W[src * D + (src + i) % D];
			for (uint32_t t = 0; t < T; ++t) v[j * T + t] = w;
		}
		return v;
	};

	double rot_s = 0, enc_s = 0, load_s = 0, mult_s = 0;
	auto total0 = clk::now();
	t0 = clk::now();
	auto pre = cc->EvalFastRotationPrecompute(ct);
	std::vector<Ciphertext<DCRTPoly>> bab{ ct };
	for (int32_t r : baby) bab.push_back(cc->EvalFastRotation(ct, r, 2 * RING, pre));
	cc->Synchronize();
	rot_s += since(t0);

	Ciphertext<DCRTPoly> out;
	for (uint32_t g = 0; g < N2; ++g) {
		Ciphertext<DCRTPoly> acc;
		for (uint32_t b = 0; b < N1; ++b) {
			auto t1 = clk::now();
			auto pt = cc->MakeCKKSPackedPlaintext(diag(g * N1 + b, g * N1), 1, MV_LEVEL);
			auto t2 = clk::now();
			cc->LoadPlaintext(pt);
			cc->Synchronize();
			auto t3 = clk::now();
			auto term = cc->EvalMult(bab[b], pt);
			acc = acc ? cc->EvalAdd(acc, term) : term;
			cc->Synchronize();
			enc_s += std::chrono::duration<double>(t2 - t1).count();
			load_s += std::chrono::duration<double>(t3 - t2).count();
			mult_s += since(t3);
		}
		t0 = clk::now();
		if (g) acc = cc->EvalRotate(acc, int32_t(g * N1 * T));
		out = out ? cc->EvalAdd(out, acc) : acc;
		cc->Synchronize();
		rot_s += since(t0);
	}
	double total = since(total0);
	j.kv("total_s", total).kv("rotate_s", rot_s).kv("encode_s", enc_s).kv("h2d_s", load_s).kv("mult_add_s", mult_s);
	j.kv("compute_only_s", rot_s + mult_s).kv("gpu_gb_peak_seen", gpu_used_gb());

	Plaintext dec;
	cc->Decrypt(keys.secretKey, out, &dec);
	dec->SetLength(D * T);
	auto got = dec->GetRealPackedValue();
	double err = 0;
	for (uint32_t t = 0; t < T; ++t)
		for (uint32_t o = 0; o < D; ++o) {
			double want = 0;
			for (uint32_t i = 0; i < D; ++i) want += W[o * D + i] * X[t * D + i];
			err = std::max(err, std::abs(got[o * T + t] - want));
		}
	j.kv("max_abs_error", err).kv("levels_used", int(out->GetLevel()) - int(MV_LEVEL));
	j.kv("host_peak_gb", host_peak_gb());
	std::cout << "RESULT " << j.str() << std::endl;
	return 0;
}

int main(int argc, char** argv) {
	std::string mode = argc > 1 ? argv[1] : "";
	try {
		if (mode == "boot" && argc > 2) return run_boot(std::stoul(argv[2]), argc > 3 ? std::stoul(argv[3]) : 4);
		if (mode == "matvec") return run_matvec();
	} catch (const std::exception& e) {
		std::cout << "RESULT {\"mode\": \"" << mode << "\", \"error\": \"" << e.what() << "\"}" << std::endl;
		return 2;
	}
	std::cerr << "usage: fides_e0 boot <slots> [budget] | fides_e0 matvec" << std::endl;
	return 1;
}
