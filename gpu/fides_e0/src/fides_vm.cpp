// fides_vm: the encrypted server of the FHE-DLM pass as a small op interpreter
// over FIDESlib (HANDOFF §12). scripts/e5_pass.py drives it over stdin/stdout:
// one command per line, one reply line "OK {json}" or "ERR message". Vectors
// travel as raw little-endian files (float64 slots, float32 weights).
//
//   fides_vm <logn> <depth> <D> <gpus> <rot,rot,...> <boot 0|1>
//
// Layout is token-major, slot = token*D + dim, T = slots/D tokens per
// ciphertext (fides_e0 matvec_enc, job 1520180). Commands:
//   enc <r> <file> <level>            encrypt SLOTS float64 at `level`
//   dec <r> <file>                    decrypt to SLOTS float64
//   matvec <n_out> <n_in> <wfile> <biasfile|-> <in...> <out...>
//          y_o = sum_i W[o][i] x_i + bias_o per token; W is float32
//          [n_out][n_in][D][D] (row = output dim), bias float64 [n_out][D].
//          Diagonals are encoded on the GPU (gpu_encode.hpp); the bias enters as
//          bias (.) Enc(1), so the device only ever does ct x pt and ct + ct.
//   add|sub|mul <r> <a> <b>   sq <r> <a>   rot <r> <a> <k>
//   mulc|addc <r> <a> <const>
//   mulpt|addpt <r> <a> <file> <period> [deg]   period D (sparse) or SLOTS
//   cheb <r> <a> <lo> <hi> <coeff file> <n>     EvalChebyshevSeries
//   mov <r> <a>                       alias (ops never modify their inputs)
//   rescale <r> <a>                   pending rescale now (deg 2 -> 1)
//   tolevel <r> <a> <level>           a + Enc(0 at level): drops a to that level, deg 1
//   boot <r> <a>   info <r>   free <r>   mem   quit
// matvec takes an optional trailing "kb=<x>": the bias file then holds bias/x and
// the server multiplies it into Enc(x), so encoded bias values stay <= 1.
// matvec, mulpt and addpt rescale a deg-2 input first and encode at the
// resulting level, so the device never adjusts a plaintext per multiplication.
// The process holds the secret key too (enc/dec are the client's side); the
// driver accounts client and server time separately.
//
// CPU dry run of every code path except the GPU encoder (host encoding
// instead):  FIDES_E0_CPU=1 fides_vm 12 2 64 1 1,8,-64 0
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <sys/resource.h>

#include <fideslib.hpp>
#include "openfhe.h"  // CPU dry run only, see mul_pt

#include "gpu_encode.hpp"

using namespace fideslib;
using Ct = Ciphertext<DCRTPoly>;

static const bool CPU_ONLY = std::getenv("FIDES_E0_CPU") != nullptr;
static uint32_t RING, DEPTH, D, N1, N2, SLOTS;

using clk = std::chrono::steady_clock;
static double since(clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); }

static double gpu_used_gb() {
	if (CPU_ONLY) return -1;
	int n = 0, cur = 0;
	if (cudaGetDeviceCount(&n) != cudaSuccess) return -1;
	cudaGetDevice(&cur);
	double worst = 0;
	for (int d = 0; d < n; ++d) {
		size_t free_b = 0, total_b = 0;
		cudaSetDevice(d);
		if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) worst = std::max(worst, double(total_b - free_b) / (1ull << 30));
	}
	cudaSetDevice(cur);
	return worst;
}
static double host_peak_gb() {
	rusage u{};
	getrusage(RUSAGE_SELF, &u);
	return double(u.ru_maxrss) / (1 << 20);
}

// FIDESlib 2.1.3's CPU fallback of EvalMult(ct, pt) any_casts the plaintext to
// the wrong type (bad any_cast); the dry run calls OpenFHE itself. GPU path untouched.
static Ct mul_pt(CryptoContext<DCRTPoly>& cc, const Ct& ct, Plaintext& pt) {
	if (!CPU_ONLY) return cc->EvalMult(ct, pt);
	auto& c  = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
	auto& x  = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
	auto& q  = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
	Ct out   = std::make_shared<CiphertextImpl<DCRTPoly>>(CryptoContext<DCRTPoly>(cc));
	out->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(c->EvalMult(x, q));
	return out;
}

template <class T> static std::vector<T> read_raw(const std::string& path, size_t n) {
	std::ifstream f(path, std::ios::binary);
	if (!f) throw std::runtime_error("cannot open " + path);
	std::vector<T> v(n);
	f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T)));
	if (size_t(f.gcount()) != n * sizeof(T)) throw std::runtime_error("short read on " + path);
	return v;
}
static void write_raw(const std::string& path, const std::vector<double>& v) {
	std::ofstream f(path, std::ios::binary);
	if (!f) throw std::runtime_error("cannot write " + path);
	f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(double)));
}

struct Vm {
	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
	std::map<std::string, Ct> reg;
	std::map<std::string, std::vector<float>> wcache;
	struct Enc {
		e0::GpuEncoder* e = nullptr;
		Plaintext ref, buf;
		double sf = 0;
	};
	std::map<uint32_t, Enc> encs;   // GPU encoder per level
	std::map<std::pair<uint32_t, double>, Ct> consts;   // Enc(c,...,c) per (level, c)
	bool fft_warm = false;

	Ct& get(const std::string& r) {
		auto it = reg.find(r);
		if (it == reg.end()) throw std::runtime_error("no register " + r);
		return it->second;
	}
	Enc& encoder(uint32_t level) {
		auto it = encs.find(level);
		if (it != encs.end()) return it->second;
		Enc en;
		std::vector<double> z(D, 0.0);
		z[0] = 1.0;
		en.ref = cc->MakeCKKSPackedPlaintext(z, 1, level, nullptr, D);
		cc->LoadPlaintext(en.ref);
		en.buf = cc->MakeCKKSPackedPlaintext(z, 1, level, nullptr, D);
		cc->LoadPlaintext(en.buf);
		en.e  = e0::gpuenc_create(cc->gpu, cc->GetDevicePlaintext(en.ref->gpu), D);
		en.sf = e0::gpuenc_scaling_factor(en.e);
		return encs[level] = en;
	}
	Ct& konst(uint32_t level, double c) {
		auto key = std::make_pair(level, c);
		auto it  = consts.find(key);
		if (it != consts.end()) return it->second;
		std::vector<double> v(SLOTS, c);
		auto pt = cc->MakeCKKSPackedPlaintext(v, 1, level);
		return consts[key] = cc->Encrypt(keys.publicKey, pt);
	}
	// FLEXIBLEAUTO leaves a product at deg 2; do the pending rescale so plaintexts
	// can be encoded at the level the multiplication will really run at.
	Ct deg1(const Ct& x) { return x->GetNoiseScaleDeg() > 1 ? cc->Rescale(x) : x; }
	Plaintext host_pt(const std::vector<double>& v, uint32_t level, uint32_t period, size_t deg = 1) {
		return period == SLOTS ? cc->MakeCKKSPackedPlaintext(v, deg, level) : cc->MakeCKKSPackedPlaintext(v, deg, level, nullptr, period);
	}
};

// y_o = sum_i W[o][i] x_i (+ bias_o). Diagonal i = g*N1 + b of a block splits by
// the mask j < D - i into rot(x, i) and rot(rot(x, -D), i) halves (the rotation
// crosses token boundaries); each half runs its own Horner pass over g, so only
// N1 baby-step ciphertexts are live. Plaintexts are pre-rolled by g*N1.
static std::string do_matvec(Vm& vm, uint32_t n_out, uint32_t n_in, const std::vector<float>& W, const std::vector<double>* bias, double kb,
                             const std::vector<std::string>& ins, const std::vector<std::string>& outs) {
	auto& cc = vm.cc;
	std::vector<Ct> xin;
	for (auto& r : ins) xin.push_back(vm.deg1(vm.get(r)));
	const uint32_t level = uint32_t(xin[0]->GetLevel());
	for (auto& x : xin)
		if (x->GetLevel() != level) throw std::runtime_error("matvec: inputs at different levels");
	Vm::Enc* en = CPU_ONLY ? nullptr : &vm.encoder(level);
	const double sf = en ? en->sf : 0;
	double rot_s = 0, fft_s = 0, up_s = 0, mult_s = 0;
	uint32_t n_mults = 0, n_rots = 0, n_skipped = 0;
	std::vector<Ct> acc_out(n_out);

	auto block = [&](uint32_t o, uint32_t i) { return W.data() + (size_t(o) * n_in + i) * D * D; };
	auto pt_vec = [&](const float* Wb, uint32_t g, uint32_t b, int half, bool& any) {
		const uint32_t i = g * N1 + b, shift = g * N1;
		std::vector<double> v(D, 0.0);
		any = false;
		for (uint32_t jj = 0; jj < D; ++jj) {
			const uint32_t src = (jj + D - shift) % D;
			if ((src < D - i) == (half == 0)) {
				v[jj] = Wb[size_t(src) * D + (src + i) % D];
				any |= v[jj] != 0.0;
			}
		}
		return v;
	};

	for (uint32_t i = 0; i < n_in; ++i) {
		for (int h = 0; h < 2; ++h) {
			auto t0 = clk::now();
			std::vector<Ct> bab{ h ? cc->EvalRotate(xin[i], -int32_t(D)) : xin[i] };
			for (uint32_t b = 1; b < N1; ++b) bab.push_back(cc->EvalRotate(bab.back(), 1));
			cc->Synchronize();
			rot_s += since(t0);
			n_rots += N1 - 1 + h;
			for (uint32_t o = 0; o < n_out; ++o) {
				const float* Wb = block(o, i);
				// host: every plaintext vector of this (block, half); empty ones are skipped
				t0 = clk::now();
				std::vector<std::vector<double>> vecs(D);
				std::vector<char> live(D, 0);
				bool block_any = false;
				for (uint32_t g = 0; g < N2; ++g)
					for (uint32_t b = 0; b < N1; ++b) {
						bool any = false;
						vecs[g * N1 + b] = pt_vec(Wb, g, b, h, any);
						live[g * N1 + b] = any;
						block_any |= any;
					}
				if (!block_any) continue;
				if (en) {
					std::vector<std::vector<int64_t>> coeffs(D);
					if (!vm.fft_warm) {   // OpenFHE builds its FFT tables on first use
						e0::ckks_coeffs(vecs[0], RING, sf);
						vm.fft_warm = true;
					}
					std::string fft_err;
#pragma omp parallel for schedule(dynamic, 16)
					for (int k = 0; k < int(D); ++k) {
						try {
							coeffs[k] = live[k] ? e0::ckks_coeffs(vecs[k], RING, sf) : std::vector<int64_t>(2 * D, 0);
						} catch (const std::exception& e) {
#pragma omp critical
							fft_err = e.what();
						}
					}
					if (!fft_err.empty()) throw std::runtime_error(fft_err);
					fft_s += since(t0);
					up_s += e0::gpuenc_upload(en->e, coeffs);
				} else
					fft_s += since(t0);
				t0 = clk::now();
				Ct half;
				for (uint32_t g = N2; g-- > 0;) {
					Ct acc;
					for (uint32_t b = 0; b < N1; ++b) {
						const uint32_t k = g * N1 + b;
						if (!live[k]) {
							++n_skipped;
							continue;
						}
						Ct term;
						if (en) {
							e0::gpuenc_encode(en->e, k, cc->GetDevicePlaintext(en->buf->gpu));
							term = cc->EvalMult(bab[b], en->buf);
						} else {
							auto pt = cc->MakeCKKSPackedPlaintext(vecs[k], 1, level, nullptr, D);
							term    = mul_pt(cc, bab[b], pt);
						}
						acc = acc ? cc->EvalAdd(acc, term) : term;
						++n_mults;
					}
					if (half) {
						half = cc->EvalRotate(half, int32_t(N1));
						++n_rots;
						if (acc) half = cc->EvalAdd(half, acc);
					} else if (acc)
						half = acc;
				}
				cc->Synchronize();
				mult_s += since(t0);
				if (half) acc_out[o] = acc_out[o] ? cc->EvalAdd(acc_out[o], half) : half;
			}
		}
	}
	if (bias) {
		auto t0 = clk::now();
		for (uint32_t o = 0; o < n_out; ++o) {
			std::vector<double> v(bias->begin() + size_t(o) * D, bias->begin() + size_t(o + 1) * D);
			bool any = false;
			for (double x : v) any |= x != 0.0;
			if (!any) continue;
			Ct term;
			if (en) {
				e0::gpuenc_upload(en->e, { e0::ckks_coeffs(v, RING, sf) });
				e0::gpuenc_encode(en->e, 0, cc->GetDevicePlaintext(en->buf->gpu));
				term = cc->EvalMult(vm.konst(level, kb), en->buf);
			} else {
				auto pt = cc->MakeCKKSPackedPlaintext(v, 1, level, nullptr, D);
				term    = mul_pt(cc, vm.konst(level, kb), pt);
			}
			acc_out[o] = acc_out[o] ? cc->EvalAdd(acc_out[o], term) : term;
			++n_mults;
		}
		cc->Synchronize();
		mult_s += since(t0);
	}
	for (uint32_t o = 0; o < n_out; ++o) {
		if (!acc_out[o]) throw std::runtime_error("matvec: output block " + std::to_string(o) + " is identically zero");
		vm.reg[outs[o]] = acc_out[o];
	}
	std::ostringstream s;
	s << "\"rot_s\": " << rot_s << ", \"host_fft_s\": " << fft_s << ", \"upload_s\": " << up_s << ", \"mult_s\": " << mult_s
	  << ", \"pt_mults\": " << n_mults << ", \"rots\": " << n_rots << ", \"skipped\": " << n_skipped << ", \"level\": " << level;
	return s.str();
}

int main(int argc, char** argv) {
	if (argc < 7) {
		std::cerr << "usage: fides_vm <logn> <depth> <D> <gpus> <rot,rot,...> <boot 0|1>" << std::endl;
		return 1;
	}
	RING  = 1u << std::atoi(argv[1]);
	DEPTH = uint32_t(std::atoi(argv[2]));
	D     = uint32_t(std::atoi(argv[3]));
	SLOTS = RING / 2;
	N1    = 1u << (int(std::log2(D)) / 2);
	N2    = D / N1;
	const int ngpu  = std::atoi(argv[4]);
	const bool boot = std::atoi(argv[6]) != 0;
	std::vector<int32_t> rots;
	{
		std::stringstream ss(argv[5]);
		std::string tok;
		while (std::getline(ss, tok, ','))
			if (!tok.empty()) rots.push_back(std::stoi(tok));
	}
	Vm vm;
	try {
		auto t0 = clk::now();
		CCParams<CryptoContextCKKSRNS> p;
		p.SetSecretKeyDist(UNIFORM_TERNARY);
		p.SetSecurityLevel(RING < (1u << 15) ? HEStd_NotSet : HEStd_128_classic);
		p.SetRingDim(RING);
		p.SetScalingModSize(59);
		p.SetFirstModSize(60);
		p.SetScalingTechnique(FLEXIBLEAUTO);
		p.SetKeySwitchTechnique(HYBRID);
		p.SetMultiplicativeDepth(DEPTH);
		if (!CPU_ONLY) {
			std::vector<int> devs;
			for (int d = 0; d < ngpu; ++d) devs.push_back(d);
			p.SetDevices(std::move(devs));
		}
		vm.cc = GenCryptoContext(p);
		for (auto f : { PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE }) vm.cc->Enable(f);
		vm.keys = vm.cc->KeyGen();
		vm.cc->EvalMultKeyGen(vm.keys.secretKey);
		if (boot) {
			vm.cc->EvalBootstrapSetup({ 4, 4 }, { 0, 0 }, SLOTS, 0);
			vm.cc->EvalBootstrapKeyGen(vm.keys.secretKey, SLOTS);
		}
		if (!rots.empty()) vm.cc->EvalRotateKeyGen(vm.keys.secretKey, rots);
		const double keygen_s = since(t0);
		t0                    = clk::now();
		vm.cc->LoadContext(vm.keys.publicKey);
		vm.cc->Synchronize();
		std::cout << "OK {\"ready\": true, \"ring\": " << RING << ", \"depth\": " << DEPTH << ", \"D\": " << D << ", \"slots\": " << SLOTS
		          << ", \"gpus\": " << ngpu << ", \"rot_keys\": " << rots.size() << ", \"boot\": " << boot << ", \"keygen_s\": " << keygen_s
		          << ", \"load_s\": " << since(t0) << ", \"gpu_gb\": " << gpu_used_gb() << ", \"host_gb\": " << host_peak_gb() << "}" << std::endl;
	} catch (const std::exception& e) {
		std::cout << "ERR setup: " << e.what() << std::endl;
		return 2;
	}

	auto& cc = vm.cc;
	std::string line;
	while (std::getline(std::cin, line)) {
		std::istringstream in(line);
		std::string op;
		in >> op;
		if (op.empty()) continue;
		if (op == "quit") break;
		auto t0 = clk::now();
		std::ostringstream extra;
		try {
			if (op == "enc") {
				std::string r, file;
				uint32_t level;
				in >> r >> file >> level;
				auto v    = read_raw<double>(file, SLOTS);
				auto pt   = cc->MakeCKKSPackedPlaintext(v, 1, level);
				vm.reg[r] = cc->Encrypt(vm.keys.publicKey, pt);
			} else if (op == "dec") {
				std::string r, file;
				in >> r >> file;
				Ct c = vm.get(r);
				Plaintext pt;
				cc->Decrypt(vm.keys.secretKey, c, &pt);
				pt->SetLength(SLOTS);
				write_raw(file, pt->GetRealPackedValue());
			} else if (op == "matvec") {
				uint32_t n_out, n_in;
				std::string wfile, bfile;
				in >> n_out >> n_in >> wfile >> bfile;
				std::vector<std::string> ins(n_in), outs(n_out);
				for (auto& s : ins) in >> s;
				for (auto& s : outs) in >> s;
				if (!in) throw std::runtime_error("matvec: bad arguments");
				double kb = 1.0;
				std::string opt;
				if (in >> opt && opt.rfind("kb=", 0) == 0) kb = std::stod(opt.substr(3));
				auto it = vm.wcache.find(wfile);
				if (it == vm.wcache.end()) it = vm.wcache.emplace(wfile, read_raw<float>(wfile, size_t(n_out) * n_in * D * D)).first;
				std::vector<double> bias;
				if (bfile != "-") bias = read_raw<double>(bfile, size_t(n_out) * D);
				extra << ", " << do_matvec(vm, n_out, n_in, it->second, bfile == "-" ? nullptr : &bias, kb, ins, outs);
			} else if (op == "add" || op == "sub" || op == "mul") {
				std::string r, a, b;
				in >> r >> a >> b;
				vm.reg[r] = op == "add" ? cc->EvalAdd(vm.get(a), vm.get(b)) : op == "sub" ? cc->EvalSub(vm.get(a), vm.get(b)) : cc->EvalMult(vm.get(a), vm.get(b));
			} else if (op == "sq") {
				std::string r, a;
				in >> r >> a;
				vm.reg[r] = cc->EvalSquare(vm.get(a));
			} else if (op == "rot") {
				std::string r, a;
				int32_t k;
				in >> r >> a >> k;
				vm.reg[r] = cc->EvalRotate(vm.get(a), k);
			} else if (op == "mulc" || op == "addc") {
				std::string r, a;
				double c;
				in >> r >> a >> c;
				vm.reg[r] = op == "mulc" ? cc->EvalMult(vm.get(a), c) : cc->EvalAdd(vm.get(a), c);
			} else if (op == "mulpt" || op == "addpt") {
				std::string r, a, file;
				uint32_t period;
				size_t deg = 1;
				in >> r >> a >> file >> period;
				if (!(in >> deg)) deg = 1;
				auto v   = read_raw<double>(file, period);
				Ct x     = vm.deg1(vm.get(a));
				auto pt  = vm.host_pt(v, uint32_t(x->GetLevel()), period, deg);
				vm.reg[r] = op == "mulpt" ? mul_pt(cc, x, pt) : cc->EvalAdd(x, pt);
			} else if (op == "cheb") {
				std::string r, a, file;
				double lo, hi;
				size_t n;
				in >> r >> a >> lo >> hi >> file >> n;
				auto c    = read_raw<double>(file, n);
				vm.reg[r] = cc->EvalChebyshevSeries(vm.get(a), c, lo, hi);
			} else if (op == "mov") {
				std::string r, a;
				in >> r >> a;
				vm.reg[r] = vm.get(a);
			} else if (op == "rescale") {
				std::string r, a;
				in >> r >> a;
				vm.reg[r] = vm.deg1(vm.get(a));
			} else if (op == "tolevel") {
				std::string r, a;
				uint32_t level;
				in >> r >> a >> level;
				vm.reg[r] = cc->EvalAdd(vm.get(a), vm.konst(level, 0.0));
			} else if (op == "boot") {
				std::string r, a;
				in >> r >> a;
				vm.reg[r] = cc->EvalBootstrap(vm.get(a));
			} else if (op == "info") {
				std::string r;
				in >> r;
				extra << ", \"level\": " << vm.get(r)->GetLevel() << ", \"deg\": " << vm.get(r)->GetNoiseScaleDeg();
			} else if (op == "free") {
				std::string r;
				while (in >> r) vm.reg.erase(r);
			} else if (op == "mem") {
				extra << ", \"gpu_gb\": " << gpu_used_gb() << ", \"host_gb\": " << host_peak_gb() << ", \"regs\": " << vm.reg.size();
			} else
				throw std::runtime_error("unknown op " + op);
			cc->Synchronize();
			std::cout << "OK {\"op\": \"" << op << "\", \"s\": " << since(t0) << extra.str() << "}" << std::endl;
		} catch (const std::exception& e) {
			std::cout << "ERR " << op << ": " << e.what() << std::endl;
		}
	}
	for (auto& [l, en] : vm.encs) e0::gpuenc_destroy(en.e);
	return 0;
}
