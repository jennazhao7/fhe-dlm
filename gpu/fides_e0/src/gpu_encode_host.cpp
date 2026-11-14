// Host half of the GPU encoder: OpenFHE's own special FFT, so the rounded
// coefficients are bit-identical to CKKSPackedEncoding::Encode (64-bit build).
#include <cmath>
#include <complex>
#include <stdexcept>

#include "math/dftransform.h"  // openfhe/core

#include "gpu_encode.hpp"

std::vector<int64_t> e0::ckks_coeffs(const std::vector<double>& v, uint32_t ring, double sf) {
	std::vector<std::complex<double>> inv(v.begin(), v.end());
	lbcrypto::DiscreteFourierTransform::FFTSpecialInv(inv, ring * 2);
	const size_t s = inv.size();
	std::vector<int64_t> t(2 * s);
	for (size_t i = 0; i < s; ++i) {
		inv[i] *= sf;  // same op order as Encode: scale the complex value, then round
		const double re = inv[i].real(), im = inv[i].imag();
		// Encode falls back to an approximate split above 2^63; our weights are
		// O(1/sqrt(D)), so refuse rather than silently diverge from OpenFHE.
		if (std::abs(re) >= 0x1p62 || std::abs(im) >= 0x1p62)
			throw std::runtime_error("ckks_coeffs: coefficient exceeds 2^62; scale the weights down");
		t[i]     = std::llround(re);
		t[i + s] = std::llround(im);
	}
	return t;
}
