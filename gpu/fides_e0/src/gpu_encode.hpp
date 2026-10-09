// GPU-side CKKS encoding of period-D plaintexts for FIDESlib (HANDOFF §10).
//
// Job 1519942: the 2-key matvec computed in 1.4 s on the GPU but spent 92 s
// encoding diagonals with OpenFHE on the host and 41 s uploading them
// (23 MB per plaintext at N=2^17, level 21). A plaintext whose slot vector is
// periodic with period D is OpenFHE's sparse encoding with slots = D: its
// polynomial has only 2D nonzero coefficients, at stride N/(2D). So the host
// does the size-D special FFT + rounding (bit-identical to
// CKKSPackedEncoding::Encode), uploads 2D int64 per plaintext, and the GPU
// scatters them into every RNS limb and runs FIDESlib's own NTT -- the same
// transform FIDESlib uses on ciphertexts, so the result is the OpenFHE
// EVALUATION form a normal LoadPlaintext would have produced.
//
// Kept out of FIDESlib: this works on the device plaintext behind a public
// handle (CryptoContextImpl::GetDevicePlaintext). Plaintext limbs are
// "constant" limbs without the aux buffer the NTT kernels need, so the NTT
// runs on a scratch RNSPoly and the result is copied device-to-device.
#pragma once
#include <any>
#include <cstdint>
#include <memory>
#include <vector>

namespace e0 {

// Host: OpenFHE's CKKS encode of a length-`slots` real vector at scaling
// factor `sf` up to (not including) the per-prime reduction: returns the 2*slots
// signed coefficients that FitToNativeVector places at stride N/(2*slots).
std::vector<int64_t> ckks_coeffs(const std::vector<double>& v, uint32_t ring, double sf);

struct GpuEncoder;

// gpu_ctx: CryptoContextImpl::gpu (holds a FIDESlib::CKKS::Context).
// ref_dev_pt: a device plaintext already loaded at the target level; its level
// and scaling factor are what every encoded plaintext gets.
GpuEncoder* gpuenc_create(std::any& gpu_ctx, std::shared_ptr<void>& ref_dev_pt, uint32_t D);
double gpuenc_scaling_factor(const GpuEncoder* e);
// Upload all coefficient sets (each 2D int64) at once; returns seconds.
double gpuenc_upload(GpuEncoder* e, const std::vector<std::vector<int64_t>>& coeffs);
// Overwrite dst_dev_pt with coefficient set `idx` (scatter + NTT + copy). Synchronizes.
void gpuenc_encode(GpuEncoder* e, size_t idx, std::shared_ptr<void>& dst_dev_pt);
// Residue-by-residue comparison of two device plaintexts: number of differing entries.
uint64_t gpuenc_count_diff(GpuEncoder* e, std::shared_ptr<void>& a, std::shared_ptr<void>& b);
void gpuenc_destroy(GpuEncoder* e);

}  // namespace e0
