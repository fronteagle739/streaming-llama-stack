// SPDX-License-Identifier: MIT
#pragma once
#include <stdint.h>

#if defined(_WIN32) && defined(QFN_NATIVE_NVFP4_BUILD)
#define QFN_NATIVE_NVFP4_API __declspec(dllexport)
#elif defined(_WIN32)
#define QFN_NATIVE_NVFP4_API __declspec(dllimport)
#else
#define QFN_NATIVE_NVFP4_API
#endif
#ifdef __cplusplus
extern "C" {
#endif

// Independent packed-layout ABI1. Existing Basalt ABI/modes0..5 are unchanged.
// Geometry: E2560,F640,top10,512 logical experts. Expert blob2764816 bytes:
// role starts0,921600,1843200; each is adjacent-nibble E2M1[N,K/2] then
// unchanged UE4M3 scales[K/64,N,4]. The complete16-byte FP32 tail is retained.
// Mode6 is the DSV grouped NVFP4 calibrated path (all nt>=1,slots512), including
// final short prefill/append windows. At nt1..4 it also accepts arbitrary physical
// cache slots: selected packed blobs (including the entire tail) are copied to
// persistent workspace route groups; logical calibration IDs remain independent.
// Repeated physical placeholders are valid. Explicit modes2/5 remain available.
// Short mode6 uses fused SwiGLU + down-activation FP4 packing. Set
// QFN_NATIVE_NVFP4_FUSED_MID=1 before process admission for the fused candidate;
// 0 is the measured default. Other values fail workspace admission. Gate/up GEMMs keep
// separate calibrated activations. This policy does not change modes0..5.
// QFN_NATIVE_NVFP4_DECODE_RESIDUAL=1 opts short mode6 (nt1..4, including short
// final prefill) into two-component activation quantization for gate/up/down.
// Each low component quantizes F32 input minus reconstructed high input with
// the same role-specific calibration. Two existing grouped FP4 products are
// added BEFORE activation/weight globals and SwiGLU. This targets the F32 input,
// not byte-identical native-Q8 arithmetic; full-model quality is unqualified.
// Default0 preserves the original mode6 path. Only exact0/1 is accepted; with1,
// short non-mode6 workspace/run requests fail instead of ignoring the option.
// Long prefill remains the original single-component policy. Both fused-mid
// values remain supported. Extra scratch at nt1/2/3/4:381440/496640/611840/727040
// bytes, reserved before cache admission (may reduce the admitted cache slots).
// Per short FFN: three additional residual quantizers, grouped GEMMs and adds,
// two scale-padding clears; GEMM arithmetic doubles. No bank/cache data changes.
// Modes0..5 invert the packed layout losslessly on-device and execute the
// ORIGINAL mode in the adjacent qfn-basalt-nvfp4.dll. Small mode0/2/5 calls
// gather only selected blobs; explicit mode1/3/4 still requires the512 bank.
// QFN_NATIVE_NVFP4_DECODE_VECTOR=1 explicitly selects the DSV packed vector
// method for mode6 nt1..4: dynamic native Q8_1 activations, direct packed E2M1
// values/UE4M3-per16 weights, fused gate/up/SwiGLU, Q8_1 hidden and packed down.
// Physical slots, original logical IDs and all FP32 role tails are retained;
// no inverse, weight conversion, selected-expert copy or host readback occurs.
// Exported input_scale is still fully validated, but applies only to FP4
// activation arithmetic; this vector path uses each Q8_1 block's dynamic scale.
// Output stays UNWEIGHTED per-route. Long mode6 prefill keeps grouped TC math.
// Default0 leaves every existing mode available. Only exact0/1 is accepted;
// vector1 requires explicit mode6 at the caller and rejects residual1 as a
// conflicting activation arithmetic selection. FUSED_MID=1 fuses vector
// gate/up/SwiGLU; 0 retains separate gate/up outputs and a separate SwiGLU
// launch before identical Q8_1 hidden quantization. Both grouped middle-stage
// schedules also remain available. Qualification pending.
// Vector policy2 is the explicit F32 activation arm: original packed weight
// values/scales/globals, no activation quantization, same short mode6 extents.
// Both middle-stage schedules remain supported; policy0/1 are unchanged.
QFN_NATIVE_NVFP4_API int qfn_native_nvfp4_abi(void);
// Initializes original DLL, AOT modules and kernel attributes before capture.
QFN_NATIVE_NVFP4_API int qfn_native_nvfp4_prepare(void * stream);
// QFN_NATIVE_NVFP4_GROUP_PARALLEL=0/1 selects original warp / stable CTA256
// grouping for long mode6 only; default0, invalid values fail admission.
// Long requests (nt>4) in modes0..5 refuse flag1 at workspace/run admission;
// short requests retain their existing algorithms in every supported mode.
// Diagnostic grouping entry: ids[nt,10] must already contain valid distinct
// expert IDs0..511 per token. Writes order[nt*10] and bounds[513], without
// allocation/synchronization; prepare first, nt1..16384, parallel exactly0/1.
// Production run retains the full existing route/tail/calibration validation.
// QFN_NATIVE_NVFP4_SHARE_GU_INPUT=0/1 optionally reuses gate activation bytes
// for up in long mode6 only; default0 retains both quantizers. Flag1 requires
// bit-identical gate/up globals for ALL512 experts, independently validated
// on device each run. Both GEMMs and all role-specific output/weight globals,
// down quantization and arithmetic remain unchanged. No workspace/ABI change.
// Invalid policy or long modes0..5 with flag1 fail admission; short requests
// retain independent calibration and every original arithmetic mode.
QFN_NATIVE_NVFP4_API int qfn_native_nvfp4_group_plan(
    const int32_t * ids, int nt, int parallel, int32_t * order,
    int32_t * bounds, void * stream);
// Zero means invalid/unsupported geometry or unavailable original DLL.
// All storage (raw inverse, route maps, scale padding and arithmetic) included.
QFN_NATIVE_NVFP4_API uint64_t qfn_native_nvfp4_packed_workspace_bytes(
    int nt, uint32_t physical_slots, int mode);
// Lossless GPU layout proof/conversion using the SAME inverse kernel as run.
// selected_count=0: unpack the whole physical bank in slot order; all four ID
// pointers must be null. Otherwise physical_ids[selected_count] selects blobs,
// duplicates allowed; gathered_ids[r]=r and canonical_ids[r]=logical_ids[r], or
// physical_ids[r] when logical_ids is null. Both output maps are mandatory.
// raw_output spans (selected_count ? selected_count : physical_slots)*2764816 bytes,
// including all16 tail bytes. Canonical IDs are copied without reinterpretation.
// Packed/output bases must be16-byte aligned, maps4-byte aligned and disjoint;
// prepare first. No allocation, host readback, synchronization or bank mutation.
QFN_NATIVE_NVFP4_API int qfn_native_nvfp4_unpack(
    const uint8_t * packed, const int32_t * physical_ids, const int32_t * logical_ids,
    uint32_t physical_slots, uint32_t selected_count, uint8_t * raw_output,
    uint64_t raw_output_bytes, int32_t * gathered_ids, int32_t * canonical_ids,
    void * stream);
// x[nt,2560], physical_ids[nt,10], output UNWEIGHTED[nt,10,2560].
// logical_ids optionally supplies canonical calibration IDs for cache slots;
// when null, physical IDs are also logical IDs. calibration[512,3] is raw
// exported input_scale=amax/(6*448), mandatory for modes4/5/6; ALL1536 values
// are checked. Separate gate/up activation globals precede SwiGLU. Hidden H
// is quantized without s_down; down activation/weight globals follow GEMM.
// prepare must succeed in the same current CUDA context before run/capture.
// run allocates nothing, reads no device data on the host, never synchronizes
// and never mutates the packed bank/cache or calibration. Workspace must be
//256-byte aligned; other pointers retain the original alignment contract.
// Invalid device IDs/scales trap asynchronously. Caller spans are disjoint,
// persistent through graph replay. Resetting the context requires DLL reload.
QFN_NATIVE_NVFP4_API int qfn_native_nvfp4_run(
    const float * x, const int32_t * physical_ids, const int32_t * logical_ids,
    const uint8_t * packed, const float * calibration, uint32_t physical_slots,
    int nt, int mode, float * output, void * workspace, uint64_t workspace_bytes,
    void * stream);
#ifdef __cplusplus
}
#endif
