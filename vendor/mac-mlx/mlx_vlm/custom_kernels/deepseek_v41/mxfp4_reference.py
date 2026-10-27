"""Pure-Python decode audit; runnable without importing MLX or mlx_vlm."""
import argparse
import math
from pathlib import Path
import struct

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f32_bits(value):
    try:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    except OverflowError:
        return 0xFF800000 if value < 0 else 0x7F800000


def bf16_bits(value):
    bits = f32_bits(value)
    if bits & 0x7FFFFFFF > 0x7F800000:
        return (bits >> 16) | 0x40
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def decode(q, scale_byte, bf16=False):
    """Mirror the Metal half-bit decode and MLX E8M0 conversion."""
    magnitude = struct.unpack("<e", struct.pack("<H", (q & 7) << 9))[0] * 16384
    value = -magnitude if q & 8 else magnitude
    scale = f32(0x400000 if scale_byte == 0 else scale_byte << 23)
    value *= scale
    return f32(bf16_bits(value) << 16) if bf16 else f32(f32_bits(value))


def verify(mlx_kernels):
    root = Path(mlx_kernels)
    fp4 = (root / "fp4.h").read_text()
    fp8 = (root / "fp8.h").read_text()
    quant = (root / "fp_quantized.h").read_text()
    # Fail if the referenced upstream conversion changes; no assumed E8M0 NaN.
    for token in ("ushort((bits & 7) << 9)", "converted *= 16384.0", "bits & 8 ? -converted : converted"):
        assert token in fp4, token
    for token in ("bits == 0 ? 0x40 :", "bits == 0 ? 0x400000 :", "bits) << 23", "bits) << 7"):
        assert token in fp8, token
    assert "scale * Dequantize<4, U>{}(w)" in quant
    metal = Path(__file__).with_name("grouped_gemm.metal").read_text()
    for token in ("s == 0 ? 0x00400000u : s << 23", "ushort((q & 7) << 9)",
                  "* 16384.0f", "(q & 8) ? -v : v", "T(scale * T(v))"):
        assert token in metal, token
    count = 0
    for q in range(16):
        for s in range(256):
            scale = math.ldexp(1.0, s - 127) if s < 255 else math.inf
            signed = math.copysign(E2M1[q & 7], -1 if q & 8 else 1)
            for bf16 in (False, True):
                expected = signed * scale
                bits = bf16_bits(expected) << 16 if bf16 else f32_bits(expected)
                actual = decode(q, s, bf16)
                if math.isnan(expected):
                    assert math.isnan(actual)
                else:
                    assert f32_bits(actual) == bits, (q, s, bf16)
                count += 1
    print(f"PASS: {count} E2M1/E8M0 FP32/BF16 combinations; MLX source constants match")
    print("Includes signed zero, subnormals, overflow, scale 255 infinity and 0*inf NaN.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-kernels", required=True)
    verify(parser.parse_args().mlx_kernels)
