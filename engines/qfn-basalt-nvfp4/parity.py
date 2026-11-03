"""Guarded, opt-in CUDA parity for the QFN bridge (no executable or model download).

Run through the task's existing process guard. --cpu-only never loads CUDA.
--allow-gpu additionally requires the coordinator's exclusive lease marker.
Sparse CUDA VMM maps only selected expert pages, including real slot 20001.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1].parent
E, F, K, BLOB, TAIL = 2560, 640, 10, 2764816, 2764800
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
UE = np.array([(i & 7) * 2.0**-9 if i < 8 else
               (1 + (i & 7) / 8) * 2.0**((i >> 3) - 7) for i in range(127)], dtype=np.float32)


def progress(message):
    with (ROOT / "QFNBASALT_PROGRESS.log").open("a", encoding="utf-8") as out:
        out.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + message + "\n")


def fixture(slot):
    rng = np.random.default_rng(0x51464E + slot)
    parts = []
    for rows, cols in ((F, E), (F, E), (E, F)):
        b = rng.integers(0, 256, (rows, cols // 64, 36), dtype=np.uint8)
        # Varied finite UE4M3 subscales, including exact zero blocks.
        b[:, :, :4] = rng.integers(16, 25, b[:, :, :4].shape, dtype=np.uint8)
        b[::127, ::11, 0] = 0
        parts.append(b.tobytes())
    i = slot % 17
    parts.append(np.array([.125 + i / 128, .1875 + i / 256, .25 + i / 512, 0], "<f4").tobytes())
    result = b"".join(parts)
    assert len(result) == BLOB
    return result


def matrix(blob, rows, cols, offset):
    b = np.frombuffer(blob, np.uint8, count=rows * cols // 64 * 36, offset=offset).reshape(rows, -1, 36)
    q = b[:, :, 4:].reshape(rows, -1, 4, 8)
    lo, hi = q & 15, q >> 4
    codes = np.concatenate((lo, hi), axis=-1)
    v = FP4[codes & 7] * np.where(codes & 8, -1.0, 1.0)
    return (v * UE[b[:, :, :4]][:, :, :, None]).reshape(rows, cols)


def q8(x):
    x = np.asarray(x, np.float32).reshape(-1, 32)
    d = np.minimum(np.max(np.abs(x), axis=1) / np.float32(127), np.float32(65504))
    z = np.divide(x, d[:, None], out=np.zeros_like(x), where=d[:, None] != 0)
    q = np.copysign(np.floor(np.abs(z) + np.float32(.5)), z).clip(-127, 127)
    return (q * d.astype(np.float16).astype(np.float32)[:, None]).reshape(-1).astype(np.float64)


def nearest_even(values, table):
    hi = np.searchsorted(table, values).clip(0, len(table) - 1)
    lo = np.maximum(hi - 1, 0)
    dl, dh = values - table[lo], table[hi] - values
    pick_lo = (dl < dh) | ((dl == dh) & ((lo & 1) == 0))
    return table[np.where(pick_lo, lo, hi)]


def a4(x):
    x = np.asarray(x, np.float32).reshape(-1, 16)
    am = np.max(np.abs(x))
    g = am * np.float32(1 / (6 * 448))
    if not g:
        return np.zeros(x.size, np.float64)
    sd = nearest_even(np.max(np.abs(x), axis=1) * (np.float32(1) / g) * np.float32(1 / 6), UE)
    scale = sd * g
    inv = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=scale != 0)
    z = x * inv[:, None]
    return (np.copysign(nearest_even(np.abs(z), FP4), z) * scale[:, None]).reshape(-1).astype(np.float64)


def reference(x, blob, mode):
    sg, su, sd, _ = np.frombuffer(blob, "<f4", count=4, offset=TAIL)
    xx = a4(x) if mode == 1 else q8(x)
    g = (matrix(blob, F, E, 0) @ xx).astype(np.float32) * sg
    u = (matrix(blob, F, E, 921600) @ xx).astype(np.float32) * su
    h = (g / (np.float32(1) + np.exp(-g)) * u).astype(np.float32)
    if mode == 1:
        h *= sd
    y = matrix(blob, E, F, 1843200) @ (a4(h) if mode == 1 else q8(h))
    return y if mode == 1 else y * sd


def cpu_checks():
    max_native = (2**31 - 1) // (K * F // 256)
    assert max_native == 85899345 and 25 * max_native <= 2**31 - 1 < 25 * (max_native + 1)
    p = json.loads((HERE / "PROVENANCE.json").read_text())
    # Receipts retain exact original bytes, including upstream license files.
    entries = p["files"]
    if isinstance(entries, dict):
        entries = [dict(path=k, **v) if isinstance(v, dict) else dict(path=k, sha256=v) for k, v in entries.items()]
    for f in entries:
        rel = f.get("path", f.get("upstream_path"))
        actual = HERE / "vendor/basalt" / rel
        assert hashlib.sha256(actual.read_bytes()).hexdigest() == f["sha256"], rel
    blob = fixture(20001)
    assert matrix(blob, F, E, 0).shape == (F, E)
    zero = np.zeros(E, np.float32)
    for mode in (1, 2):
        assert np.array_equal(reference(zero, blob, mode), zero)
    assert np.array_equal(q8(np.zeros(32, np.float32)), np.zeros(32))
    assert np.array_equal(a4(np.zeros(16, np.float32)), np.zeros(16))
    assert np.isfinite(reference(np.random.default_rng(7).normal(0, .1, E).astype(np.float32), blob, 2)).all()
    return {"provenance_files": len(entries), "fixture_bytes": len(blob), "cpu_fixture_checks": "PASS"}


def abi_checks():
    # Loading cudart imports does not initialize a CUDA context. Only these two
    # exports run: pure host ABI/query; never prepare/run/a CUDA runtime function.
    home = Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"))
    paths = [p for p in (home / "bin", home / "bin/x64") if p.is_dir()]
    directories = [os.add_dll_directory(str(p)) for p in paths]
    try:
        lib = C.CDLL(str(HERE.parent.parent / "bin/qfn-basalt-nvfp4.dll"))
        abi = bind(lib, "qfn_basalt_nvfp4_abi", [])
        query = bind(lib, "qfn_basalt_nvfp4_workspace_bytes", [I, I, C.c_uint, I], U64)
        assert abi() == 1
        assert query(85899345, K, 20002, 2) > 0
        assert query(85899346, K, 20002, 2) == 0
        assert query((2**31 - 1) // K, K, 512, 0) == query(16384, K, 512, 0)
        assert query(2048, K, 512, 0) == 284652032
        assert query(16384, K, 512, 0) == 2193691136
        return {"host_only_dll_query": "PASS", "native_max_tokens": 85899345}
    finally:
        for directory in directories: directory.close()


P, U64, SZ, I = C.c_void_p, C.c_uint64, C.c_size_t, C.c_int


def bind(lib, name, args, result=I):
    f = getattr(lib, name)
    f.argtypes, f.restype = args, result
    return f


class CUDA:
    def __init__(self):
        home = Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"))
        candidates = sorted((home / "bin").glob("cudart64_*.dll")) + sorted((home / "bin/x64").glob("cudart64_*.dll"))
        if len(candidates) != 1:
            raise RuntimeError(f"Need exactly one CUDA runtime: {candidates}")
        self.dll_dir = os.add_dll_directory(str(candidates[0].parent))
        self.lib = C.WinDLL(str(candidates[0]))
        for name, args in {
            "cudaMalloc": [C.POINTER(P), SZ], "cudaFree": [P],
            "cudaHostAlloc": [C.POINTER(P), SZ, C.c_uint], "cudaFreeHost": [P],
            "cudaMemcpyAsync": [P, P, SZ, I, P], "cudaStreamCreateWithFlags": [C.POINTER(P), C.c_uint],
            "cudaStreamDestroy": [P], "cudaStreamSynchronize": [P],
            "cudaStreamBeginCapture": [P, I], "cudaStreamEndCapture": [P, C.POINTER(P)],
            "cudaGraphInstantiateWithFlags": [C.POINTER(P), P, U64],
            "cudaGraphLaunch": [P, P], "cudaGraphExecDestroy": [P], "cudaGraphDestroy": [P],
        }.items():
            setattr(self, name, bind(self.lib, name, args))
        self.error = bind(self.lib, "cudaGetErrorString", [I], C.c_char_p)
        self.stream = P()
        self.ok(self.cudaStreamCreateWithFlags(C.byref(self.stream), 1))
        self.device, self.host = [], []
        self.device_bytes, self.vmm_bytes = 0, 0

    def ok(self, status):
        if status:
            raise RuntimeError(f"CUDA {status}: {self.error(status).decode()}")

    def malloc(self, size):
        if self.device_bytes + self.vmm_bytes + size > 12 * 2**30:
            raise MemoryError("Explicit CUDA physical-allocation budget is 12GiB")
        p = P()
        self.ok(self.cudaMalloc(C.byref(p), size))
        self.device.append(p)
        self.device_bytes += size
        return p

    def pinned(self, shape, dtype):
        dtype = np.dtype(dtype)
        p = P()
        self.ok(self.cudaHostAlloc(C.byref(p), int(np.prod(shape)) * dtype.itemsize, 0))
        self.host.append(p)
        a = np.ctypeslib.as_array((C.c_ubyte * (int(np.prod(shape)) * dtype.itemsize)).from_address(p.value))
        return a.view(dtype).reshape(shape)

    def copy(self, dst, src, size, kind):
        self.ok(self.cudaMemcpyAsync(dst, src, size, kind, self.stream))

    def sync(self):
        self.ok(self.cudaStreamSynchronize(self.stream))

    def close(self):
        for p in reversed(self.device): self.ok(self.cudaFree(p))
        for p in reversed(self.host): self.ok(self.cudaFreeHost(p))
        self.ok(self.cudaStreamDestroy(self.stream))


class Location(C.Structure):
    _fields_ = [("type", I), ("id", I)]


class AllocFlags(C.Structure):
    _fields_ = [("compression", C.c_ubyte), ("rdma", C.c_ubyte), ("usage", C.c_ushort), ("reserved", C.c_ubyte * 4)]


class AllocProp(C.Structure):
    _fields_ = [("type", I), ("handle_types", I), ("location", Location), ("win32", P), ("flags", AllocFlags)]


class Access(C.Structure):
    _fields_ = [("location", Location), ("flags", I)]


class SparseBank:
    """Real high slot addresses: virtual reservation, only selected expert pages physically mapped."""
    def __init__(self, cuda, slots, blobs):
        self.c = cuda
        self.lib = C.WinDLL("nvcuda.dll")
        for name, args in {
            "cuMemGetAllocationGranularity": [C.POINTER(SZ), C.POINTER(AllocProp), I],
            "cuMemAddressReserve": [C.POINTER(U64), SZ, SZ, U64, U64],
            "cuMemCreate": [C.POINTER(U64), SZ, C.POINTER(AllocProp), U64],
            "cuMemMap": [U64, SZ, SZ, U64, U64],
            "cuMemSetAccess": [U64, SZ, C.POINTER(Access), SZ],
            "cuMemUnmap": [U64, SZ], "cuMemRelease": [U64], "cuMemAddressFree": [U64, SZ],
        }.items(): setattr(self, name, bind(self.lib, name, args))
        assert C.sizeof(AllocProp) == 32 and C.sizeof(Access) == 12
        prop = AllocProp(type=1, location=Location(1, 0))
        gran = SZ()
        self.ok(self.cuMemGetAllocationGranularity(C.byref(gran), C.byref(prop), 0))
        g = gran.value
        self.size = (slots * BLOB + g - 1) // g * g
        self.base, self.maps = U64(), []
        self.ok(self.cuMemAddressReserve(C.byref(self.base), self.size, g, 0, 0))
        ranges = []
        for slot in sorted(blobs):
            start, end = slot * BLOB // g * g, ((slot + 1) * BLOB + g - 1) // g * g
            if ranges and start <= ranges[-1][1]: ranges[-1][1] = max(end, ranges[-1][1])
            else: ranges.append([start, end])
        try:
            for start, end in ranges:
                if cuda.device_bytes + cuda.vmm_bytes + end - start > 12 * 2**30:
                    raise MemoryError("Explicit CUDA physical-allocation budget is 12GiB")
                h = U64()
                self.ok(self.cuMemCreate(C.byref(h), end - start, C.byref(prop), 0))
                at = self.base.value + start
                try:
                    self.ok(self.cuMemMap(at, end - start, 0, h, 0))
                except BaseException:
                    self.ok(self.cuMemRelease(h))
                    raise
                self.maps.append((at, end - start, h))
                cuda.vmm_bytes += end - start
                self.ok(self.cuMemSetAccess(at, end - start, C.byref(Access(Location(1, 0), 3)), 1))
            # One reusable pinned staging blob; no CUDA copy from pageable fixture bytes.
            stage = cuda.pinned((BLOB,), np.uint8)
            for slot, blob in blobs.items():
                stage[:] = np.frombuffer(blob, np.uint8)
                cuda.copy(P(self.base.value + slot * BLOB), P(stage.ctypes.data), BLOB, 1)
                cuda.sync()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def ok(status):
        if status: raise RuntimeError(f"CUDA Driver VMM status {status}; high-slot proof BLOCKED, no dense fallback")

    def close(self):
        for at, size, handle in reversed(self.maps):
            self.ok(self.cuMemUnmap(at, size))
            self.ok(self.cuMemRelease(handle))
            self.c.vmm_bytes -= size
        self.maps.clear()
        if self.base.value:
            self.ok(self.cuMemAddressFree(self.base, self.size))
            self.base.value = 0


def gpu_checks(lease):
    marker = Path(lease)
    if not marker.is_file() or not marker.read_text().startswith("ACTIVE QFNBASALT"):
        raise RuntimeError("Exclusive coordinator lease must begin ACTIVE QFNBASALT")
    c = CUDA()
    dll_path = HERE.parent.parent / "bin/qfn-basalt-nvfp4.dll"
    dll_before = hashlib.sha256(dll_path.read_bytes()).hexdigest()
    lib = C.CDLL(str(dll_path))
    assert hashlib.sha256(dll_path.read_bytes()).hexdigest() == dll_before, "DLL changed between hash and load"
    abi = bind(lib, "qfn_basalt_nvfp4_abi", [])
    prep = bind(lib, "qfn_basalt_nvfp4_prepare", [P])
    query = bind(lib, "qfn_basalt_nvfp4_workspace_bytes", [I, I, C.c_uint, I], U64)
    run = bind(lib, "qfn_basalt_nvfp4_run", [P, P, P, U64, C.c_uint, I, I, P, P, U64, P, I])
    assert abi() == 1
    c.ok(prep(c.stream))
    assert query(0, K, 512, 0) == query(1, K - 1, 512, 0) == query(1024, K, 20002, 0) == 0
    results = {"abi": abi(), "dll_sha256_before": dll_before,
               "workspace_bytes": {str(n): query(n, K, 512, 0) for n in (1, 1023, 2048, 16384)}}
    banks = []
    try:
        for slots, selected, cases in (
            (20002, [0, 1, 511, 512, 1023, 1024, 2047, 8191, 20000, 20001], [(1, 2), (4, 2), (1023, 0)]),
            (512, [0, 1, 15, 16, 31, 32, 63, 64, 510, 511], [(1024, 0), (16385, 0)]),
        ):
            blobs = {slot: fixture(slot) for slot in selected}
            bank = SparseBank(c, slots, blobs)
            banks.append(bank)
            results[f"bank_{slots}"] = {"virtual_bytes": bank.size, "physical_bytes": sum(m[1] for m in bank.maps), "maximum_tested_slot": max(selected)}
            for nt, mode in cases:
                x = c.pinned((nt, E), np.float32)
                ids = c.pinned((nt, K), np.int32)
                out = c.pinned((nt, K, E), np.float32)
                x[:] = np.random.default_rng(71).normal(0, .1, x.shape).astype(np.float32)
                if nt > 1: x[1] = 0
                for t in range(nt): ids[t] = np.roll(selected, t % K)
                # Repeated physical placeholders are valid on native decode, including capture.
                if slots > 512 and nt > 1: ids[nt - 1, 3:6] = selected[0]
                dx, di, dy = c.malloc(x.nbytes), c.malloc(ids.nbytes), c.malloc(out.nbytes)
                nw = query(nt, K, slots, mode)
                work = c.malloc(nw)
                def invoke(n=nt, m=mode, xx=dx, ii=di, yy=dy):
                    return run(xx, ii, P(bank.base.value), BLOB, slots, n, K, yy, work, nw, c.stream, m)
                def upload():
                    c.copy(dx, P(x.ctypes.data), x.nbytes, 1)
                    c.copy(di, P(ids.ctypes.data), ids.nbytes, 1)
                def download():
                    c.copy(P(out.ctypes.data), dy, out.nbytes, 2)
                    c.sync()
                upload()
                assert run(dx, di, P(bank.base.value), BLOB - 1, slots, nt, K, dy, work, nw, c.stream, mode) != 0
                assert run(dx, di, P(bank.base.value), BLOB, slots, nt, K, dy, work, nw - 1, c.stream, mode) != 0
                c.ok(invoke())
                download()
                assert np.isfinite(out).all()
                if nt > 1: assert np.array_equal(out[1], np.zeros((K,E), np.float32)), "Zero input must give zero output"
                # CPU double products with explicit source-compatible activation quantization.
                errors = []
                for k in (0, K - 1):
                    ref = reference(x[0], blobs[int(ids[0, k])], 1 if slots == 512 else 2)
                    err = float(np.linalg.norm(out[0, k] - ref) / max(np.linalg.norm(ref), 1e-20))
                    errors.append(err)
                    if err > .02: raise AssertionError(f"CPU quantized reference RMS {err} > .02")
                # Sampled token independence/scatter reference uses the very same source kernels,
                # but a separate one-token invocation; this is not an independent math proof.
                sampled = out[[0, nt - 1]].copy()
                for j, t in enumerate((0, nt - 1)):
                    c.ok(invoke(1, 1 if slots == 512 else 2, P(dx.value + t * E * 4), P(di.value + t * K * 4), dy))
                    one = c.pinned((K, E), np.float32)
                    c.copy(P(one.ctypes.data), dy, one.nbytes, 2)
                    c.sync()
                    if not np.array_equal(one, sampled[j]):
                        raise AssertionError(f"Token independence/scatter mismatch t={t} n={nt}")
                graph, executable = P(), P()
                c.ok(c.cudaStreamBeginCapture(c.stream, 0))
                c.ok(invoke())
                c.ok(c.cudaStreamEndCapture(c.stream, C.byref(graph)))
                try:
                    c.ok(c.cudaGraphInstantiateWithFlags(C.byref(executable), graph, 0))
                    # Pinned inputs change between launches; persistent device/workspace addresses do not.
                    x[0] *= np.float32(.75)
                    upload()
                    c.ok(c.cudaGraphLaunch(executable, c.stream))
                    download()
                    captured = out.copy()
                    c.ok(invoke())
                    download()
                    assert np.array_equal(out, captured), "Captured replay differs from ordinary same-input run"
                finally:
                    if executable.value: c.ok(c.cudaGraphExecDestroy(executable))
                    c.ok(c.cudaGraphDestroy(graph))
                results[f"n{nt}_mode{mode}_slots{slots}"] = {"cpu_relative_rms": errors, "token_independence": "bitwise PASS", "capture_pinned_replay": "bitwise PASS", "workspace_bytes": nw}
                progress(f"measurement=NVFP4 bridge nt={nt} mode={mode} slots={slots} RMS={errors}; next=remaining boundary case")
            bank.close()
            banks.pop()
    finally:
        for bank in reversed(banks): bank.close()
        c.close()
    results["dll_sha256_after"] = hashlib.sha256(dll_path.read_bytes()).hexdigest()
    assert results["dll_sha256_after"] == dll_before, "DLL changed during GPU tests"
    return results


def main():
    ap = argparse.ArgumentParser(__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--cpu-only", action="store_true")
    group.add_argument("--abi-only", action="store_true")
    group.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--lease", default=str(ROOT / "runtime/nvfp4-parity.lease"))
    ap.add_argument("--receipt", default=str(ROOT / "nvfp4-parity.json"))
    args = ap.parse_args()
    receipt = Path(args.receipt).resolve()
    if not receipt.is_relative_to(ROOT): raise ValueError("Receipt must remain under task root: " + str(ROOT))
    report = {"cpu": cpu_checks(), "gpu": "NOT RUN"}
    if args.abi_only: report["abi"] = abi_checks()
    if args.allow_gpu: report["gpu"] = gpu_checks(args.lease)
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    progress("measurement=bridge parity " + ("GPU PASS" if args.allow_gpu else "CPU fixture PASS; GPU NOT RUN") + "; next=coordinator review")


if __name__ == "__main__": main()
