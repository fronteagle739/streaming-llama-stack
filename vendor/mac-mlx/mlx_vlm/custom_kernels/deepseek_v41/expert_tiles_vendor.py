"""Read pinned upstream objects without invoking Git; reproduce the header bundle.

Development utility only. Never imported by the production kernel wrapper.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import struct
import zlib

COMMIT = "1f8e74e3f12f31365464a6867c6579f0e9b29d85"


class TagSource:
    def __init__(self, root):
        self.objects = Path(root) / ".git" / "objects"
        self.index = {}
        for path in (self.objects / "pack").glob("*.idx"):
            raw = path.read_bytes()
            assert raw[:8] == b"\xfftOc\x00\x00\x00\x02"
            count = struct.unpack_from(">I", raw, 8 + 255 * 4)[0]
            base = 8 + 256 * 4
            for i in range(count):
                oid = raw[base + i * 20:base + (i + 1) * 20].hex()
                offset = struct.unpack_from(">I", raw, base + count * 24 + i * 4)[0]
                if offset & 0x80000000:
                    offset = struct.unpack_from(">Q", raw, base + count * 28 + (offset & 0x7fffffff) * 8)[0]
                self.index[oid] = (path.with_suffix(".pack"), offset)

    @lru_cache(None)
    def object(self, oid):
        loose = self.objects / oid[:2] / oid[2:]
        if loose.exists():
            raw = zlib.decompress(loose.read_bytes())
            header, data = raw.split(b"\0", 1)
            kind = header.split()[0].decode()
        else:
            kind, data = self.packed(*self.index[oid])
        header = f"{kind} {len(data)}".encode() + b"\0"
        assert hashlib.sha1(header + data).hexdigest() == oid
        return kind, data

    @lru_cache(None)
    def packed(self, path, offset):
        with path.open("rb") as f:
            f.seek(offset)
            c = f.read(1)[0]
            kind = (c >> 4) & 7
            while c & 128:
                c = f.read(1)[0]
            if kind == 6:
                c = f.read(1)[0]
                distance = c & 127
                while c & 128:
                    c = f.read(1)[0]
                    distance = ((distance + 1) << 7) | (c & 127)
                base_kind, base = self.packed(path, offset - distance)
            elif kind == 7:
                base_kind, base = self.object(f.read(20).hex())
            decoder = zlib.decompressobj()
            chunks = []
            while not decoder.eof:
                chunk = f.read(65536)
                assert chunk
                chunks.append(decoder.decompress(chunk))
            data = b"".join(chunks)
        if kind not in (6, 7):
            return {1: "commit", 2: "tree", 3: "blob", 4: "tag"}[kind], data
        pos = 0

        def varint():
            nonlocal pos
            value = shift = 0
            while True:
                byte = data[pos]
                pos += 1
                value |= (byte & 127) << shift
                if not byte & 128:
                    return value
                shift += 7

        assert varint() == len(base)
        size = varint()
        result = bytearray()
        while pos < len(data):
            op = data[pos]
            pos += 1
            if op & 128:
                start = length = 0
                for bit in range(7):
                    if op & (1 << bit):
                        value = data[pos]
                        pos += 1
                        if bit < 4:
                            start |= value << (bit * 8)
                        else:
                            length |= value << ((bit - 4) * 8)
                result.extend(base[start:start + (length or 65536)])
            else:
                assert op
                result.extend(data[pos:pos + op])
                pos += op
        assert len(result) == size
        return base_kind, bytes(result)

    def read(self, name):
        kind, commit = self.object(COMMIT)
        assert kind == "commit"
        oid = commit.splitlines()[0].split()[1].decode()
        for part in name.split("/"):
            kind, tree = self.object(oid)
            assert kind == "tree"
            entries = {}
            pos = 0
            while pos < len(tree):
                end = tree.index(b"\0", pos)
                filename = tree[pos:end].split(b" ", 1)[1].decode()
                entries[filename] = tree[end + 1:end + 21].hex()
                pos = end + 21
            oid = entries[part]
        kind, result = self.object(oid)
        assert kind == "blob"
        return result


def bundle(source):
    """Inline exact headers, removing only local includes and pragma once.

    utils() is prepended by MLX itself. Hash it and its dependencies for the
    provenance record, but never redeclare those types in our header.
    """
    import re
    root = "mlx/backend/metal/kernels/"
    hashes, seen, parts = {}, set(), []

    def record(path):
        raw = source.read(path)
        hashes[path] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    def expand(path, implicit=False):
        if path in seen:
            return
        seen.add(path)
        raw = record(path)
        for dep in re.findall(r'^#include "([^"]+)"', raw, re.M):
            expand(dep, implicit)
        if implicit:
            return
        if path == root + "fp_quantized.h":
            # Exact helper prefix, ending immediately after QuantizedBlockLoader.
            raw = raw.split("template <typename T, int group_size, int bits, int D>", 1)[0]
            raw = re.sub(r'^constant bool align_[MNK].*\n', '', raw, flags=re.M)
        raw = re.sub(r'^#include "[^\n]+\n|^#pragma once\n', '', raw, flags=re.M)
        parts.append(f'// BEGIN UPSTREAM {path}\n#line 1 "{path}"\n' + raw
                     + f'\n// END UPSTREAM {path}\n')

    license_text = record("LICENSE")
    expand(root + "utils.h", implicit=True)
    expand(root + "quantized_utils.h")
    expand(root + "steel/gemm/gemm.h")
    expand(root + "fp_quantized.h")
    header = (f"// Vendored from MLX v0.32.2 ({COMMIT}).\n"
              "// Regenerate: python expert_tiles_vendor.py ./GitHub/mlx --bundle\n"
              "// utils()/BF16 definitions are provided by MLX 0.32.2 itself.\n/*\n"
              + license_text + "*/\n" + "\n".join(parts))
    return header, hashes


if __name__ == "__main__":
    import sys
    source = TagSource(sys.argv[1])
    if sys.argv[2:] == ["--bundle"]:
        import pprint
        folder = Path(__file__).parent
        header, hashes = bundle(source)
        metal = folder / "expert_tiles_qmm.metal"
        marker = "// --- metadata body ---"
        bodies = marker + metal.read_text(encoding="utf-8").split(marker, 1)[1]
        metal.write_text(header + bodies, encoding="utf-8", newline="\n")
        wrapper = folder / "expert_tiles_qmm.py"
        text = wrapper.read_text(encoding="utf-8")
        start, rest = text.split("# BEGIN SOURCE HASHES\n", 1)
        _, end = rest.split("# END SOURCE HASHES", 1)
        manifest = "SOURCE_HASHES = " + pprint.pformat(hashes, sort_dicts=True)
        wrapper.write_text(start + "# BEGIN SOURCE HASHES\n" + manifest
                           + "\n# END SOURCE HASHES" + end, encoding="utf-8", newline="\n")
        print(f"Bundled {len(hashes)} pinned source hashes, {len(header)} header characters")
    else:
        for name in sys.argv[2:]:
            print(f"// {name} @ {COMMIT}")
            print(source.read(name).decode())
