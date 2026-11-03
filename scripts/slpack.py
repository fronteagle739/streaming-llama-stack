"""Streaming Llama Pack v1: deterministic, bounded section container.

The envelope preserves every payload byte. Tensor encodings are architecture-specific.
Legacy containers are accepted only when the caller explicitly permits migration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

MAGIC = b'SLPACK\0\0'
VERSION = 1
HEADER_BYTES = 64 << 10
SECTION_ALIGN = 2 << 20
TENSOR_ALIGN = 4096
HEADER = struct.Struct('<8sIIQ40s')
ENTRY = struct.Struct('<64sQQ32s16s')
MAX_SECTIONS = (HEADER_BYTES - HEADER.size) // ENTRY.size
COPY_CHUNK = 4 << 20
# Historical format identity; used only by the explicit compatibility reader.
LEGACY_MAGIC = bytes.fromhex('424153414c544d46')


def align(n):
    return (n + SECTION_ALIGN - 1) // SECTION_ALIGN * SECTION_ALIGN


def file_chunks(path, offset=0, size=None):
    with open(path, 'rb') as stream:
        stream.seek(offset)
        left = Path(path).stat().st_size - offset if size is None else size
        if left < 0:
            raise ValueError('Negative file range')
        while left:
            chunk = stream.read(min(left, COPY_CHUNK))
            if not chunk:
                raise ValueError('Short source file range')
            yield chunk
            left -= len(chunk)


class Stream:
    def __init__(self, size, chunks):
        self.size, self.chunks = size, chunks


class ModelWriter:
    """Exclusive creation; failed partials remain available for inspection."""
    def __init__(self, path):
        self.path = str(path)
        self.f = open(path, 'xb')
        self.f.write(bytes(HEADER_BYTES))
        self.entries = []

    def _start(self, name):
        raw = name.encode('utf8')
        if not raw or b'\0' in raw or len(raw) > 63:
            raise ValueError('Section name must contain 1..63 UTF8 bytes without NUL')
        if any(row[0] == name for row in self.entries):
            raise ValueError('Duplicate section: ' + name)
        if len(self.entries) >= MAX_SECTIONS:
            raise ValueError('Too many sections')
        offset = align(self.f.tell())
        self.f.write(bytes(offset - self.f.tell()))
        return offset

    def add_chunks(self, name, chunks):
        offset = self._start(name)
        digest, count = hashlib.sha256(), 0
        for chunk in chunks:
            self.f.write(chunk)
            digest.update(chunk)
            count += len(chunk)
        self.entries.append((name, offset, count, digest.digest()))
        return count

    def add_bytes(self, name, data):
        return self.add_chunks(name, (data,))

    def add_file(self, name, source, offset=0, size=None):
        return self.add_chunks(name, file_chunks(source, offset, size))

    def add_parts(self, name, parts, align=TENSOR_ALIGN):
        offsets, position = [], 0
        def chunks():
            nonlocal position
            for part in parts:
                padding = (-position) % align
                yield bytes(padding)
                position += padding
                offsets.append(position)
                if isinstance(part, (bytes, bytearray)):
                    source, size = (bytes(part),), len(part)
                elif isinstance(part, Stream):
                    source, size = part.chunks, part.size
                else:
                    path, start, size = part
                    source = file_chunks(path, start, size)
                before = position
                for chunk in source:
                    position += len(chunk)
                    yield chunk
                if position - before != size:
                    raise ValueError('Stream part length differs from declared size')
        self.add_chunks(name, chunks())
        return offsets

    def close(self):
        if not self.entries:
            raise ValueError('A complete pack requires at least one section')
        end = self.f.tell()
        self.f.seek(0)
        self.f.write(HEADER.pack(MAGIC, VERSION, len(self.entries), end, bytes(40)))
        for name, offset, size, digest in self.entries:
            self.f.write(ENTRY.pack(name.encode('utf8'), offset, size, digest, bytes(16)))
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()


class ModelFile:
    def __init__(self, path, *, allow_legacy=False, versions=None):
        self.path = str(path)
        with open(path, 'rb') as stream:
            header = stream.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES:
            raise ValueError('Short section header')
        magic, version, count, size, reserved = HEADER.unpack_from(header)
        self.legacy = magic == LEGACY_MAGIC
        if magic != MAGIC and not (allow_legacy and self.legacy):
            raise ValueError('Not a Streaming Llama Pack; legacy migration requires explicit permission')
        if (self.legacy and version not in (1, 2)) or (not self.legacy and version != VERSION):
            raise ValueError('Unsupported container version')
        if versions is not None and version not in versions:
            raise ValueError('Container version not admitted by this caller')
        if size != Path(path).stat().st_size or not 0 < count <= MAX_SECTIONS:
            raise ValueError('Declared file size or section count differs')
        if not self.legacy and reserved != bytes(40):
            raise ValueError('Unknown active header fields')
        self.version, self.sections = version, {}
        end = HEADER_BYTES
        for index in range(count):
            raw, offset, length, sha, tail = ENTRY.unpack_from(header, 64 + 128 * index)
            name_bytes, separator, padding = raw.partition(b'\0')
            if not separator or not name_bytes or padding.strip(b'\0'):
                raise ValueError('Invalid section name')
            name = name_bytes.decode('utf8')
            if name in self.sections or offset % SECTION_ALIGN or offset < align(end) or offset + length > size:
                raise ValueError('Duplicate, overlapping or out-of-bounds section: ' + name)
            if not self.legacy and tail != bytes(16):
                raise ValueError('Unknown active section fields')
            self.sections[name] = (offset, length, sha)
            end = offset + length
        if end != size:
            raise ValueError('Unexpected file tail')
        unused = header[64 + 128 * count:]
        if not self.legacy and unused.strip(b'\0'):
            raise ValueError('Unknown active header/table bytes')

    def names(self):
        return list(self.sections)

    def section(self, name):
        return self.sections[name][:2]

    def read(self, name, max_bytes=32 << 20):
        offset, size = self.section(name)
        if size > max_bytes:
            raise ValueError('Use bounded file_chunks for large sections')
        return b''.join(file_chunks(self.path, offset, size))

    def verify(self, name):
        offset, size, expected = self.sections[name]
        digest = hashlib.sha256()
        for chunk in file_chunks(self.path, offset, size):
            digest.update(chunk)
        return digest.digest() == expected

    def verify_all(self):
        for name in self.names():
            if not self.verify(name):
                raise ValueError('Section SHA256 differs: ' + name)
        digest = hashlib.sha256()
        for chunk in file_chunks(self.path):
            digest.update(chunk)
        return dict(format='Streaming Llama Pack', version=self.version,
                    legacy=self.legacy, bytes=Path(self.path).stat().st_size,
                    sha256=digest.hexdigest(), sections={name: dict(offset=row[0], bytes=row[1], sha256=row[2].hex())
                                                         for name, row in self.sections.items()})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--allow-legacy', action='store_true')
    args = parser.parse_args()
    print(json.dumps(ModelFile(args.model, allow_legacy=args.allow_legacy).verify_all(), indent=2))


if __name__ == '__main__':
    main()
