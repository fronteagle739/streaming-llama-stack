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


def envelope(entries, end):
    data = bytearray(HEADER_BYTES)
    HEADER.pack_into(data, 0, MAGIC, VERSION, len(entries), end, bytes(40))
    for index, (name, offset, size, digest) in enumerate(entries):
        ENTRY.pack_into(data, 64 + ENTRY.size * index, name.encode('utf8'), offset,
                        size, digest, bytes(16))
    return bytes(data)


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

    def _validate_name(self, name):
        raw = name.encode('utf8')
        if not raw or b'\0' in raw or len(raw) > 63:
            raise ValueError('Section name must contain 1..63 UTF8 bytes without NUL')
        if any(row[0] == name for row in self.entries):
            raise ValueError('Duplicate section: ' + name)
        if len(self.entries) >= MAX_SECTIONS:
            raise ValueError('Too many sections')

    def _start(self, name):
        self._validate_name(name)
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
        self.f.write(envelope(self.entries, end))
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()

    def readback(self, name, offset=0, size=None):
        self.f.flush()
        row = next(row for row in self.entries if row[0] == name)
        size = row[2] - offset if size is None else size
        if offset < 0 or size < 0 or offset + size > row[2]:
            raise ValueError('Readback range outside section: ' + name)
        return b''.join(file_chunks(self.path, row[1] + offset, size))

    def abort(self):
        self.f.close()


def file_identity(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class VerificationWriter(ModelWriter):
    """Consume real regenerated bytes, then hash their exact virtual envelope.

    No payload file is created. Readback for derived sections is admitted only
    after both regenerated and reference section digests have been checked.
    """
    def __init__(self, reference, canonical_metadata):
        self.identity = file_identity(reference)
        self.reference = ModelFile(reference)
        self._unchanged()
        self.canonical_metadata = canonical_metadata
        self.entries, self.position = [], HEADER_BYTES
        self.aborted = False
        self.metadata = None
        self.readback_verified = set()
        self.report = None
        if self.reference.names()[-1] != 'meta':
            raise ValueError('Raw-import verification requires the final meta section')

    def _unchanged(self):
        if getattr(self, 'aborted', False):
            raise ValueError('Verification was aborted')
        if file_identity(self.reference.path) != self.identity:
            raise ValueError('Reference identity changed during verification')

    def add_chunks(self, name, chunks):
        self._unchanged()
        self._validate_name(name)
        names = self.reference.names()
        if len(self.entries) >= len(names) or names[len(self.entries)] != name:
            raise ValueError('Regenerated section order differs: ' + name)
        offset, digest, count = align(self.position), hashlib.sha256(), 0
        metadata = bytearray() if name == 'meta' else None
        for chunk in chunks:
            digest.update(chunk)
            count += len(chunk)
            if metadata is not None:
                if count > 32 << 20:
                    raise ValueError('Regenerated metadata exceeds bounded verification limit')
                metadata.extend(chunk)
        row = (name, offset, count, digest.digest())
        if name != 'meta' and row[1:] != self.reference.sections[name]:
            raise ValueError('Regenerated section offset/size/SHA256 differs: ' + name)
        if metadata is not None:
            if offset != self.reference.sections[name][0]:
                raise ValueError('Regenerated metadata offset differs')
            if not self.reference.verify('meta'):
                raise ValueError('Reference metadata SHA256 differs')
            expected = self.canonical_metadata(json.loads(self.reference.read('meta')))
            actual = json.loads(metadata)
            serialized = (json.dumps(expected, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()
            if serialized != metadata:
                raise ValueError('Regenerated metadata differs from canonical reference: ' +
                                 ', '.join(metadata_differences(expected, actual)))
            self.metadata = bytes(metadata)
        self.entries.append(row)
        self.position = offset + count
        return count

    def readback(self, name, offset=0, size=None):
        self._unchanged()
        if name not in {row[0] for row in self.entries}:
            raise ValueError('Readback before regeneration: ' + name)
        if name not in self.readback_verified:
            if not self.reference.verify(name):
                raise ValueError('Reference readback section SHA256 differs: ' + name)
            self._unchanged()
            self.readback_verified.add(name)
        start, length, _ = self.reference.sections[name]
        size = length - offset if size is None else size
        if offset < 0 or size < 0 or offset + size > length:
            raise ValueError('Readback range outside section: ' + name)
        return b''.join(file_chunks(self.reference.path, start + offset, size))

    def close(self):
        self._unchanged()
        if [row[0] for row in self.entries] != self.reference.names() or self.metadata is None:
            raise ValueError('Not every reference section was regenerated')
        projected = hashlib.sha256(envelope(self.entries, self.position))
        reference_hash = hashlib.sha256()
        with open(self.reference.path, 'rb') as stream:
            reference_header = stream.read(HEADER_BYTES)
            reference_rows = [(name, *row) for name, row in self.reference.sections.items()]
            if reference_header != envelope(reference_rows, self.identity[2]):
                raise ValueError('Reference header changed or is not canonical')
            reference_hash.update(reference_header)
            position = HEADER_BYTES
            for name, offset, size, expected in self.entries:
                gap = offset - position
                while gap:
                    data = stream.read(min(gap, COPY_CHUNK))
                    if not data or any(data):
                        raise ValueError('Reference alignment gap is not zero-filled')
                    reference_hash.update(data)
                    projected.update(data)
                    gap -= len(data)
                _, old_size, old_sha = self.reference.sections[name]
                actual = hashlib.sha256()
                left = old_size
                while left:
                    data = stream.read(min(left, COPY_CHUNK))
                    if not data:
                        raise ValueError('Short reference payload: ' + name)
                    reference_hash.update(data)
                    actual.update(data)
                    if name != 'meta':
                        projected.update(data)
                    left -= len(data)
                if actual.digest() != old_sha or (name != 'meta' and actual.digest() != expected):
                    raise ValueError('Reference/regenerated section SHA256 differs: ' + name)
                if name == 'meta':
                    projected.update(self.metadata)
                position = offset + old_size
            if stream.read(1):
                raise ValueError('Unexpected reference tail')
        self._unchanged()
        self.report = dict(mode='complete-raw-import-verification', format='Streaming Llama Pack',
            version=VERSION, bytes=self.position,
            sha256=projected.hexdigest(), reference_bytes=self.identity[2],
            reference_sha256=reference_hash.hexdigest(),
            reference_identity=list(self.identity),
            every_nonmetadata_section_regenerated_and_verified=True,
            metadata_canonicalized_without_dropped_fields=True,
            sections={name: dict(offset=offset, bytes=size, sha256=sha.hex())
                      for name, offset, size, sha in self.entries})
        return self.report

    def abort(self):
        self.aborted = True


def metadata_differences(expected, actual, prefix=''):
    """Bounded field names for diagnosis, never dump model metadata values."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in dict.fromkeys([*expected, *actual]):
            path = prefix + '.' + str(key) if prefix else str(key)
            if key not in expected or key not in actual:
                result.append(path)
            elif expected[key] != actual[key]:
                result.extend(metadata_differences(expected[key], actual[key], path))
            if len(result) >= 16:
                break
        return result[:16]
    return [prefix]


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
