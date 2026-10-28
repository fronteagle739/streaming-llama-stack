#!/usr/bin/env python3
"""Audit the pinned checkpoint using bounded HTTP ranges; never fetch weights.

Full shard SHA-256 values in the manifest are expected LFS identities, not a
claim that a header-only audit verified the shard payloads.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import http.client
import json
from pathlib import Path, PurePosixPath
import re
import struct
import time
import urllib.request

REPO = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'
BASE = f'https://huggingface.co/{REPO}/resolve/{REVISION}/'
MAX_HEADER = 8 << 20
CHUNK = 1 << 20


def request(url, start=None, end=None, total=None):
    headers = {}
    if start is not None:
        headers['Range'] = f'bytes={start}-{end}'
        url += ('&' if '?' in url else '?') + f'nativefp4_range={start}-{end}'
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                if start is not None:
                    expected = f'bytes {start}-{end}/{total}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('Server did not honor the exact bounded range')
                limit = end - start + 1 if start is not None else MAX_HEADER
                data = response.read(limit + 1)
                if len(data) > limit or (start is not None and len(data) != limit):
                    raise ValueError('Truncated or oversized HTTP response')
                return data
        except (OSError, ValueError, http.client.IncompleteRead) as error:
            last = error
            time.sleep(min(attempt + 1, 3))
    raise RuntimeError(f'HTTP metadata request failed: {last}')


def fetch_blob(row):
    name, size = row['rfilename'], row['size']
    if size > 16 << 20:
        raise ValueError('Metadata blob exceeds bound')
    data = b''.join(request(BASE + name, start, min(start + CHUNK, size) - 1, size)
                    for start in range(0, size, CHUNK))
    oid = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
    if oid != row['blobId']:
        raise ValueError('Metadata Git blob identity mismatch: ' + name)
    return data


def fetch_header(row):
    size, name = row['size'], row['rfilename']
    prefix = request(BASE + name, 0, 7, size)
    length, = struct.unpack('<Q', prefix)
    if not 2 <= length <= MAX_HEADER or length + 8 > size:
        raise ValueError('Invalid safetensors header length')
    raw = b''.join(request(BASE + name, start, min(start + CHUNK, length + 8) - 1, size)
                   for start in range(8, length + 8, CHUNK))
    header = json.loads(raw)
    extents = []
    widths = {'I8': 1, 'U8': 1, 'F8_E4M3': 1, 'F8_E8M0': 1, 'BF16': 2, 'F16': 2, 'F32': 4, 'I32': 4, 'I64': 8}
    for tensor, info in header.items():
        if tensor == '__metadata__':
            continue
        shape = info['shape']
        if any(not isinstance(dim, int) or dim < 0 for dim in shape):
            raise ValueError('Invalid tensor shape: ' + tensor)
        count = 1
        for dim in shape:
            count *= dim
        begin, end = info['data_offsets']
        if begin < 0 or end < begin or end + length + 8 > size or end - begin != count * widths[info['dtype']]:
            raise ValueError('Invalid tensor extent: ' + tensor)
        extents.append((begin, end))
    end = 0
    for begin, next_end in sorted(extents):
        if begin != end:
            raise ValueError('Overlapping or noncontiguous tensor payload')
        end = next_end
    if end + length + 8 != size:
        raise ValueError('Unclaimed shard bytes')
    return prefix + raw, header


def audit(destination):
    destination.mkdir(parents=True, exist_ok=True)
    api = json.loads(request(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'))
    if api['sha'] != REVISION:
        raise ValueError('Source revision mismatch')
    rows = {row['rfilename']: row for row in api['siblings']}
    metadata = {}
    for name in ('config.json', 'model.safetensors.index.json', 'inference/model.py', 'inference/kernel.py'):
        data = fetch_blob(rows[name])
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        metadata[name] = {'bytes': len(data), 'git_blob_sha1': rows[name]['blobId'], 'sha256': hashlib.sha256(data).hexdigest()}
    config = json.loads((destination / 'config.json').read_bytes())
    index = json.loads((destination / 'model.safetensors.index.json').read_bytes())
    tensor_map = index['weight_map']
    shards = sorted(set(tensor_map.values()))
    for name in shards:
        if PurePosixPath(name).name != name or not re.fullmatch(r'model-\d{5}-of-00048\.safetensors', name):
            raise ValueError('Unexpected shard path')
    tensors, manifest = {}, []
    for name in shards:
        row = rows[name]
        if not re.fullmatch('[0-9a-f]{64}', row['lfs']['sha256']) or row['size'] != row['lfs']['size']:
            raise ValueError('Missing LFS identity')
        raw, header = fetch_header(row)
        (destination / (name + '.header')).write_bytes(raw)
        for tensor, info in header.items():
            if tensor == '__metadata__':
                continue
            if tensor in tensors or tensor_map.get(tensor) != name:
                raise ValueError('Index/header inventory mismatch: ' + tensor)
            tensors[tensor] = info
        manifest.append({'path': name, 'bytes': row['size'], 'expected_sha256': row['lfs']['sha256'],
                         'header_bytes': len(raw), 'header_sha256': hashlib.sha256(raw).hexdigest(),
                         'payload_sha256_verified': False})
        print(f'{len(manifest)}/{len(shards)} {name}: header checked', flush=True)
    if set(tensors) != set(tensor_map):
        raise ValueError('Source tensor inventory differs from index')
    groups = collections.Counter()
    for name, info in tensors.items():
        if not name.endswith('.weight') or info['dtype'] not in ('I8', 'F8_E4M3'):
            continue
        scale = tensors.get(name.removesuffix('.weight') + '.scale')
        shape = info['shape']
        if info['dtype'] == 'I8':
            expected, group = [shape[0], shape[1] // 16], 'fp4_e2m1_packed_i8_e8m0_per_32'
        elif '.engram.embed.' in name:
            expected, group = [shape[0], shape[1] // 32], 'engram_fp8_e4m3_e8m0_per_32'
        else:
            expected, group = [(dim + 31) // 32 for dim in shape], 'dense_fp8_e4m3_e8m0_per_32x32'
        if not scale or scale['dtype'] != 'F8_E8M0' or scale['shape'] != expected:
            raise ValueError('Source scale format mismatch: ' + name)
        groups[group] += 1
    result = {'repo': REPO, 'revision': REVISION, 'mode': 'headers_only', 'metadata': metadata,
              'quantization_config': config['quantization_config'], 'tensor_count': len(tensors),
              'dtype_counts': dict(collections.Counter(info['dtype'] for info in tensors.values())),
              'quantized_weight_groups': dict(groups), 'shard_count': len(manifest),
              'shard_bytes': sum(row['bytes'] for row in manifest), 'shards': manifest,
              'full_download_complete': False, 'all_payload_hashes_verified': False}
    (destination / 'source-manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key not in ('shards', 'metadata')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out)
