from __future__ import annotations
import struct
from pathlib import Path
from typing import Optional
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16, _T_UINT32, _T_INT32 = (0, 1, 2, 3, 4, 5)
_T_FLOAT32, _T_BOOL, _T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = (6, 7, 8, 9, 10, 11, 12)

def _read_kv_header(path: Path) -> Optional[dict]:
    with path.open('rb') as f:
        if f.read(4) != b'GGUF':
            return None
        struct.unpack('<I', f.read(4))[0]
        struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]

        def rstr() -> str:
            n = struct.unpack('<Q', f.read(8))[0]
            return f.read(n).decode('utf-8', 'replace')

        def rval(t: int):
            if t == _T_STRING:
                return rstr()
            if t == _T_UINT32:
                return struct.unpack('<I', f.read(4))[0]
            if t == _T_INT32:
                return struct.unpack('<i', f.read(4))[0]
            if t == _T_UINT64:
                return struct.unpack('<Q', f.read(8))[0]
            if t == _T_INT64:
                return struct.unpack('<q', f.read(8))[0]
            if t == _T_UINT8:
                return struct.unpack('<B', f.read(1))[0]
            if t == _T_INT8:
                return struct.unpack('<b', f.read(1))[0]
            if t == _T_UINT16:
                return struct.unpack('<H', f.read(2))[0]
            if t == _T_INT16:
                return struct.unpack('<h', f.read(2))[0]
            if t == _T_FLOAT32:
                return struct.unpack('<f', f.read(4))[0]
            if t == _T_FLOAT64:
                return struct.unpack('<d', f.read(8))[0]
            if t == _T_BOOL:
                return struct.unpack('<?', f.read(1))[0]
            if t == _T_ARRAY:
                et = struct.unpack('<I', f.read(4))[0]
                n = struct.unpack('<Q', f.read(8))[0]
                return [rval(et) for _ in range(n)]
            return None
        meta: dict = {}
        for _ in range(kv_count):
            key = rstr()
            vtype = struct.unpack('<I', f.read(4))[0]
            meta[key] = rval(vtype)
        return meta
