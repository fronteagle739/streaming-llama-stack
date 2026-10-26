# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
from dataclasses import dataclass
import subprocess
import sys
from uuid import UUID

class CudaDeviceIdentityError(RuntimeError):
    pass

def _uuid(value) -> str:
    text = str(value or '').strip()
    if text.lower().startswith('gpu-'):
        text = text[4:]
    try:
        return 'GPU-' + str(UUID(text))
    except (ValueError, AttributeError) as exc:
        raise CudaDeviceIdentityError('CUDA device UUID is missing or invalid') from exc

@dataclass(frozen=True)
class CudaDeviceIdentity:
    selected: str
    physical_index: str
    uuid: str
    logical_index: int

    @property
    def device(self) -> str:
        return f'cuda:{self.logical_index}'

def resolve_cuda_identity(selected, physical_devices: list[dict], torch_devices: list[dict]) -> CudaDeviceIdentity:
    selection = str(selected).strip()
    target_uuid = None if selection.isdigit() else _uuid(selection)
    physical = []
    indices = set()
    uuids = set()
    for row in physical_devices:
        index = str(row.get('index', '')).strip()
        identity = _uuid(row.get('uuid'))
        if not index.isdigit() or index in indices or identity in uuids:
            raise CudaDeviceIdentityError('Physical CUDA inventory has an invalid or duplicate identity')
        indices.add(index)
        uuids.add(identity)
        if target_uuid is None and index == selection or identity == target_uuid:
            physical.append((index, identity))
    if len(physical) != 1:
        raise CudaDeviceIdentityError(f'Configured physical GPU {selection!r} is not uniquely identified')
    physical_index, target_uuid = physical[0]
    logical = []
    ordinals = set()
    uuids = set()
    for row in torch_devices:
        ordinal = row.get('index')
        identity = _uuid(row.get('uuid'))
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0 or (ordinal in ordinals) or (identity in uuids):
            raise CudaDeviceIdentityError('Visible Torch inventory has an invalid or duplicate identity')
        ordinals.add(ordinal)
        uuids.add(identity)
        if identity == target_uuid:
            logical.append(ordinal)
    if len(logical) != 1:
        raise CudaDeviceIdentityError(f'Configured GPU {target_uuid} is not uniquely visible to Torch')
    return CudaDeviceIdentity(selection, physical_index, target_uuid, logical[0])

def nvidia_device_inventory() -> list[dict]:
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError) as exc:
        raise CudaDeviceIdentityError('Cannot verify configured physical GPU identity with nvidia-smi') from exc
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) != 2:
            raise CudaDeviceIdentityError('Invalid nvidia-smi GPU identity inventory')
        rows.append({'index': fields[0], 'uuid': fields[1]})
    return rows

def configured_torch_device(selected, torch_module) -> str:
    visible = [{'index': index, 'uuid': getattr(torch_module.cuda.get_device_properties(index), 'uuid', None)} for index in range(torch_module.cuda.device_count())]
    identity = resolve_cuda_identity(selected, nvidia_device_inventory(), visible)
    print(f'[cuda-device] selected={identity.selected} physical_index={identity.physical_index} uuid={identity.uuid} logical_device={identity.device}', file=sys.stderr, flush=True)
    return identity.device

def dedicated_free_bytes(gpu_uuid: str) -> int:
    identity = _uuid(gpu_uuid)
    try:
        result = subprocess.run(['nvidia-smi', '--id=' + identity, '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=5, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        fields = result.stdout.strip().split(',')
        if len(fields) != 2 or _uuid(fields[0].strip()) != identity:
            raise ValueError('Dedicated memory reading belongs to another GPU')
        free = int(fields[1].strip()) * 1024 ** 2
        if free <= 0:
            raise ValueError('No dedicated headroom')
        return free
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise CudaDeviceIdentityError('Cannot verify dedicated GPU headroom') from exc
