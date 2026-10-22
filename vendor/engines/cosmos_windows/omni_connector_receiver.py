# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/omni_connector_receiver.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import inspect
import json
import socket
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
TBP_MAGIC = 1413632049
TBP_PROTO_MAJOR = 1
TBP_PROTO_MINOR = 0
TBP_HEADER_SIZE = 52
TBP_MSG_HELLO = 1
TBP_MSG_SESSION_START = 3
TBP_MSG_KV_SEG_BEGIN = 7
TBP_MSG_KV_CHUNK = 8
TBP_MSG_KV_SEG_END = 9
TBP_MSG_KV_ACK = 10
TBP_MSG_KV_DONE = 12
_TBP_HEADER = struct.Struct('<IHHHHIIQQQII')

class TBPError(RuntimeError):
    pass

@dataclass(frozen=True)
class TBPFrame:
    msg_type: int
    flags: int
    session_id: int
    stream_id: int
    seq_no: int
    payload: bytes
    proto_major: int = TBP_PROTO_MAJOR
    proto_minor: int = TBP_PROTO_MINOR
    payload_crc32: int = 0

@dataclass(frozen=True)
class OmniArtifact:
    from_stage: str
    to_stage: str
    put_key: str
    payload: bytes
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.payload)

@dataclass(frozen=True)
class CosmosGenerationRequest:
    request_id: str
    prompt: str
    negative_prompt: Optional[str] = None
    height: int = 256
    width: int = 256
    num_frames: int = 1
    num_inference_steps: int = 35
    guidance_scale: float = 6.0
    flow_shift: float = 10.0
    seed: Optional[int] = None
    ar_conditioning: bytes = b''
    metadata: Dict[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class CosmosGenerationResult:
    request: CosmosGenerationRequest
    output_dir: Path
    image_path: Path
    metadata_path: Path
    seconds: float

def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 4294967295

def _recv_exact(sock: socket.socket, size: int, timeout_s: float) -> bytes:
    sock.settimeout(timeout_s)
    buf = bytearray()
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise TBPError(f'socket closed while reading {size} bytes')
        buf.extend(chunk)
    return bytes(buf)

def read_tbp_frame(sock: socket.socket, *, timeout_s: float=30.0, validate_crc: bool=True) -> TBPFrame:
    header = _recv_exact(sock, TBP_HEADER_SIZE, timeout_s)
    magic, proto_major, proto_minor, msg_type, flags, header_size, payload_size, session_id, stream_id, seq_no, _header_crc32, payload_crc32 = _TBP_HEADER.unpack(header)
    if magic != TBP_MAGIC:
        raise TBPError(f'bad TBP1 magic 0x{magic:08x}')
    if header_size != TBP_HEADER_SIZE:
        raise TBPError(f'unsupported TBP1 header size {header_size}')
    if proto_major != TBP_PROTO_MAJOR:
        raise TBPError(f'unsupported TBP1 protocol {proto_major}.{proto_minor}')
    payload = b''
    if payload_size:
        payload = _recv_exact(sock, payload_size, timeout_s)
        if validate_crc and payload_crc32 and (_crc32(payload) != payload_crc32):
            raise TBPError(f'payload crc mismatch for seq={seq_no}: got 0x{_crc32(payload):08x}, expected 0x{payload_crc32:08x}')
    return TBPFrame(msg_type=msg_type, flags=flags, session_id=session_id, stream_id=stream_id, seq_no=seq_no, payload=payload, proto_major=proto_major, proto_minor=proto_minor, payload_crc32=payload_crc32)

def build_tbp_frame(msg_type: int, payload: bytes=b'', *, flags: int=0, session_id: int=0, stream_id: int=0, seq_no: int=0) -> bytes:
    payload_crc32 = _crc32(payload) if payload else 0
    header = bytearray(_TBP_HEADER.pack(TBP_MAGIC, TBP_PROTO_MAJOR, TBP_PROTO_MINOR, msg_type, flags, TBP_HEADER_SIZE, len(payload), session_id, stream_id, seq_no, 0, payload_crc32))
    header_crc32 = _crc32(bytes(header))
    struct.pack_into('<I', header, 44, header_crc32)
    return bytes(header) + payload

def parse_kv_pairs(payload: bytes) -> Dict[str, str]:
    text = payload.decode('utf-8', errors='replace').strip()
    if not text:
        return {}
    fields: Dict[str, str] = {}
    for token in text.replace(';', ' ').replace('\n', ' ').split():
        if '=' not in token:
            continue
        key, value = token.split('=', 1)
        if key:
            fields[key] = value
    return fields

def format_kv_pairs(values: Mapping[str, object]) -> bytes:
    return ';'.join((f'{key}={value}' for key, value in values.items())).encode('utf-8')

def decode_generation_request(artifact: OmniArtifact) -> CosmosGenerationRequest:
    metadata = dict(artifact.metadata)
    text = artifact.payload.decode('utf-8', errors='replace').strip()
    body: Dict[str, Any] = {}
    if text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                body = parsed
        except json.JSONDecodeError:
            body = {'prompt': text}
    prompt = str(body.get('prompt') or metadata.get('prompt') or '')
    if not prompt:
        raise ValueError('Omni artifact does not contain a prompt for Diffusers generation')
    request_id = str(body.get('request_id') or body.get('id') or metadata.get('request_id') or metadata.get('put_key') or artifact.put_key)
    return CosmosGenerationRequest(request_id=request_id, prompt=prompt, negative_prompt=_optional_str(body.get('negative_prompt', metadata.get('negative_prompt'))), height=int(body.get('height', metadata.get('height', 256))), width=int(body.get('width', metadata.get('width', 256))), num_frames=int(body.get('num_frames', metadata.get('num_frames', 1))), num_inference_steps=int(body.get('num_inference_steps', metadata.get('num_inference_steps', 35))), guidance_scale=float(body.get('guidance_scale', metadata.get('guidance_scale', 6.0))), flow_shift=float(body.get('flow_shift', metadata.get('flow_shift', 10.0))), seed=_optional_int(body.get('seed', metadata.get('seed'))), ar_conditioning=artifact.payload, metadata={**metadata, 'artifact_size': artifact.size, 'from_stage': artifact.from_stage})

def generate_from_request(request: CosmosGenerationRequest, pipeline: Callable[..., object], output_dir: Path) -> CosmosGenerationResult:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f'{request.request_id}.png'
    metadata_path = output_dir / f'{request.request_id}.json'
    call_kwargs: Dict[str, object] = {'prompt': request.prompt, 'negative_prompt': request.negative_prompt, 'num_frames': request.num_frames, 'height': request.height, 'width': request.width, 'num_inference_steps': request.num_inference_steps, 'guidance_scale': request.guidance_scale, 'enable_sound': False, 'output_type': 'pil', 'enable_safety_check': False}
    if request.seed is not None and _call_accepts_kwarg(pipeline, 'seed'):
        call_kwargs['seed'] = request.seed
    if _call_accepts_kwarg(pipeline, 'flow_shift'):
        call_kwargs['flow_shift'] = request.flow_shift
    output = pipeline(**call_kwargs)
    video = getattr(output, 'video', output)
    frame = video[0] if isinstance(video, list) else video
    if not hasattr(frame, 'save'):
        raise TypeError(f'pipeline output frame does not support save(): {type(frame)!r}')
    frame.save(image_path)
    seconds = time.perf_counter() - started
    metadata_path.write_text(json.dumps({'request_id': request.request_id, 'prompt': request.prompt, 'height': request.height, 'width': request.width, 'num_frames': request.num_frames, 'num_inference_steps': request.num_inference_steps, 'guidance_scale': request.guidance_scale, 'flow_shift': request.flow_shift, 'seed': request.seed, 'ar_conditioning_bytes': len(request.ar_conditioning), 'seconds': seconds, 'image_path': str(image_path)}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return CosmosGenerationResult(request, output_dir, image_path, metadata_path, seconds)

def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None

def _optional_int(value: object) -> Optional[int]:
    if value is None or value == '':
        return None
    return int(value)

def _call_accepts_kwarg(func: Callable[..., object], name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters

class ArtifactStore:

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._artifacts: Dict[Tuple[str, str, str], OmniArtifact] = {}

    @staticmethod
    def key(from_stage: str, to_stage: str, put_key: str) -> Tuple[str, str, str]:
        return (from_stage, to_stage, put_key)

    def put_artifact(self, artifact: OmniArtifact) -> Tuple[bool, int, Dict[str, object]]:
        with self._cv:
            self._artifacts[self.key(artifact.from_stage, artifact.to_stage, artifact.put_key)] = artifact
            self._cv.notify_all()
        return (True, artifact.size, dict(artifact.metadata))

    def put(self, from_stage: str, to_stage: str, put_key: str, data: bytes, metadata: Optional[Mapping[str, object]]=None) -> Tuple[bool, int, Dict[str, object]]:
        artifact = OmniArtifact(from_stage=from_stage, to_stage=to_stage, put_key=put_key, payload=bytes(data), metadata=dict(metadata or {}))
        return self.put_artifact(artifact)

    def get_artifact(self, from_stage: str, to_stage: str, put_key: str, *, timeout_s: Optional[float]=None, remove: bool=False) -> OmniArtifact:
        key = self.key(from_stage, to_stage, put_key)
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._cv:
            while key not in self._artifacts:
                if deadline is None:
                    self._cv.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'timed out waiting for Omni artifact {key}')
                self._cv.wait(remaining)
            artifact = self._artifacts[key]
            if remove:
                del self._artifacts[key]
            return artifact

    def get(self, from_stage: str, to_stage: str, put_key: str, *, timeout_s: Optional[float]=None, remove: bool=False) -> Tuple[bytes, int]:
        artifact = self.get_artifact(from_stage, to_stage, put_key, timeout_s=timeout_s, remove=remove)
        return (artifact.payload, artifact.size)

@dataclass
class _SessionAssembly:
    session_id: int
    metadata: Dict[str, object] = field(default_factory=dict)
    chunks: Dict[int, bytes] = field(default_factory=dict)
    done_streams: set[int] = field(default_factory=set)
    expected_streams: int = 1
    expected_bytes: Optional[int] = None
    finalized: bool = False

    def update_metadata(self, values: Mapping[str, str]) -> None:
        self.metadata.update(values)
        if 'streams' in values:
            self.expected_streams = max(1, int(values['streams']))
        if 'stream_count' in values:
            self.expected_streams = max(1, int(values['stream_count']))
        if 'bytes' in values:
            self.expected_bytes = int(values['bytes'])
        if 'payload_bytes' in values:
            self.expected_bytes = int(values['payload_bytes'])

    def add_chunk(self, frame: TBPFrame) -> None:
        self.chunks[int(frame.seq_no)] = frame.payload

    def mark_done(self, frame: TBPFrame) -> bool:
        self.done_streams.add(int(frame.stream_id or 1))
        return len(self.done_streams) >= self.expected_streams

    def to_artifact(self) -> OmniArtifact:
        payload = b''.join((self.chunks[seq_no] for seq_no in sorted(self.chunks)))
        if self.expected_bytes is not None and len(payload) != self.expected_bytes:
            raise TBPError(f'session {self.session_id} reassembled {len(payload)} bytes, expected {self.expected_bytes}')
        from_stage = str(self.metadata.get('from_stage', 'rtx_ar'))
        to_stage = str(self.metadata.get('to_stage', 'mac_cosmos'))
        put_key = str(self.metadata.get('put_key', self.metadata.get('artifact', f'session_{self.session_id:016x}')))
        metadata = dict(self.metadata)
        metadata.update({'session_id': self.session_id, 'bytes_received': len(payload), 'chunks_received': len(self.chunks), 'streams_done': len(self.done_streams)})
        return OmniArtifact(from_stage, to_stage, put_key, payload, metadata)

class OmniConnectorReceiver:

    def __init__(self, host: str='0.0.0.0', port: int=9100, *, timeout_s: float=30.0, validate_crc: bool=True, store: Optional[ArtifactStore]=None) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.validate_crc = validate_crc
        self.store = store or ArtifactStore()
        self._stop = threading.Event()
        self._listen_sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._assemblies: Dict[int, _SessionAssembly] = {}
        self._assemblies_lock = threading.Lock()

    @property
    def bound_port(self) -> int:
        if self._listen_sock is None:
            return self.port
        return int(self._listen_sock.getsockname()[1])

    def put(self, from_stage: str, to_stage: str, put_key: str, data: bytes, metadata: Optional[Mapping[str, object]]=None) -> Tuple[bool, int, Dict[str, object]]:
        return self.store.put(from_stage, to_stage, put_key, data, metadata)

    def get(self, from_stage: str, to_stage: str, put_key: str, *, timeout_s: Optional[float]=None, remove: bool=False) -> Tuple[bytes, int]:
        return self.store.get(from_stage, to_stage, put_key, timeout_s=timeout_s, remove=remove)

    def get_artifact(self, from_stage: str, to_stage: str, put_key: str, *, timeout_s: Optional[float]=None, remove: bool=False) -> OmniArtifact:
        return self.store.get_artifact(from_stage, to_stage, put_key, timeout_s=timeout_s, remove=remove)

    def generate_from_artifact(self, from_stage: str, to_stage: str, put_key: str, *, pipeline: Callable[..., object], output_dir: Path, timeout_s: Optional[float]=None, remove: bool=False) -> CosmosGenerationResult:
        artifact = self.get_artifact(from_stage, to_stage, put_key, timeout_s=timeout_s, remove=remove)
        request = decode_generation_request(artifact)
        return generate_from_request(request, pipeline, output_dir)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listen_sock.bind((self.host, self.port))
        self._listen_sock.listen()
        self._listen_sock.settimeout(0.2)
        self._stop.clear()
        self._thread = threading.Thread(target=self.serve_forever, name='cosmos-omni-recv', daemon=True)
        self._thread.start()

    def serve_forever(self) -> None:
        if self._listen_sock is None:
            raise RuntimeError('receiver must be started before serve_forever')
        while not self._stop.is_set():
            try:
                client, _addr = self._listen_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                raise
            thread = threading.Thread(target=self._handle_client, args=(client,), daemon=True)
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except OSError:
                pass
            self._listen_sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _assembly_for(self, session_id: int) -> _SessionAssembly:
        with self._assemblies_lock:
            assembly = self._assemblies.get(session_id)
            if assembly is None:
                assembly = _SessionAssembly(session_id=session_id)
                self._assemblies[session_id] = assembly
            return assembly

    def _finalize_if_ready(self, assembly: _SessionAssembly, frame: TBPFrame) -> None:
        with self._assemblies_lock:
            if assembly.finalized or not assembly.mark_done(frame):
                return
            artifact = assembly.to_artifact()
            assembly.finalized = True
            self._assemblies.pop(assembly.session_id, None)
        self.store.put_artifact(artifact)

    def _handle_client(self, client: socket.socket) -> None:
        with client:
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            current: Optional[_SessionAssembly] = None
            while not self._stop.is_set():
                try:
                    frame = read_tbp_frame(client, timeout_s=self.timeout_s, validate_crc=self.validate_crc)
                except TBPError:
                    return
                current = self._assembly_for(frame.session_id)
                if frame.msg_type in (TBP_MSG_SESSION_START, TBP_MSG_KV_SEG_BEGIN):
                    current.update_metadata(parse_kv_pairs(frame.payload))
                elif frame.msg_type == TBP_MSG_KV_CHUNK:
                    current.add_chunk(frame)
                elif frame.msg_type == TBP_MSG_KV_DONE:
                    self._finalize_if_ready(current, frame)
                    return
                elif frame.msg_type in (TBP_MSG_HELLO, TBP_MSG_KV_SEG_END, TBP_MSG_KV_ACK):
                    continue

def send_tbp1_artifact(host: str, port: int, payload: bytes, *, from_stage: str='rtx_ar', to_stage: str='mac_cosmos', put_key: str='conditioning', session_id: int=1, stream_id: int=1, chunk_bytes: int=4 * 1024 * 1024, extra_metadata: Optional[Mapping[str, object]]=None) -> None:
    metadata: Dict[str, object] = {'mode': 'bulk', 'artifact': put_key, 'from_stage': from_stage, 'to_stage': to_stage, 'put_key': put_key, 'bytes': len(payload), 'streams': 1}
    if extra_metadata:
        metadata.update(extra_metadata)
    seq = 0
    with socket.create_connection((host, port), timeout=5.0) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.sendall(build_tbp_frame(TBP_MSG_HELLO, b'node_role=prefill-server;capabilities=bulk', session_id=session_id, stream_id=0, seq_no=seq))
        seq += 1
        sock.sendall(build_tbp_frame(TBP_MSG_SESSION_START, format_kv_pairs(metadata), session_id=session_id, stream_id=0, seq_no=seq))
        seq += 1
        sock.sendall(build_tbp_frame(TBP_MSG_KV_SEG_BEGIN, format_kv_pairs({'segment_id': 0, 'payload_bytes': len(payload), 'chunk_bytes_nominal': chunk_bytes, 'stream_count': 1}), session_id=session_id, stream_id=stream_id, seq_no=seq))
        seq += 1
        for offset in range(0, len(payload), chunk_bytes):
            chunk = payload[offset:offset + chunk_bytes]
            sock.sendall(build_tbp_frame(TBP_MSG_KV_CHUNK, chunk, session_id=session_id, stream_id=stream_id, seq_no=seq))
            seq += 1
        sock.sendall(build_tbp_frame(TBP_MSG_KV_SEG_END, format_kv_pairs({'segment_id': 0}), session_id=session_id, stream_id=stream_id, seq_no=seq))
        seq += 1
        sock.sendall(build_tbp_frame(TBP_MSG_KV_DONE, b'done=1', session_id=session_id, stream_id=stream_id, seq_no=seq))

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9100)
    args = parser.parse_args()
    receiver = OmniConnectorReceiver(args.host, args.port)
    receiver.start()
    print(f'Cosmos OmniConnector receiver listening on {args.host}:{receiver.bound_port}', flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        receiver.stop()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
