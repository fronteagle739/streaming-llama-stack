"""QFN layout/lifecycle adapter for the shared native installation contract."""
import json,re,threading
from pathlib import Path

from .mac import Receiver,source_digest
from .config import ROOT,path
from .journal import atomic_json,sha256
from .model_compatibility import digest
from .gguf_header import read_header


def installation_digest():
    files=[*sorted((ROOT/'slstack').glob('*.py')),ROOT/'examples/dual-machine/qfn/native.py',
           ROOT/'examples/dual-machine/qfn/native_vision.py',ROOT/'examples/dual-machine/qfn/native_pair.py',ROOT/'envvars.json']
    return digest({p.relative_to(ROOT).as_posix():sha256(p) for p in files})


def launch_arguments(argv):
    """Remove only per-run output location and the self-referencing contract path."""
    values=[];i=0
    while i<len(argv):
        if argv[i] in ('--output','--installation-profile-file'):
            if i+1>=len(argv):raise ValueError('Incomplete native launch argument')
            i+=2;continue
        if argv[i]=='--dry-run':i+=1;continue
        values.append(argv[i]);i+=1
    return values


def validate_launch_contract(file,role,argv):
    row=json.loads(Path(file).read_text(encoding='utf-8'))
    if set(row)!={'profile','model_settings_sha256','role_argv'} or set(row['role_argv'])!={'rtx','mac'}:
        raise ValueError('Unknown or incomplete installation preset contract')
    if launch_arguments(argv)!=row['role_argv'][role]:raise ValueError('Actual peer flags differ from the complete resolved installation preset')
    return digest(row)


class QfnReceiver(Receiver):
    def __init__(self,cfg,build,*,describe_model,start_engine,retire_engine,current_engine,
                 command_for_mode,lifecycle_lock,producer_source_sha256,preset_sha256):
        self._describe_model=describe_model;self._start=start_engine;self._retire=retire_engine
        self._current=current_engine;self._command=command_for_mode;self._life=lifecycle_lock
        self._producer_source=producer_source_sha256;self._preset=preset_sha256
        self._installation=installation_digest();self._description=describe_model()
        for value in (producer_source_sha256,preset_sha256):
            if not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('QFN source/preset pin missing')
        self._vocab=len(read_header(path(cfg['model']))['metadata']['tokenizer.ggml.tokens'])
        if not 1<=self._vocab<=10000000:raise ValueError('QFN qualification vocabulary is invalid')
        if not self.valid_pair_token(cfg.get('pair_token')):raise ValueError('QFN installation requires its explicit private hex pairing token')
        super().__init__(cfg,'ds4',build)
        if self.peer and any(self.peer.get(k)!=v for k,v in self._expected_extra().items()):
            self.peer['status']='stale'
            print('QFN_INSTALLATION_REQUALIFY reason=source_model_renderer_or_preset_changed',flush=True)

    def valid_pair_token(self,value):
        return isinstance(value,str) and 32<=len(value)<=512 and all(c in '0123456789abcdef' for c in value)

    def pair_extra_fields(self):
        return set(self._expected_extra())

    def _expected_extra(self):
        return dict(producer_source_sha256=self._producer_source,receiver_source_sha256=self.build['source_sha256'],
            renderer_sha256=self.build['renderer_sha256'],preset_sha256=self._preset,
            receiver_model_sha256=self._description['model_sha256'],installation_sha256=self._installation)

    def health_fields(self):
        return dict(family='qwen4exp',receiver_model_compatibility=self._description,**self._expected_extra())

    def validate_pair_extra(self,row):
        expected=self._expected_extra()
        if (row['producer_sha256'],row['producer_source_revision'])!=(self.cfg['producer_sha256'],self.cfg['producer_source_revision']):
            raise ValueError('QFN producer executable/revision differs from the selected installation')
        if {k:row[k] for k in expected}!=expected:raise ValueError('QFN installation source/renderer/preset pins differ')
        if row['tier']!='accurate':raise ValueError('QFN uses its explicit family preset; DSV-only tier algorithms are not substituted')
        self._assert_build(check_peer=False)

    def local_model_description(self,keep):
        if keep:raise ValueError('QFN retains its complete family state; DeepSeek YOCO is a different layout')
        return self._describe_model()

    def normalize_request(self,body):
        from .paired_generation import decode_request
        return decode_request(body,vocab=self._vocab,top_k_max=self._vocab,bias_max=8)

    def qualification_vocab(self):
        return self._vocab

    def qualification_directory(self):
        return self.token_file.parent

    def _assert_build(self,check_peer=True):
        if installation_digest()!=self._installation:
            raise ValueError('QFN installation controller source changed after launch')
        if sha256(Path(self.build['binary']))!=self.build['binary_sha256']:
            raise ValueError('QFN receiver binary changed after installation build')
        if source_digest(ROOT/'engines/ds4-mac')!=self.build['source_sha256']:
            raise ValueError('QFN receiver source changed after its build')
        if sha256(ROOT/'engines/ds4-mac/ds4_server.c')!=self.build['renderer_sha256']:
            raise ValueError('QFN renderer changed after its build')
        if check_peer and self.peer and self._describe_model()!=self.peer['receiver_model_compatibility']:
            raise ValueError('QFN receiver model changed after pairing')

    def ready(self,native):
        with self._life:
            self._assert_build()
            if not self.peer or self.peer.get('status') not in ('candidate','qualified'):raise ValueError('Pair and qualify the QFN producer before loading a decoder')
            current=self._current()
            if current is not None and current.proc.poll() is None and self.engine_mode==native:
                self.engine=current;return
            self.close()
            model=path(self.cfg['model']);st=model.stat()
            state=self.qualification_directory();state.mkdir(parents=True,exist_ok=True)
            artifact=state/'qfn-native-qualification.json'
            row=dict(schema='qfn-native-session-qualification-v1',state=self.peer['status'],
                model_sha256=self.peer['model_compatibility']['model_sha256'],
                receiver_model_sha256=self.peer['receiver_model_compatibility']['model_sha256'],
                compatibility_sha256=self.peer['model_compatibility']['compatibility_sha256'],
                mode=3,wire_version=2,tokenization='native-final-token-ids-le32-v1',
                producer_binary_sha256=self.peer['producer_sha256'],producer_source_revision=self.peer['producer_source_revision'],
                receiver_binary_sha256=self.build['binary_sha256'],renderer_sha256=self.build['renderer_sha256'],
                receiver_model_size=st.st_size,receiver_model_device=st.st_dev,receiver_model_inode=st.st_ino,
                receiver_model_mtime_ns=st.st_mtime_ns)
            atomic_json(artifact,row)
            command=self._command(native,artifact,sha256(artifact),self.peer,state/'qualification-current.f32')
            self._start(command)
            self.engine=self._current();self.engine_mode=native;self.engine_budget=self.cfg['host_budget_gib']
            atomic_json(state/'qfn-active-contract.json',dict(family='qwen4exp',mode='paired' if native else 'native',
                qualification=artifact.name,qualification_sha256=sha256(artifact),pins=self.peer,command=command))

    def close(self):
        with self._life:
            if self._current() is not None:self._retire()
            self.engine=None;self.engine_mode=None;self.engine_budget=None

    def accept(self):
        self._assert_build()
        result=super().accept()
        result.update(family='qwen4exp',source_pins_verified=True,renderer_pin_verified=True,
            preset_sha256=self._preset,qualification_vocab=self._vocab)
        self.peer.update(result)
        atomic_json(self.token_file.with_name('peer.json'),self.peer)
        return result
