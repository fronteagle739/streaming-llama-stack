"""Bank, capture-format and KV identity regressions; no models or GPU."""
import hashlib
import struct
from types import SimpleNamespace
import pytest
from slstack import writer_directions as w
from slstack.local_cache import Cache


def registry(tmp_path):
    return w.Registry('unused',tmp_path,model_info=dict(layers=40,hidden=5120,native_id='a'*64))


def packet(identity='a'*64):
    site=struct.pack('<2f',1.,1.)+bytes(5119*4)
    return struct.pack('<4I',1,1,40,5120)+bytes.fromhex(identity)+site*80


def test_registration_and_target_bank_are_immutable_and_model_bound(tmp_path):
    bank=registry(tmp_path)
    raw=packet();identifier=bank.register(raw)['id']
    assert identifier==hashlib.sha256(raw).hexdigest()
    assert bank.register(raw)['id']==identifier
    with pytest.raises(ValueError,match='different model'):bank.register(packet('b'*64))
    selected=bank.select([dict(id=identifier,strength=.5)])
    assert selected['count']==1 and selected['key']!=identifier
    assert w.bank_arguments(dict(direction_bank_file=selected['path'],direction_bank_sha256=selected['key']))['--direction-bank-file']==selected['path']
    assert bank.select([dict(id=identifier,strength=0)])['key']==''
    from pathlib import Path
    Path(selected['path']).write_bytes(b'altered')
    with pytest.raises(ValueError):bank.select([dict(id=identifier,strength=.5)])


def test_bank_change_retires_the_old_frontier_and_preserves_identical_bank(tmp_path):
    bank=registry(tmp_path);identifier=bank.register(packet())['id'];events=[]
    engine=SimpleNamespace(cfg={},producer=object(),pending_snapshot='old',tool_memory={'old':True},cut_turn='old',
        directions=lambda:bank,save_snapshot=lambda reason:events.append('save'),retire=lambda reason:events.append('retire'))
    chosen=[dict(id=identifier,strength=1.)]
    w.apply_selection(engine,chosen)
    assert events==['save','retire'] and engine.pending_snapshot is None and engine.tool_memory=={}
    w.apply_selection(engine,chosen)
    assert events==['save','retire']
    w.apply_selection(engine,[])
    assert events==['save','retire','save','retire'] and engine.cfg['direction_bank_sha256']==''


def test_capture_groups_and_exact_offsets_preserve_the_wire_contract():
    raw=struct.pack('<11I',2,3,2,2,4,5,6,0,0,2,1)
    tokens,plan=w.capture_plan(raw)
    assert tokens==[4,5,6] and plan['positions']==[0,2] and plan['groups']==[0,1]
    assert w.capture_headers(plan)['X-DS4-Aggregation']=='mean'
    with pytest.raises(ValueError):w.capture_plan(struct.pack('<11I',2,3,2,2,4,5,6,2,0,0,1))
    tokenizer=SimpleNamespace(text=lambda text:[4,5],token_bytes=lambda token:{4:b'a',5:'\u00e9'.encode()}[token])
    assert w.token_offsets(tokenizer,'a\u00e9')==dict(tokens=[4,5],byte_offsets=[[0,1],[1,3]],text_bytes=3)
    with pytest.raises(ValueError):w.token_offsets(tokenizer,'wrong')


def test_ready_acknowledges_exact_bank_and_caches_bind_the_same_identity():
    cfg=dict(producer_sha256='a'*64,writer_directions_qualified_sha256='a'*64)
    with pytest.raises(ValueError):w.verify_ready(cfg,{})
    w.verify_ready(cfg,dict(direction_binding=''))
    cfg.update(direction_bank_sha256='b'*64)
    with pytest.raises(ValueError):w.verify_ready(cfg,dict(direction_binding=''))
    w.verify_ready(cfg,dict(direction_binding='b'*64))
    cache=Cache(cfg)
    assert cache.provenance()['direction_binding']=='b'*64
    cfg['direction_bank_sha256']='c'*64
    assert cache.provenance()['direction_binding']=='c'*64
    assert w.device_reserve(0)==0 and w.device_reserve(8)>w.device_reserve(1)>4*1024**3
    with pytest.raises(ValueError):w.device_reserve(9)
