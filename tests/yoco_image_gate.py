"""Numerical YOCO image qualification using one real paired projection fixture.

Run only after ejecting the live pair. No other model process is started on Mac.
The fixture is a broker session directory containing initial.tokens.le32/vision.
Each arm uses the shipping native producer and drains its authenticated export;
only the frontier logit row is retained, never a disk KV artifact.
"""
import argparse,array,hashlib,json,math,os,secrets,struct,sys,threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slstack.config import load
from slstack.guard import admit
from slstack.native_broker import OwnedProducer,vendor
from slstack.ds4_render import Tokenizer
from slstack.yoco import state_version
vendor()
from studio.dsv41_session_contract import MODEL_SHA256


def frontier(cfg,raw,images,keep):
    admit(cfg['host_budget_gib'],cfg['min_available_gib'])
    session=secrets.token_hex(16)
    request=dict(session=session,tier='ultra',tokens=raw,tokens_sha256=hashlib.sha256(raw).hexdigest(),
                 yoco_keep=keep,yoco_state_version=state_version(keep),vision=images)
    owned=OwnedProducer(dict(cfg,yoco_keep=keep),request,threading.Event())
    try:
        p=owned.producer;cp=p.ready['checkpoint']
        offset=60+len(raw);end=offset+129280*4;position=0;logits=bytearray()
        p.begin_export(dict(session=session,model_sha256=MODEL_SHA256,sequence=p.sequence,
                            operation='export_initial',base=cp))
        while block:=p.read_export():
            left=max(0,offset-position);right=min(len(block),end-position)
            if right>left:logits.extend(block[left:right])
            position+=len(block)
        p.end_export()
        assert len(logits)==129280*4
        assert hashlib.sha256(logits).hexdigest()==cp['logits_sha256']
        values=array.array('f');values.frombytes(logits)
        if sys.byteorder!='little':values.byteswap()
        assert all(math.isfinite(v) for v in values)
        return values,dict(position=cp['position'],logits_sha256=cp['logits_sha256'],export_bytes=position)
    finally:owned.close()


def compare(full,yoco):
    mf=max(full);my=max(yoco)
    zf=mf+math.log(sum(math.exp(v-mf) for v in full))
    zy=my+math.log(sum(math.exp(v-my) for v in yoco))
    kl=sum(math.exp(f-zf)*((f-zf)-(y-zy)) for f,y in zip(full,yoco))
    tf=sorted(range(len(full)),key=full.__getitem__,reverse=True)[:10]
    ty=sorted(range(len(yoco)),key=yoco.__getitem__,reverse=True)[:10]
    return dict(kl_full_to_yoco=kl,top10_recall=len(set(tf)&set(ty))/10,
                full_argmax=tf[0],yoco_argmax=ty[0],full_argmax_in_yoco_top10=tf[0] in ty,
                full_top10=tf,yoco_top10=ty)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--fixture',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--approximate-yoco',action='store_true',help='Opt into legacy approximate140 probability-drift qualification')
    args=parser.parse_args();cfg=load(args.config);fixture=Path(args.fixture)
    keep=140 if args.approximate_yoco else 2541
    cfg.pop('_gpu_parent_lease',None)  # the live GUI pair has been gracefully ejected
    os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid'];os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    original=(fixture/'initial.tokens.le32').read_bytes()
    descriptors=[json.loads(p.read_text()) for p in sorted((fixture/'vision').glob('*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)]
    rows=next(rows for rows in descriptors if rows)
    images=[dict(position=r['position'],count=r['count'],sha256=r['sha256'],data=Path(r['file']).read_bytes()) for r in rows]
    assert len(images)==1,'Use the one-image GUI fixture'
    image=images[0];image_end=image['position']+image['count']
    tok=Tokenizer.from_gguf(cfg['model']);filler=tok.rendered(' The weather is clear today.'*100)
    filler=filler*(-(-keep//len(filler))+1)
    ids=list(struct.unpack('<%dI'%(len(original)//4),original))
    assert all(t==129264 for t in ids[image['position']:image_end]),'Fixture image span must cover its marker IDs'
    assert sum(t==129264 for t in ids)==image['count'],'Fixture has unbound image markers'
    result=dict(status='INCOMPLETE',scope='RTX full evaluation vs bounded image frontier',yoco_keep=keep,yoco_state_version=state_version(keep),cases={})
    try:
        for name,extra in [('intersects_tail',[]),('before_tail',filler)]:
            tokens=ids[:image_end]+extra+ids[image_end:]
            assert (image_end>len(tokens)-keep) if not extra else (image_end<=len(tokens)-keep)
            raw=struct.pack('<%dI'%len(tokens),*tokens)
            full,full_meta=frontier(cfg,raw,images,0)
            yoco,yoco_meta=frontier(cfg,raw,images,keep)
            comparison=compare(full,yoco)
            comparison['logits_exact']=full_meta['logits_sha256']==yoco_meta['logits_sha256']
            comparison['pass']=(comparison['kl_full_to_yoco']<=.1 and comparison['full_argmax_in_yoco_top10']) if keep==140 else comparison['logits_exact']
            result['cases'][name]=dict(tokens=len(tokens),image_start=image['position'],image_end=image_end,
                                      full=full_meta,yoco=yoco_meta,**comparison)
            print(name,json.dumps(result['cases'][name]),flush=True)
        result['status']='PASS' if all(c['pass'] for c in result['cases'].values()) else 'FAIL'
    except BaseException as exc:result.update(status='FAIL',error=repr(exc));raise
    finally:Path(args.out).write_text(json.dumps(result,indent=2))
    return result['status']=='PASS'


if __name__=='__main__':raise SystemExit(0 if main() else 1)
