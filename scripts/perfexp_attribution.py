"""Validate synchronizing E2c layer traces; report first hidden/route divergence.

Input is DS4COMBO_V41_DUMP, not a timing receipt. Both arms must cover all40
layers at the same single append position, including layer0 input and pre seed.
"""
from __future__ import annotations
import argparse
import array
import json
import math
from pathlib import Path
import struct
import sys

WORDS={0:5120,1:20480,2:5120,3:6,4:6,5:5120,6:20480,7:32768,
       8:512,9:32768,10:5120,11:20480,12:4}
NAMES={0:'attention_input',1:'after_attention',2:'ffn_input',3:'expert_ids',
       6:'layer_output',7:'query_rope',8:'raw_kv_qat',9:'heads_inverse_rope',10:'attention_projection',11:'layer_input',12:'initial_pre'}

def load(path):
    path=Path(path)
    if path.stat().st_size>256*1024*1024:raise ValueError('Trace exceeds256MiB bound')
    records={};positions=set()
    with path.open('rb') as source:
        while header:=source.read(20):
            if len(header)!=20:raise ValueError('Truncated trace header')
            magic,layer,pos,point,words=struct.unpack('<5I',header)
            if magic!=0x52313444 or layer>=40 or point not in WORDS or words!=WORDS[point]:
                raise ValueError('Invalid trace record identity/shape')
            key=(layer,pos,point)
            if key in records:raise ValueError('Duplicate trace key')
            raw=source.read(words*4)
            if len(raw)!=words*4:raise ValueError('Truncated trace payload')
            values=array.array('i' if point==3 else 'f');values.frombytes(raw)
            if sys.byteorder!='little':values.byteswap()
            if point==3:
                if len(set(values))!=6 or any(x<0 or x>=384 for x in values):raise ValueError('Invalid routed expert IDs')
            elif any(not math.isfinite(x) for x in values):raise ValueError('Nonfinite trace value')
            records[key]=values;positions.add(pos)
            if len(positions)>16:raise ValueError('More than16 trace positions')
    if not records:raise ValueError('Empty trace')
    return records

def compare(left,right):
    positions={k[1] for k in left}|{k[1] for k in right}
    if len(positions)!=1:raise ValueError('Compare exactly one append position')
    pos=next(iter(positions))
    required={(layer,pos,point) for layer in range(40) for point in (0,1,2,3,6,11)}|{(0,pos,12)}
    if not required<=left.keys() or not required<=right.keys():raise ValueError('Incomplete40-layer attribution coverage')
    rows=[];first_hidden=None;first_routes=None;first_order=None
    for layer in range(40):
        for point in (11,12,0,7,8,9,10,1,2,3,6):
            key=(layer,pos,point)
            if key not in required and not (point in (7,8,9,10) and key in left and key in right):continue
            a,b=left[key],right[key]
            exact=a.tobytes()==b.tobytes()
            row=dict(layer=layer,point=NAMES[point],bit_exact=exact)
            if point==3:
                row.update(control=list(a),candidate=list(b),same_set=set(a)==set(b),same_order=list(a)==list(b))
                if not row['same_set'] and first_routes is None:first_routes=dict(row)
                if not row['same_order'] and first_order is None:first_order=dict(row)
            else:
                error=sum((float(x)-float(y))**2 for x,y in zip(a,b))
                norm=sum(float(x)**2 for x in a)
                row.update(max_abs=max(abs(float(x)-float(y)) for x,y in zip(a,b)),relative_l2=math.sqrt(error/max(norm,1e-300)))
                if not exact and first_hidden is None:first_hidden=dict(row)
            rows.append(row)
    return dict(position=pos,diagnostic_only=True,first_hidden_divergence=first_hidden,
                first_expert_set_divergence=first_routes,
                first_expert_order_divergence=first_order,
                same_expert_sets=sum(x.get('same_set',False) for x in rows),layers=40,rows=rows)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('control',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--control-receipt',required=True,type=Path);p.add_argument('--candidate-receipt',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    control=json.loads(args.control_receipt.read_text());candidate=json.loads(args.candidate_receipt.read_text())
    for name in ('day','model','model_bytes','model_mtime_ns','input_sha256','append_input_sha256',
                 'producer_sha256','decode_dll_sha256','append_base_checkpoint'):
        if name not in control or name not in candidate or control[name]!=candidate[name]:
            raise ValueError('Unmatched attribution receipts: '+name)
    for receipt in (control,candidate):
        if receipt.get('status')!='PASS' or not receipt.get('trace_append') or receipt.get('append_quality_bias')!=0:
            raise ValueError('Attribution requires completed bias0 diagnostic receipts')
    result=compare(load(args.control),load(args.candidate))
    if result['position']!=control['append_base_checkpoint']['position']:raise ValueError('Trace position differs from saved append boundary')
    result['identical_base_checkpoint']=True
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
