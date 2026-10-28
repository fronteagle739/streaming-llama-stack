#!/usr/bin/env python3
"""Wrap pinned antirez/ds4 planners for Q4 or native-FP4 experts/Q8 dense.

Dry-run requires only config.json and model.safetensors.index.json. It never
opens safetensors, a quantizer library, a tokenizer, or tensor payloads.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RECIPE = Path(__file__).with_name('dsv41')
sys.path.insert(0,str(ROOT))
from slstack.gguf_header import read_header, tensor_map, TYPE_NAMES


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_tools(directory, fetch=False):
    pin = read_json(RECIPE/'antirez-ds4.json')
    directory = Path(directory).resolve()
    if fetch:directory.mkdir(parents=True,exist_ok=True)
    for name,digest in pin['files'].items():
        file = directory/name
        if fetch and not file.exists():
            url = f"https://raw.githubusercontent.com/antirez/ds4/{pin['commit']}/gguf-tools/{name}"
            with urllib.request.urlopen(url) as response:
                data = response.read(2 << 20)
            if hashlib.sha256(data).hexdigest() != digest:raise ValueError('Upstream source SHA mismatch: '+name)
            with file.open('xb') as output:output.write(data)
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned antirez/ds4 source required: '+str(file))
    sys.path.insert(0,str(directory))
    return {name:importlib.import_module(name) for name in
            ('deepseek41_quantize','deepseek41_dspark','deepseek41_vision','glm53_quantize')}


class IndexPlanDB:
    """Config-derived shape assertions against names in an existing source index.

    The source index has no shape/dtype information. These are predictions from
    upstream's unchanged shape expressions, verified against real source headers
    only during conversion. No dry-run claim implies a source-payload check.
    """
    def __init__(self,index):
        self.tensors = dict(index['weight_map'])
        self.claimed = {}
    def claim(self,name,shape,dtype=None):
        if name not in self.tensors:raise ValueError('Missing indexed source tensor '+name)
        dtype = dtype or ('F8_E4M3' if name.removesuffix('.weight')+'.scale' in self.tensors else 'F32')
        info = dict(shape=list(shape),dtype=dtype)
        self.claimed[name] = info
        if dtype in ('I8','F8_E4M3') and name.endswith('.weight'):
            scale = name.removesuffix('.weight')+'.scale'
            if scale not in self.tensors:raise ValueError('Missing indexed source scale '+scale)
        return info
    def info(self,name):return self.claimed[name]


def index_planner(module):
    """Adapt ONLY shard-dependent claim validation; retain upstream plan logic.

    Replacing the nested `claim` with an index assertion makes the upstream
    declarative plan usable without ~475 GiB of shards. Real conversion calls
    the original build_plan and SourceDB, including all original shape checks.
    """
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_plan')
    class Claims(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if node.name != 'claim':return self.generic_visit(node)
            shape = node.args.args[1].arg
            node.body = ast.parse(f'''info = db.claim(name, {shape}, dtype)
consumed.add(name)
if info['dtype'] in ('I8','F8_E4M3') and name.endswith('.weight'):
    consumed.add(scale_name(name))
return info
''').body
            return node
    function = Claims().visit(function)
    namespace = dict(vars(module))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),module.__file__,'exec'),namespace)
    return namespace['build_plan']


def apply_layout(plan,reference,quantizer,alignment=16384):
    """The per-tensor recipe is extracted from the owner's GGUF header."""
    if {p.name for p in plan} != set(reference):raise ValueError('Planned tensor names differ from reference header')
    offset = 0
    for item in plan:
        row = reference[item.name]
        if list(item.shape) != row['shape']:raise ValueError('Planned shape differs: '+item.name)
        item.qtype = row['type']
        item.nbytes = quantizer.qtype_nbytes(item.qtype,item.shape)
        item.offset = offset
        offset += quantizer.align(item.nbytes,alignment)
    return plan


def make_plan(component,modules,config,index,reference,db=None):
    q,d,v,g = (modules[n] for n in ('deepseek41_quantize','deepseek41_dspark','deepseek41_vision','glm53_quantize'))
    dry = db is None
    db = IndexPlanDB(index) if dry else db
    if component == 'main':
        plan = (index_planner(q) if dry else q.build_plan)(db,config,'q4')
    elif component == 'dspark':
        # The public planner determines names/shapes. Its newer lossless recipe
        # is overridden by our header's Q4/Q8/F16/F32 map, using the SAME C
        # quantizer and streaming GGUF writer as the main model.
        class LayoutTypes:
            def dense_type(self,db,name):return g.QTYPE_F16
        plan = (index_planner(d) if dry else d.build_plan)(db,config,LayoutTypes())
    else:
        if dry:
            v.validate_inventory(db.tensors)
            for name,shape in v.source_shapes().items():
                db.claimed[name] = dict(shape=list(shape),dtype='BF16',nbytes=2*g.product(shape))
        plan = v.build_plan(db)
    return apply_layout(plan,reference,g,32 if component=='vision' else 16384)


def planned_map(plan):return {p.name:dict(type=p.qtype,shape=list(p.shape)) for p in plan}


def native_layout(reference):
    """Keep the measured layout except the 120 main routed expert tensors."""
    return {name:dict(row,type=39) if name.endswith(('_gate_exps.weight','_up_exps.weight','_down_exps.weight'))
            else dict(row) for name,row in reference.items()}


def pack_native_expert(db,name):
    """Reorder packed E2M1 nibbles and interleave unchanged E8M0 scales.

    Source byte j holds values 2*j and 2*j+1. GGML byte j holds values j
    and j+16. Both use the same E2M1 codebook and one E8M0 scale per 32;
    no floating point dequantization, rounding or scale adjustment occurs.
    """
    import numpy as np
    info,scale = db.info(name),db.info(name.removesuffix('.weight')+'.scale')
    shape = info['shape']
    if info['dtype']!='I8' or len(shape)!=2 or shape[1]%16:
        raise ValueError('Native FP4 expert requires packed I8 rows divisible by16: '+name)
    if scale['dtype']!='F8_E8M0' or scale['shape']!=[shape[0],shape[1]//16]:
        raise ValueError('Native FP4 expert requires E8M0 per32 scales: '+name)
    packed=np.frombuffer(db.read(name),dtype=np.uint8).reshape(-1,16)
    scales=np.frombuffer(db.read(name.removesuffix('.weight')+'.scale'),dtype=np.uint8).reshape(-1)
    if np.any(scales==255):raise ValueError('Nonfinite native FP4 scale: '+name)
    result=np.empty((len(scales),17),dtype=np.uint8)
    result[:,0]=scales
    result[:,1::2]=(packed[:,:8]&15)|((packed[:,8:]&15)<<4)
    result[:,2::2]=(packed[:,:8]>>4)|(packed[:,8:]&240)
    return result.tobytes()


def native_writer(module):
    """Add one expert-copy branch to the hash-pinned upstream streaming writer.

    Retain its dense quantizer, Engram copier, alignment, durable resume journal,
    disk reserve and output checks. The fetched upstream files stay unchanged.
    Like index_planner(), this adapter operates on a pinned function's AST.
    """
    tree=ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='write_gguf')
    callbacks=[n for n in ast.walk(function) if isinstance(n,ast.FunctionDef) and n.name=='convert']
    if len(callbacks)!=1:raise ValueError('Pinned writer expert callback changed')
    callbacks[0].body[:0]=ast.parse('''if item.qtype == 39:
    return pack_native_expert(db, item.source.format(expert=expert)), False
''').body
    namespace=dict(vars(module),pack_native_expert=pack_native_expert)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),module.__file__,'exec'),namespace)
    return namespace['write_gguf']


def convert(args,modules,config,index,reference):
    q,d,v,g = (modules[n] for n in ('deepseek41_quantize','deepseek41_dspark','deepseek41_vision','glm53_quantize'))
    db = g.SourceDB(str(args.source),index_validator=lambda _:None,scale_validator=q.validate_scales)
    try:
        plan = make_plan(args.component,modules,config,index,reference,db)
        if args.component == 'main':
            _,records = q.metadata(str(args.source),args.source_revision)
            native=args.expert_format=='native-fp4'
            records += [g.kv_string('deepseek41.quantization',
                        'native E2M1/E8M0 MXFP4 routed experts; Q8_0 dense' if native else q.QUANTIZATION['q4']),
                        g.kv_string('deepseek41.calibration',
                        'lossless expert repack; weight-energy bootstrap dense' if native else 'weight-energy bootstrap')]
        elif args.component == 'dspark':
            records = d.metadata(config,args.source_revision)
            # Remove the newer lossless recipe's encoding assertion.
            records = [record for record in records if b'dspark.weight_encoding' not in record]
            records += [g.kv_string('deepseek41.dspark.quantization','Q4_K routed experts; Q8_0 attention/shared/main_proj/markov'),
                        g.kv_string('deepseek41.dspark.calibration','weight-energy bootstrap')]
        else:
            v.SOURCE_REVISION=args.source_revision
            records = v.sidecar_metadata(str(args.source))
            v.create_gguf(str(args.out),plan,records,db,overwrite=False)
            return
        args.hf,args.out,args.imatrix = str(args.source),str(args.out),None
        writer=native_writer(q) if args.component=='main' and args.expert_format=='native-fp4' else q.write_gguf
        writer(args,plan,records,db)
    finally:db.close()


def guarded_conversion(args):
    import psutil
    from slstack.guard import supervise
    if not args.host_budget_gib or not math.isfinite(args.host_budget_gib) or args.host_budget_gib<=0:
        raise ValueError('--execute requires a finite positive --host-budget-gib')
    output = args.out.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    runtime = (getattr(args,'runtime_root',None) or output.parent).resolve()
    runtime.mkdir(parents=True,exist_ok=True)
    command = [sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
    # A plan requested by the caller was already published by this parent.
    if '--plan' in command:
        at=command.index('--plan');del command[at:at+2]
    receipt = runtime/(output.name+'.memory.json')
    env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TEMP=str(runtime),TMP=str(runtime),TMPDIR=str(runtime))
    env['SLSTACK_DSV41_CONVERSION_PARENT']=str(os.getpid())
    return supervise(dict(command=command,cwd=str(ROOT),env=env,host_budget_gib=args.host_budget_gib,
        job_commit_gib=args.host_budget_gib,min_available_gib=args.runtime_floor_gib,timeout_s=21600,
        owner_pid=os.getpid(),owner_created=psutil.Process().create_time(),runtime_root=str(runtime),
        stop=str(runtime/(output.name+'.stop')),log=str(runtime/(output.name+'.conversion.log')),receipt=str(receipt)))


def main():
    if '--requant-gguf' in sys.argv:
        from dsv41_requant import main as requant_main
        return requant_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Existing raw checkpoint directory; never downloaded')
    parser.add_argument('--ds4-tools',type=Path,required=True,help='Pinned gguf-tools source directory')
    parser.add_argument('--fetch-tools',action='store_true',help='Fetch only pinned converter source files, never models')
    parser.add_argument('--component',choices=('main','dspark','vision'),default='main')
    parser.add_argument('--expert-format',choices=('q4','native-fp4'),default='q4',
                        help='Main routed experts only; dense remains Q8_0 (no supported native FP8 type)')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--execute',action='store_true')
    parser.add_argument('--reference-gguf',type=Path,help='Compare with a local GGUF header; never opens its payload')
    parser.add_argument('--plan',type=Path,help='Write the predicted tensor inventory as JSON')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--source-revision',default='1bae0e1088993b9b82de08f59de5be5a34e42170')
    parser.add_argument('--quants-library',help='Built antirez/ds4 libds4quants shared library')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--host-budget-gib',type=float)
    parser.add_argument('--runtime-root',type=Path,help='Guard logs, receipts and temporary files; defaults to the output directory')
    parser.add_argument('--runtime-floor-gib',type=float,default=4)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.expert_format=='native-fp4' and args.component!='main':
        parser.error('--expert-format native-fp4 applies only to the main model; preserve the existing sidecars')
    if not 1<=args.threads<=32 or args.runtime_floor_gib<4:parser.error('threads must be 1..32 and RAM floor at least 4 GiB')
    if not re.fullmatch('[0-9a-f]{40}',args.source_revision):parser.error('source revision must be a full commit SHA')
    if args.execute and (not args.out or (args.component!='vision' and not args.quants_library)):
        parser.error('--execute needs --out and (main/DSpark) --quants-library')
    modules = load_tools(args.ds4_tools,args.fetch_tools)
    config,index = read_json(args.source/'config.json'),read_json(args.source/'model.safetensors.index.json')
    if config.get('model_type')!='deepseek_v41':raise ValueError('Expected DeepSeek V4.1 source config')
    reference = read_json(RECIPE/'reference-layout.json')['components'][args.component]['tensors']
    if args.expert_format=='native-fp4':reference=native_layout(reference)
    plan = make_plan(args.component,modules,config,index,reference)
    inventory = planned_map(plan)
    if args.reference_gguf and inventory != tensor_map(read_header(args.reference_gguf)):
        raise ValueError('Dry-run names/types/shapes differ from supplied GGUF header')
    if args.plan:
        with args.plan.open('x',encoding='utf-8') as file:json.dump(inventory,file,indent=2);file.write('\n')
    print(f'{args.component}: {len(plan)} tensor names/types/shapes match the reference layout; payload not read',flush=True)
    if args.execute:
        if not args.worker:return guarded_conversion(args)
        import psutil
        if os.environ.get('SLSTACK_DSV41_CONVERSION_PARENT')!=str(psutil.Process().ppid()):
            raise ValueError('--worker is internal to the memory supervisor')
        convert(args,modules,config,index,reference)
        if inventory != tensor_map(read_header(args.out)):raise ValueError('Converted GGUF header differs from plan')
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,KeyError) as error:sys.exit('convert-dsv41: '+str(error))
