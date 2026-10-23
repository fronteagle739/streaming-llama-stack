"""Stage an immutable normal-Studio runtime after hardware acceptance.

No installation or store mutation occurs here. The stage refers to its final
installation path so retiring the qualification scratch cannot strand a preset.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Optional

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
from slstack.combo_measure import require_parity
from slstack.combo_crossover import qualified_threshold


def sha(path):
    with Path(path).open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def normal_argv(source,preset):
    """Execute the actual Studio pure builders, without importing its daemon."""
    tree=ast.parse(source.read_text(encoding='utf-8'))
    names={'cache_affinity_args','ds4_argv'}
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if len(selected)!=2:raise ValueError('Current normal native argv builders are unavailable')
    namespace=dict(Optional=Optional,re=re,Path=Path,preset_binary=lambda _:Path(preset['rtx']['ds4_binary']))
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),namespace)
    return namespace['ds4_argv'](preset,port=18748)[1:]


def acceptance(cfg,measurements,baseline,alternating):
    require_parity(cfg);qualified_threshold(cfg)
    rows=[read(path) for path in measurements]
    if len(rows)!=3:raise ValueError('Three measured quality settings are mandatory')
    if {row.get('requested_quality_speed') for row in rows}!={0,.35,1.36125}:
        raise ValueError('Max quality, Balanced and Max speed measurements are mandatory')
    for row in rows:
        if (row.get('status')!='PASS' or row.get('identity')!=cfg['combo_identity'] or
            any(row.get(k)!=cfg[k] for k in ('native_sha256','producer_sha256','provider_schema','native_prefill_chunk')) or
            row.get('native_split_reads')!=8):
            raise ValueError('A50K measurement is incomplete or from another model profile')
        turns=row.get('turns',[])
        if len(turns)<2 or abs(turns[0]['metrics']['prompt_tokens']-50000)>8:
            raise ValueError('A real50K prompt and follow-up are mandatory')
        if turns[0]['metrics']['route']!='ultra' or turns[1]['metrics']['reused_prefix_tokens']<=0:
            raise ValueError('The50K measurement did not transfer/reuse its actual prefix')
        if not all(str(count) in turns[0]['engine_milestones'] for count in (100,500,1000)):
            raise ValueError('First100/500 and warm500-token decode measurements are incomplete')
    native=read(baseline)
    if native.get('status') not in ('PASS','CENSORED'):
        raise ValueError('Native-only TTFT comparison or explicit bounded censoring receipt is mandatory')
    policies=set()
    for path in alternating:
        row=read(path);policies.add(row.get('reasoning_history'))
        if (row.get('status')!='PASS' or len(row.get('turns',[]))!=6 or row.get('identity')!=cfg['combo_identity'] or
            any(row.get(k)!=cfg[k] for k in ('native_sha256','producer_sha256','provider_schema'))):
            raise ValueError('Both six-turn alternating conversations must pass')
    if policies!={True,False}:raise ValueError('Reasoning history ON and OFF are mandatory')


def stage(args):
    cfg=read(args.config)
    acceptance(cfg,args.measurements,args.native_baseline,args.alternating)
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=SOURCE,text=True).strip()
    selected_branch = __import__("os").environ.get("STACK_BUILD_BRANCH", "agent/stackclean-sol-20261025")
    if branch != selected_branch or not branch.startswith('agent/') or re.search(r'(^|[/._-])(main|master)($|[/._-])',branch,re.I):
        raise ValueError('Owned task source branch required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip():
        raise ValueError('Commit and publish the final source before staging an immutable runtime')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    target=Path(args.stage).resolve();installed=Path(args.install_root).resolve()
    if target.exists() or installed.exists():raise FileExistsError('Runtime stage and immutable destination must be new paths')
    root=Path(cfg['runtime_root']).resolve()
    if root not in target.parents:raise ValueError('Stage must remain inside the single task scratch root')
    if sha(cfg['ds4_command'][0])!=cfg['native_sha256'] or sha(cfg['llama_producer'])!=cfg['producer_sha256']:
        raise ValueError('Qualified engine binary changed')
    target.mkdir()
    def copy(source,relative):
        destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        if sha(source)!=sha(destination):raise ValueError('Runtime copy hash differs: '+relative)
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=SOURCE).decode().split('\0')
    for relative in tracked:
        if (relative.startswith(('slstack/','vendor/studio/')) and Path(relative).suffix in ('.py','.json')) or relative=='envvars.json':
            copy(SOURCE/relative,relative)
    copy(args.launcher,'ds4-server.exe');copy(cfg['ds4_command'][0],'native/ds4-server.exe')
    copy(cfg['llama_producer'],'combo-prefill.exe')
    native=Path(cfg['ds4_command'][0]).parent
    for name in ('dstorage.dll','dstoragecore.dll'):
        copy(native/name,name);copy(native/name,'native/'+name)
    proof_root=Path(cfg.get('combo_qualification_root',cfg['runtime_root']))
    for name in ('native-parity.json','native-parity-binding.json','ultra-parity.json'):
        copy(proof_root/name,'qualification/'+name)
    copy(cfg['combo_crossover_receipt'],'qualification/crossover.json')
    copy(cfg['model_receipt'],'qualification/original-model.json')
    copy(cfg['cache_affinity_spread_file'],'qualification/affinity-spreads.txt')
    for index,path in enumerate([*args.measurements,args.native_baseline,*args.alternating]):
        copy(path,f'qualification/measurement-{index+1}.json')
    result=deepcopy(cfg);result.update(host_budget_gib=9.25,combo_qualification_root=str(installed/'qualification'),
        combo_crossover_receipt=str(installed/'qualification/crossover.json'),
        model_receipt=str(installed/'qualification/original-model.json'),
        cache_affinity_spread_file=str(installed/'qualification/affinity-spreads.txt'),
        llama_producer=str(installed/'combo-prefill.exe'),source_revision=revision,studio_root=str(installed/'vendor/studio'))
    result['ds4_command'][0]=str(installed/'native/ds4-server.exe')
    result['producer_library_directories']=[str(installed/'native'),*[p for p in cfg['producer_library_directories'] if Path(p).resolve()!=native.resolve()]]
    preset=dict(name='DSV4.1 1017 (RTX: ULTRA prefill \u2192 ds4-win decode)',mode='local',mac={},include_previous_reasoning=True,
        rtx=dict(backend='ds4',model='deepseek-reasoner',model_path=cfg['model'],ctx=cfg['context'],
            ds4_binary=str(installed/'ds4-server.exe'),ds4_chdir=str(installed),ds4_ssd_streaming='1',ds4_cache_experts='24GB',
            prefill_chunk=cfg['native_prefill_chunk'],ds4_cache_affinity_bias=0,
            ds4_cache_affinity_spread_file=result['cache_affinity_spread_file'],ds4_extra_flags='--cuda',
            ds4_host_memory_budget_mb=9472,ds4_host_memory_reserve_mb=2048))
    result['studio_expected_args']=normal_argv(Path(args.studio_native),preset)
    (target/'combo.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (target/'python-path.txt').write_text(str(Path(sys.executable).resolve())+'\n',encoding='utf-8')
    (target/'preset.json').write_text(json.dumps(preset,indent=2,ensure_ascii=False),encoding='utf-8')
    manifest=dict(source_revision=revision,source_branch=branch,installed_root=str(installed),
        native_sha256=cfg['native_sha256'],producer_sha256=cfg['producer_sha256'],
        studio_native_builder_sha256=sha(args.studio_native),files={})
    for file in target.rglob('*'):
        if file.is_file():manifest['files'][file.relative_to(target).as_posix()]=dict(sha256=sha(file),bytes=file.stat().st_size)
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(stage=str(target),files=len(manifest['files']),bytes=sum(v['bytes'] for v in manifest['files'].values()),revision=revision)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('config','stage','install-root','launcher','studio-native','native-baseline'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--measurements',nargs=3,required=True);parser.add_argument('--alternating',nargs=2,required=True)
    stage(parser.parse_args())
