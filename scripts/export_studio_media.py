"""Copy only the Studio modules needed by the existing media adapter.

Author exports on RTX. Per-file source/export hashes record the exact input;
machine-specific absolute fallback paths are cleared, leaving explicit settings.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

MODULES = ('krea2_diffusion','krea_hf_stream','boogu_krea','krea2_lokr','krea_tiled',
           'moss_image','qwen_image21','qwen_worker_client','qwen_comfy_load','ram_preflight',
           'direct_vram','res4lyf_registry','qwen_direct_load','qwen_block_weights','qwen_lora',
           'h3_weight_io','qwen_attention','progress','cuda_device_identity','safety',
           'h3_live','h3_single_file','h3_image_mode','h3_model_profiles')
JOB_FUNCTIONS = ('_cosmos_windows_roots','_unbuffered_env','_popen_group_kwargs','kill_process_tree','managed_popen')
ROOT = Path(__file__).resolve().parents[1]


def digest(data):return hashlib.sha256(data).hexdigest()


class Portable(ast.NodeTransformer):
    def visit_Expr(self,node):
        # Do not publish machine-specific paths embedded in source docstrings.
        if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):return None
        return self.generic_visit(node)
    def visit_Constant(self,node):
        if isinstance(node.value,str) and (re.match(r'^[A-Za-z]:[\\/]',node.value) or node.value.startswith(('/Users/','/Volumes/'))):
            return ast.copy_location(ast.Constant(''),node)
        return node


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--studio',type=Path,required=True);args=parser.parse_args()
    source=args.studio/'backend/studio';destination=ROOT/'vendor/studio/studio'
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.studio,text=True).strip();records=[]
    for name in (*MODULES,'jobs'):
        target=destination/(name+'.py')
        if target.exists():raise ValueError('Preserve and reconcile existing module before export: '+name)
        raw=(source/(name+'.py')).read_bytes();tree=ast.parse(raw.decode('utf-8-sig'))
        if name=='jobs':
            selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in JOB_FUNCTIONS]
            imports=ast.parse('from __future__ import annotations\nimport os, sys, signal, subprocess\nfrom pathlib import Path\nfrom typing import Optional\nfrom . import worker_registry\n').body
            tree=ast.Module(body=imports+selected,type_ignores=[])
        tree=Portable().visit(tree);ast.fix_missing_locations(tree)
        output=('# Exported from Studio '+revision+'; machine-specific fallback paths cleared.\n'+ast.unparse(tree)+'\n').encode()
        compile(output,str(target),'exec');target.write_bytes(output)
        records.append(dict(path=target.relative_to(ROOT).as_posix(),source_path='backend/studio/'+name+'.py',
                            source_sha256=digest(raw),export_sha256=digest(output),selected=list(JOB_FUNCTIONS) if name=='jobs' else None))
    provenance=ROOT/'vendor/studio/PROVENANCE.json';row=json.loads(provenance.read_text())
    for relative in ('vendored/krea2/transformer_krea2.py','vendored/krea2/_shim.txt'):
        target=destination/relative
        if target.exists():raise ValueError('Preserve existing vendored source before export: '+relative)
        data=(source/relative).read_bytes();target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        records.append(dict(path=target.relative_to(ROOT).as_posix(),source_path='backend/studio/'+relative,
                            source_sha256=digest(data),export_sha256=digest(data),selected=None))
    row['media_support_export']=dict(source_commit=revision,files=records,transformation='AST source export; absolute machine fallback paths cleared; jobs exports only owned-worker launch/cleanup helpers')
    provenance.write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(modules=len(records),bytes=sum((ROOT/r['path']).stat().st_size for r in records))))


if __name__=='__main__':main()
