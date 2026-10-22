"""Opt-in real exporter/journal check on a synthetic 4x4 GGUF, never an inference model.
Run beneath slstack guarded with a 2 GiB host budget in the Studio environment.
"""
import sys,json,hashlib,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"engines/streaming-llama/prefill_llama.cpp/gguf-py"))
import numpy as np,gguf
from slstack.config import load
from slstack.journal import Journal,sha256
from slstack.lora import merge_lora,unmerge_lora
parser=argparse.ArgumentParser();parser.add_argument("--exporter",required=True);parser.add_argument("--output-root",required=True);args=parser.parse_args();r=Path(args.output_root).resolve();r.mkdir(parents=True,exist_ok=False);base=r/"TEST-base.gguf";adapter=r/"TEST-adapter.gguf";out=r/"TEST-merged.gguf"
w=np.arange(16,dtype=np.float32).reshape(4,4)/8;a=np.array([[1,2,3,4],[2,1,0,-1]],dtype=np.float32)/8;b=np.array([[1,2],[0,1],[-1,2],[2,0]],dtype=np.float32)/4
x=gguf.GGUFWriter(str(base),"llama");x.add_name("TEST synthetic 4x4 tensor fixture, not an inference model");x.add_file_type(0);x.add_tensor("blk.0.attn_q.weight",w);x.write_header_to_file();x.write_kv_data_to_file();x.write_tensors_to_file();x.close()
x=gguf.GGUFWriter(str(adapter),"llama");x.add_string("general.type","adapter");x.add_string("adapter.type","lora");x.add_float32("adapter.lora.alpha",2.0);x.add_tensor("blk.0.attn_q.weight.lora_a",a);x.add_tensor("blk.0.attn_q.weight.lora_b",b);x.write_header_to_file();x.write_kv_data_to_file();x.write_tensors_to_file();x.close()
cfg=load(overrides={"runtime_root":str(r),"host_budget_gib":.5,"job_commit_gib":1});j=Journal(r);j.recover();before=sha256(base);result=merge_lora(cfg,str(base),[(str(adapter),.5)],str(out),str(Path(args.exporter).resolve()),test=True)
reader=gguf.GGUFReader(str(out));actual=reader.tensors[0].data.reshape(4,4);error=float(np.max(np.abs(actual-(w+.5*(b@a)))));assert error<1e-6,error;assert sha256(base)==before
identity=j.read()["entries"][-1]["id"];switched=unmerge_lora(cfg,identity);assert sha256(base)==before and out.is_file();assert j.read()["entries"][-1]["status"]=="detached"
proof={"fixture_only":True,"not_a_complete_model":True,"maximum_absolute_error":error,"base_sha256_before_and_after":before,"merge":result,"unmerge":switched,"journal":j.read(),"exporter_sha256":sha256(str(Path(args.exporter).resolve()))};(r/"proof.json").write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
