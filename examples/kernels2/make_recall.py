"""Create a reproducible retrieval prompt using the model's own chat tokens."""
import argparse,hashlib,json,struct,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from slstack.ds4_render import Tokenizer,gguf_tokenizer

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=100000)
    a=p.parse_args()
    if not 384<=a.tokens<=200000:p.error('Fixture length must be384..200000 tokens')
    tok=Tokenizer(*gguf_tokenizer(a.model));key='QUARTZ-7429-LANTERN'
    prefix=[tok.bos,tok.user]+tok.text('TEST archive. The archive retrieval key is '+key+'. Remember this exact key for the final question. The following weather observations are background notes.\n')
    tail=tok.text('\nEnd of notes. What is the archive retrieval key? Return only the exact key, without extra words.')+[tok.assistant,tok.think_end]
    pattern=tok.text(' The weather station records calm winds and clear skies.\n')
    n=a.tokens-len(prefix)-len(tail)
    tokens=prefix+(pattern*(n//len(pattern)+1))[:n]+tail
    binary=struct.pack('<'+'I'*len(tokens),*tokens);a.output.write_bytes(binary)
    receipt=dict(tokens=len(tokens),key=key,eos=tok.eos,sha256=hashlib.sha256(binary).hexdigest())
    a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
