"""Real ABI and Metal admission check; an import alone is insufficient."""
import importlib.util
import json
from pathlib import Path
import sys

import mlx.core as mx

SYMBOLS = ('dsa_indexer_scores','dsa_topk_indices','glm_dsa_sparse_mla_attention',
           'glm_dsa_exact_block_attention','glm_dsa_q8_vup_flat','glm_moe_weighted_sum')


def check(ext):
    missing=[name for name in SYMBOLS if not hasattr(ext,name)]
    if missing:
        raise RuntimeError(f'Native extension missing symbols: {missing}')
    # Pass actual mx.array objects to catch incompatible nb_internals domains.
    q=mx.full((1,32,64,128),1/128,mx.bfloat16)
    k=mx.ones((1,1,4096,128),mx.bfloat16)
    w=mx.full((1,64,32),1/32,mx.bfloat16)
    scores=ext.dsa_indexer_scores(q,k,w,causal=False)
    values=mx.concatenate([mx.zeros((1,1,1,2048),mx.bfloat16),
                           mx.ones((1,1,1,2048),mx.bfloat16)],axis=-1)
    selected=ext.dsa_topk_indices(values,2048)
    mx.eval(scores,selected)
    if not bool(mx.all(scores==1)) or not bool(mx.all(mx.sort(selected)==mx.arange(2048,4096))):
        raise RuntimeError('Native ABI/Metal smoke produced incorrect scores or top-k')
    return dict(symbols=list(SYMBOLS),array_abi=True,metal_evaluated=True)


if __name__=='__main__':
    path=next(Path(sys.argv[1]).glob('_ext*.so'))
    spec=importlib.util.spec_from_file_location('_ext',path)
    ext=importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
    print(json.dumps(check(ext)))
