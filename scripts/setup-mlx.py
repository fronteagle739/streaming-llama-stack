"""Use the core runtime: bin/slstack guarded -- python scripts/setup-mlx.py."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slstack.config import load
from slstack.mlx_setup import setup

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--no-native-glm',action='store_true',help='Explicit pure-MLX baseline; does not claim accelerated GLM performance')
    args=parser.parse_args()
    result=setup(load(args.config),check=args.check,native_glm=not args.no_native_glm)
    print(json.dumps(result,indent=2));return 0 if result['ready'] else 1

if __name__=='__main__':raise SystemExit(main())
