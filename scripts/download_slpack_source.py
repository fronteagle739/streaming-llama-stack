"""Download or verify every file of a recipe's immutable Hugging Face snapshot.

No files are removed or overwritten. Existing complete files are hash-verified;
interrupted downloads are retained as .partial and require explicit --resume.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.slpack import file_chunks


def verify(path, row):
    if path.stat().st_size != row['bytes']:
        raise ValueError('Source length differs: '+row['name'])
    digest=hashlib.new(row['hash_algorithm'])
    if row['hash_algorithm']=='sha1':
        digest.update(('blob '+str(row['bytes'])+'\0').encode())
    for chunk in file_chunks(path):
        digest.update(chunk)
    if digest.hexdigest()!=row['hash']:
        raise ValueError('Source hash differs: '+row['name'])


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    recipe=json.loads(args.recipe.read_text(encoding='utf8'))
    source=recipe['source']
    if not re.fullmatch(r'[\w.-]+/[\w.-]+',source['repo']) or not re.fullmatch('[0-9a-f]{40}',source['revision']):
        raise ValueError('Recipe requires repo and full immutable40hex revision')
    destination=args.out.resolve()
    destination.mkdir(parents=True,exist_ok=True)
    for row in source['files']:
        name=PurePosixPath(row['name'])
        if name.is_absolute() or '..' in name.parts or '\\' in row['name']:
            raise ValueError('Unsafe source filename')
        path=destination/str(name)
        if not path.exists():
            if args.verify_only:
                raise ValueError('Missing source file: '+row['name'])
            path.parent.mkdir(parents=True,exist_ok=True)
            partial=Path(str(path)+'.partial')
            if partial.exists() and not args.resume:
                raise ValueError('Partial retained; use explicit --resume: '+row['name'])
            completed=partial.stat().st_size if partial.exists() else 0
            if completed>row['bytes']:
                raise ValueError('Partial exceeds expected bytes')
            if completed<row['bytes']:
                url=f"https://huggingface.co/{source['repo']}/resolve/{source['revision']}/{name}"
                request=urllib.request.Request(url,headers={'Range':f'bytes={completed}-'} if completed else {})
                with urllib.request.urlopen(request,timeout=120) as response:
                    if completed and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {completed}-')):
                        raise ValueError('Server did not honor exact resume range')
                    with partial.open('ab' if completed else 'xb') as stream:
                        while chunk:=response.read(4<<20):
                            stream.write(chunk)
            verify(partial,row)
            partial.rename(path)
        verify(path,row)
        print('verified '+row['name']+' '+str(row['bytes'])+'bytes '+row['hash_algorithm']+':'+row['hash'],flush=True)
    print(json.dumps(dict(repo=source['repo'],revision=source['revision'],files=len(source['files']),
                          bytes=sum(row['bytes'] for row in source['files']),verified=True)))


if __name__=='__main__':
    main()
