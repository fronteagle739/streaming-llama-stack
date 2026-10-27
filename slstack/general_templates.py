"""Preserve the model template while accommodating minja's numeric indexing."""
from __future__ import annotations
import hashlib
from pathlib import Path
import re

DOT_INDEX = re.compile(r'(?<=[\w\)\]])\.(\d+)(?=\.[A-Za-z_])')


def minja_template(text):
    return DOT_INDEX.sub(lambda match:'['+match.group(1)+']',text)


def template_file(runtime, *, inline=None, filename=None, metadata=None):
    explicit=inline is not None or filename is not None
    text=Path(filename).read_text(encoding='utf-8') if filename else inline
    if text is None:text=(metadata or {}).get('tokenizer.chat_template') or ''
    if not isinstance(text,str):raise ValueError('The selected chat template must be text')
    changed=minja_template(text)
    if not explicit and changed==text:return None
    if not changed:return None
    if filename and changed==text:return str(Path(filename).resolve())
    dest=Path(runtime)/'templates'/(hashlib.sha256(changed.encode()).hexdigest()+'.jinja')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if dest.read_text(encoding='utf-8')!=changed:raise ValueError('Cached template content changed')
    else:
        with dest.open('x',encoding='utf-8',newline='') as stream:stream.write(changed)
    return str(dest)
