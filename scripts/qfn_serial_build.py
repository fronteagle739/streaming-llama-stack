"""Serialize QFN build controllers; their existing memory/priority guards still apply."""
import argparse
import datetime
import msvcrt
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True)
parser.add_argument('--lock', required=True, type=Path)
parser.add_argument('--progress-log', required=True, type=Path)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
if not command:
    parser.error('a guarded build command is required')
if not args.task.startswith('QFN') or not args.task.replace('_', '').isalnum():
    parser.error('expected QFN task name')
lock = args.lock
lock.parent.mkdir(parents=True, exist_ok=True)

def progress(message):
    text = f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} {message}'
    print(text, flush=True)
    with args.progress_log.open('a', encoding='utf-8') as f:
        f.write(text + '\n')

with lock.open('a+b') as stream:
    if stream.seek(0, os.SEEK_END) == 0:
        stream.write(b'0')
        stream.flush()
    started = last = time.monotonic()
    while True:
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            break
        except OSError:
            if time.monotonic() - last >= 600 or last == started:
                progress('doing=waiting for serial RTX build admission; latest=another QFN build holds lock; next=run existing guarded build when it exits')
                last = time.monotonic()
            time.sleep(5)
    try:
        progress('doing=serial RTX build admitted; latest=exclusive build lock held; next=existing memory/priority guarded child')
        rc = subprocess.call(command)
        progress(f'doing=serial RTX build finished; latest=child exit{rc}; next=inspect artifact and release lock')
    finally:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
raise SystemExit(rc)
