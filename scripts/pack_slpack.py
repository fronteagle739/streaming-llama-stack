"""Reproduce QFN native NVFP4 from a pinned checkpoint and explicit auxiliary pack.

Routed NVFP4 values/scales are losslessly relaid out; dense weights use the
declared source Q6_K/Q8_0 conversion. Existing MTP, vision, profiles, metadata
and sampler defaults are retained from --sections-from. No model is downloaded
unless the separate download action is requested.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.slpack import ModelWriter, ModelFile
from scripts.dsv41_fp4_layout import NVFP4, pack_nvfp4_blob

BLOB = 2764816


def native_descriptor(layout):
    return dict(version=1, layout=NVFP4, lossless=True, expert_stride=BLOB,
                role_offsets=[0, 921600, 1843200], tail_offset=2764800, tail_bytes=16,
                weight_scale='UE4M3 per16; unchanged FP32 globals',
                values='adjacent low/high E2M1 nibbles [N,K/2]', scales='UE4M3 [K/64,N,4]',
                kernel='DSV grouped native NVFP4 derivative; all arithmetic modes remain explicit',
                source_experts_layout=layout)


class NativeWriter(ModelWriter):
    """One source expert at a time, with a byte-exact inverse before emission."""
    def __init__(self, path):
        super().__init__(path)
        self.source_layout = None
        self.expert_count = 0

    def add_chunks(self, name, chunks):
        if name != 'experts':
            return super().add_chunks(name, chunks)
        def converted():
            pending = bytearray()
            for chunk in chunks:
                pending.extend(chunk)
                while len(pending) >= BLOB:
                    raw = bytes(pending[:BLOB])
                    del pending[:BLOB]
                    packed = pack_nvfp4_blob(raw)
                    if pack_nvfp4_blob(packed, inverse=True) != raw:
                        raise ValueError('Native expert inverse differs at ' + str(self.expert_count))
                    self.expert_count += 1
                    yield packed
            if pending:
                raise ValueError('Incomplete NVFP4 expert blob')
        return super().add_chunks(name, converted())

    def add_bytes(self, name, data):
        if name == 'experts.layout':
            layout = data.decode('utf8')
            if not layout.splitlines()[0].startswith('# experts v6:'):
                raise ValueError('Expected original v6 native NVFP4 expert descriptor')
            rows = [list(map(int, row.split())) for row in layout.splitlines()[1:] if row.strip()]
            match = re.search(r'\(n_expert (\d+)', layout.splitlines()[0])
            if match is None or not rows:
                raise ValueError('Missing expert-count descriptor')
            experts = int(match.group(1))
            if experts <= 0 or rows != [[layer, 40, 40, layer * experts * BLOB, BLOB] for layer in range(len(rows))]:
                raise ValueError('Expert descriptors disagree with native projection geometry')
            if self.expert_count != len(rows) * experts:
                raise ValueError('Expert payload count differs from layout')
            self.source_layout = layout
            data = ('# experts v7 ' + NVFP4 + '\n' + '\n'.join(layout.splitlines()[1:]) + '\n').encode()
        elif name == 'meta':
            if self.source_layout is None:
                raise ValueError('Missing source expert layout')
            meta = json.loads(data)
            if 'qfn_native_nvfp4' in meta:
                raise ValueError('Metadata already describes a native relayout')
            meta['qfn_native_nvfp4'] = native_descriptor(self.source_layout)
            data = (json.dumps(meta, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()
        return super().add_bytes(name, data)


def check_source(source, recipe):
    """Authenticate every source file, retaining the complete pinned snapshot."""
    from scripts.slpack import file_chunks
    rows = recipe['source']['files']
    if not rows:
        raise ValueError('Recipe has no pinned source file identities')
    for row in rows:
        path = source / row['name']
        if path.stat().st_size != row['bytes']:
            raise ValueError('Source length differs: ' + row['name'])
        digest = hashlib.new(row['hash_algorithm'])
        if row['hash_algorithm'] == 'sha1':
            digest.update(('blob ' + str(row['bytes']) + '\0').encode())
        for chunk in file_chunks(path):
            digest.update(chunk)
        if digest.hexdigest() != row['hash']:
            raise ValueError('Pinned source hash differs: ' + row['name'])
        print('source verified: ' + row['name'], flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sections-from', type=Path, required=True,
                        help='Compatible QFN auxiliary pack providing MTP/vision/profile/sampler defaults')
    parser.add_argument('--allow-legacy-base', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, default=ROOT/'recipes/qfn-nvfp4.json')
    parser.add_argument('--name', default='Qwen3.8-Flash-Next NVIDIA-NVFP4 with existing QFN MTP and vision')
    parser.add_argument('--dense-q8-0', action='store_true', help='Use Q8_0 dense projections instead of Q6_K')
    parser.add_argument('--mtp-dense', type=Path, help='Explicit same-size tuned MTP dense replacement')
    parser.add_argument('--scratch', type=Path, required=True, help='Tokenizer temp output; retained after the run')
    parser.add_argument('--host-budget-gib', type=float, required=True)
    parser.add_argument('--runtime-floor-gib', type=float, default=4)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.out.suffix != '.slpack':
        parser.error('Output must use .slpack')
    if args.host_budget_gib <= 0 or args.runtime_floor_gib < 4:
        parser.error('Positive explicit host budget and at least4GiB free physical floor required')
    args.out = args.out.resolve()
    args.scratch = args.scratch.resolve()
    args.scratch.mkdir(parents=True, exist_ok=True)
    if not args.worker:
        from slstack.guard import supervise
        import psutil
        environment = dict(os.environ, TEMP=str(args.scratch), TMP=str(args.scratch),
                           TMPDIR=str(args.scratch), PYTHONDONTWRITEBYTECODE='1',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           SLSTACK_SLPACK_PARENT=str(os.getpid()))
        return supervise(dict(command=[sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
            cwd=str(ROOT), env=environment, host_budget_gib=args.host_budget_gib,
            job_commit_gib=args.host_budget_gib, min_available_gib=args.runtime_floor_gib,
            timeout_s=21600, owner_pid=os.getpid(), owner_created=psutil.Process().create_time(),
            runtime_root=str(args.scratch), stop=str(args.scratch/'slpack.stop'),
            log=str(args.scratch/'slpack.log'), receipt=str(args.scratch/'slpack.memory.json')))
    import psutil
    if os.environ.get('SLSTACK_SLPACK_PARENT') != str(psutil.Process().ppid()):
        raise ValueError('Internal worker must be launched by its memory supervisor')
    recipe = json.loads(args.recipe.read_text(encoding='utf8'))
    if recipe['model'] != 'qfn' or recipe['source']['revision'] != 'fc694b54fb0174e0913e6adf86691ef85a4ead47':
        raise ValueError('Wrong model or source revision')
    check_source(args.source, recipe)
    # Only explicitly supplied auxiliary bytes are reused. Unknown sections/defaults fail in the importer.
    ModelFile(args.sections_from, allow_legacy=args.allow_legacy_base)
    package = Path(__file__).with_name('slpack_qfn_source')
    sys.path.insert(0, str(package))
    import convert
    tokenizer_dir = args.scratch/'tokenizer-export'
    tokenizer_dir.mkdir(exist_ok=True)
    if any(tokenizer_dir.iterdir()):
        raise ValueError('Tokenizer export directory must be empty; existing files are retained')
    @contextlib.contextmanager
    def tokenizer_scratch():
        yield str(tokenizer_dir)
    args.tokenizer_scratch = tokenizer_scratch
    args.checkpoint, args.workers, args.force = args.source, 1, False
    convert.from_checkpoint(args)
    report = ModelFile(args.out).verify_all()
    report['routed_experts_inverse_exact'] = True
    report['whole_model_lossless'] = False
    report['auxiliary_sections_dependency_explicit'] = True
    with Path(str(args.out)+'.verify.json').open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
