import os
from pathlib import Path
import uuid
from unittest.mock import patch

from slstack.single_runtime import PRODUCER_FILES, serving_config, stack_config


def test_live_stack_preserves_parent_but_packaged_config_does_not():
    root = Path(os.environ['STUDIO_RUNTIME_ROOT']) / ('single-parent-' + uuid.uuid4().hex)
    root.mkdir()
    producer = root / 'producer.exe'
    for file in (producer, *(root / name for name in PRODUCER_FILES)):
        file.write_bytes(b'CPU configuration fixture; never executed')
    cfg = dict(llama_producer=str(producer), runtime_root=str(root), model='unused',
               ds4_url='http://localhost:8181', studio_root=str(root), host_budget_gib=9.5, context=262144,
               _gpu_parent_lease=str(root/'ancestor.json'))
    with patch('slstack.single_runtime.subprocess.check_output', return_value='fixture'):
        result = stack_config(cfg)
    assert result['_gpu_parent_lease'] == cfg['_gpu_parent_lease']
    packaged = serving_config(cfg, runtime_root=root, studio_root=root,
                              llama_producer=producer, library_directories=[], revision='fixture')
    assert '_gpu_parent_lease' not in packaged
