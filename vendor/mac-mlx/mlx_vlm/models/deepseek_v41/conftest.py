"""Collect standalone CPU tests without importing the MLX parent package."""

import importlib.util
import sys

import pytest


class StandaloneModule(pytest.Module):
    def _getobj(self):
        spec = importlib.util.spec_from_file_location("standalone_v41_" + self.path.stem, self.path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


def pytest_pycollect_makemodule(module_path, parent):
    if module_path.name in ("test_checkpoint_layout_v41.py", "test_reference_cpu_dsv41.py",
                            "test_engram_hash_parity.py", "test_vision_dsv41.py",
                            "test_dspark_dsv41.py", "test_encoding_dsv41.py",
                            "test_language_dsv41_mac.py", "test_kv_selection_width.py",
                            "test_prefill_linear.py", "test_prefill_ledger.py",
                            "test_megakernel_prefill.py", "test_ffn_down_bf16_cpu.py",
                            "test_ffn_down_bf16.py", "test_ffn_expert_pad_cpu.py",
                            "test_ffn_expert_pad.py"):
        return StandaloneModule.from_parent(parent.session, path=module_path)
