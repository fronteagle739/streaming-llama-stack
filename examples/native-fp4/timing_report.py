"""Read the engine's sequential prompt clock separately from launch-to-ready TTFT."""
import hashlib
import json
import math
from pathlib import Path


def from_receipt(path, measurement):
    data = Path(path).read_bytes()
    if len(data) > 2 << 20:
        raise ValueError('Unexpected native timing receipt size')
    receipt = json.loads(data)
    if receipt.get('status') != 'complete':
        raise ValueError('Native timing receipt is incomplete')
    if receipt['tokens']['count'] != measurement['prompt_tokens'] or receipt['tokens']['sha256'] != measurement['input_sha256']:
        raise ValueError('Native timing receipt has different prompt tokens')
    if receipt['model']['sha256'] != measurement['model_sha256']:
        raise ValueError('Native timing receipt has a different model')
    if any(receipt[key] != measurement['producer_sha256'] for key in (
            'producer_binary_sha256', 'caller_expected_build_sha256')):
        raise ValueError('Native timing receipt has a different producer')
    phase = receipt['phase_ms']
    names = ('embedding_40_layers_head', 'append_owner_admission', 'chunk_token_preparation')
    if any(not math.isfinite(phase[key]) or phase[key] < 0 for key in names):
        raise ValueError('Invalid native prompt timing')
    chunks = receipt['stateful_chunks']['chunks']
    position, chunk_ms = 0, 0.0
    for chunk in chunks:
        value = chunk['prompt_processing_ms']
        if (chunk['first'] != position or chunk['tokens'] <= 0 or not math.isfinite(value) or value <= 0):
            raise ValueError('Invalid native prompt chunk timing')
        position += chunk['tokens']
        if chunk['completed_context_tokens'] != position:
            raise ValueError('Native prompt chunks are not a fresh contiguous prompt')
        chunk_ms += value
    if position != measurement['prompt_tokens']:
        raise ValueError('Native prompt chunks omit tokens')
    expected = chunk_ms + phase['append_owner_admission'] + phase['chunk_token_preparation']
    if not math.isclose(expected, phase['embedding_40_layers_head'], rel_tol=1e-10, abs_tol=1e-4):
        raise ValueError('Native prompt clocks do not reconcile')
    prompt = phase['embedding_40_layers_head'] / 1000
    ttft = measurement['prefill_seconds']
    if not math.isfinite(ttft) or prompt <= 0 or ttft < prompt:
        raise ValueError('TTFT is shorter than the native prompt interval')
    return dict(prompt_prefill_seconds=prompt, prompt_prefill_tok_s=position / prompt,
        ttft_seconds=ttft, ttft_tok_s=position / ttft,
        prefill_seconds_scope='legacy alias for launch-to-ready TTFT; startup included',
        prompt_clock_definition='Native sequential embedding_40_layers_head clock: all prompt sweeps, token preparation and between-sweep ownership admission through synchronized frontier logits. Initial model open/allocation, final bulk export/certification, recall and decode excluded. Early Engram prefetch can overlap initial open.',
        ttft_clock_definition='Host ComboProducer launch boundary through frontier READY: process startup, model open/allocation, prompt processing, final state certification and IPC included; external harness prechecks, recall and decode excluded.',
        setup_and_ready_overhead_seconds=ttft-prompt,
        overhead_definition='TTFT minus prompt interval; includes setup and post-head certification/IPC, not just process startup.',
        native_prompt_receipt_sha256=hashlib.sha256(data).hexdigest(),
        native_prompt_phase_ms={key: phase[key] for key in names})


def recover(measurement_path, trace_path):
    """Recover clocks from a completed run without changing its original files."""
    measurement = json.loads(Path(measurement_path).read_text())
    prefix = 'Pending receipt (retained on failure): '
    first = ''
    with Path(trace_path).open(encoding='utf-8', errors='replace') as stream:
        # A fold-plan diagnostic may precede the receipt on long prompts.
        for _ in range(64):
            line = stream.readline()
            if not line:
                break
            if line.startswith(prefix):
                first = line.rstrip()
                break
    if not first.startswith(prefix) or '.pending-' not in first:
        raise ValueError('Trace does not identify its native receipt')
    receipt = Path(first[len(prefix):].rsplit('.pending-', 1)[0])
    return from_receipt(receipt, measurement)
