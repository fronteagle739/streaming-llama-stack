# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/gemma4_tokenizer_from_gguf.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import sys

def build_gemma4_tokenizer_from_gguf(gguf_path: str):
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import Unigram
    from tokenizers.normalizers import Replace
    from tokenizers.pre_tokenizers import Metaspace as MetaspacePre
    from tokenizers.decoders import Metaspace as MetaspaceDec
    from transformers import PreTrainedTokenizerFast
    sys.path.insert(0, '')
    from cosmos_windows.gguf_qat.reader import GGUFTensorReader
    r = GGUFTensorReader(gguf_path, device='cpu')
    tokens = list(r.meta('tokenizer.ggml.tokens'))
    scores = r.meta('tokenizer.ggml.scores')
    ttypes = r.meta('tokenizer.ggml.token_type')
    unk_id = int(r.meta('tokenizer.ggml.unknown_token_id', 3))
    bos_id = int(r.meta('tokenizer.ggml.bos_token_id', 2))
    eos_id = int(r.meta('tokenizer.ggml.eos_token_id', 1))
    pad_id = int(r.meta('tokenizer.ggml.padding_token_id', 0))
    add_bos = bool(r.meta('tokenizer.ggml.add_bos_token', True))
    chat_template = r.meta('tokenizer.chat_template', None)
    r.close()
    scores = [float(s) for s in scores] if scores is not None else [0.0] * len(tokens)
    vocab = [(tok, scores[i]) for i, tok in enumerate(tokens)]
    model = Unigram(vocab, unk_id=unk_id, byte_fallback=True)
    tk = Tokenizer(model)
    tk.normalizer = Replace(' ', '▁')
    tk.pre_tokenizer = MetaspacePre(replacement='▁', prepend_scheme='first')
    tk.decoder = decoders.Sequence([MetaspaceDec(replacement='▁', prepend_scheme='first'), decoders.ByteFallback()])

    def tid(name):
        try:
            return tokens.index(name)
        except ValueError:
            return None
    special_map = {'bos_token': tokens[bos_id], 'eos_token': tokens[eos_id], 'pad_token': tokens[pad_id], 'unk_token': tokens[unk_id]}
    sot = tid('<start_of_turn>')
    eot = tid('<end_of_turn>')
    additional = [t for t in ('<start_of_turn>', '<end_of_turn>') if tid(t) is not None]
    hf = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token=special_map['bos_token'], eos_token=special_map['eos_token'], pad_token=special_map['pad_token'], unk_token=special_map['unk_token'], additional_special_tokens=additional)
    hf.add_bos_token = add_bos
    if chat_template:
        hf.chat_template = chat_template
    return hf
if __name__ == '__main__':
    sys.path.insert(0, '')
    from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb
    start_rss_guard(limit_gb=6.0, label='gemma4_tok')
    gguf = sys.argv[1] if len(sys.argv) > 1 else ''
    tok = build_gemma4_tokenizer_from_gguf(gguf)
    print('type:', type(tok).__name__, 'vocab', tok.vocab_size, 'bos', tok.bos_token_id, 'eos', tok.eos_token_id, 'pad', tok.pad_token_id)
    for s in ['who are you?', 'MA name is JEFF!!', 'The capital of France is Paris.', '2+2 = 4.']:
        ids = tok(s, add_special_tokens=False)['input_ids']
        back = tok.decode(ids)
        print(f"  {s!r:40s} -> {ids[:14]}{('...' if len(ids) > 14 else '')} -> {back!r}  RT={('OK' if back == s else 'DIFF')}")
    try:
        msgs = [{'role': 'user', 'content': 'who are you?'}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
        print('chat-template len', len(ids), 'decoded:', repr(tok.decode(ids))[:200])
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('chat template FAILED:', repr(e)[:200])
    print(f'peak RSS {peak_rss_gb():.3f} GB')
