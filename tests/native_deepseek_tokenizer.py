"""Standalone fixture test: actual C pre-tokenizer versus tokenizer.json.

Only emission is intercepted: the BPE implementation is unchanged by this port.
No engine or weight file is opened. Requires an existing tokenizer JSON and
DS4_TEST_SCRATCH; all generated files use stable names in that directory.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import unittest


def c_function(source, name):
    match = re.search(r'^static [^\n;]*\b' + re.escape(name) + r'\([^\n;]*\)\s*\{', source, re.M)
    assert match, name
    start = match.start()
    stripped = re.sub(r'''//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*' ''',
                      lambda m: ' ' * len(m.group()), source[start:], flags=re.X)
    brace = stripped.index('{')
    depth, pos = 1, brace + 1
    while depth:
        depth += (stripped[pos] == '{') - (stripped[pos] == '}')
        pos += 1
    return source[start:start+pos]


class DeepSeekPretokenizerTests(unittest.TestCase):
    def test_released_unicode_splits_and_complete_byte_coverage(self):
        from tokenizers import Regex
        from tokenizers.pre_tokenizers import Sequence, Split
        scratch = Path(os.environ['DS4_TEST_SCRATCH'])
        self.assertTrue(scratch.is_absolute() and scratch.is_dir())
        tokenizer = Path(os.environ['DS4_TEST_TOKENIZER_JSON'])
        config = json.loads(tokenizer.read_text(encoding='utf-8'))
        steps = config['pre_tokenizer']['pretokenizers']
        self.assertEqual([s['type'] for s in steps], ['Split'] * 3 + ['ByteLevel'])
        self.assertFalse(steps[3]['use_regex'])
        reference = Sequence([Split(Regex(s['pattern']['Regex']), 'isolated') for s in steps[:3]])
        root = Path(__file__).resolve().parents[1]
        engine = root / 'engines/ds4-mac'
        source = (engine / 'ds4.c').read_text(encoding='utf-8')
        fragment = source[source.index('#include "ds4_deepseek_unicode.inc"'):source.index('\nstatic int vocab_lookup', source.index('#include "ds4_deepseek_unicode.inc"'))]
        helpers = '\n'.join(c_function(source, name) for name in ['utf8_len_from_first_byte', 'ascii_alpha', 'ascii_newline', 'joyai_ascii_punct_symbol', 'utf8_is_cjk_hira_kata', 'utf8_peek_one'])
        header = r'''
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#define DS4_MODEL_FAMILY 0
#define DS4_MODEL_FAMILY_GLM_DSA 1
typedef struct { const char *ptr; uint64_t len; } ds4_str;
typedef struct { int unused; } ds4_vocab;
typedef struct { const char *base; uint64_t *pairs; size_t count, cap; } token_vec;
static void bpe_emit_piece(const ds4_vocab *v, ds4_str str, token_vec *out) {
    (void)v;assert(out->count<out->cap);
    out->pairs[2*out->count]=(uint64_t)(str.ptr-out->base);
    out->pairs[2*out->count+1]=str.len;out->count++;
}
static void bpe_tokenize_text_glm4(const ds4_vocab *v,const char *s,token_vec *out) {
    (void)v;(void)s;(void)out;assert(0);
}
'''
        footer = r'''
size_t actual_split(const char *text,uint64_t *pairs,size_t cap) {
    token_vec out={text,pairs,0,cap};ds4_vocab v={0};
    bpe_tokenize_text(&v,text,&out);return out.count;
}
'''
        unit = scratch / 'native-tokenizer.c'
        library = scratch / 'native-tokenizer.dylib'
        unit.write_text(header + helpers + '\n' + fragment + footer, encoding='utf-8')
        subprocess.run(['clang', '-O2', '-shared', '-fPIC', '-I', str(engine), str(unit), '-o', str(library)], check=True)
        native = ctypes.CDLL(str(library)).actual_split
        native.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        native.restype = ctypes.c_size_t
        rng = random.Random(1)
        pool = list('abc XYZ019{}();\r\n\t') + list('éà中語…“”¼①〇\u00a0\u2009\u2028') + ['e\u0301', '👩‍💻']
        texts = ['', '.”\n', '…', ' ½', '\u00a0word', '    int', '1234中文foo\n']
        texts += [''.join(rng.choice(pool) for _ in range(rng.randint(1, 100))) for _ in range(20000)]
        # Every Unicode scalar, with contexts that exercise all three splits.
        scalars = [chr(cp) for cp in range(1, 0x110000) if not 0xd800 <= cp <= 0xdfff]
        texts += [''.join(f' {c}a{c}{c}1 .{c}\n' for c in scalars[i:i+512]) for i in range(0, len(scalars), 512)]
        for file in [scratch / 'promessi_sposi.txt', scratch / 'round2-code.txt']:
            if file.exists(): texts.append(file.read_text(encoding='utf-8'))
        pieces = 0
        for text in texts:
            raw = text.encode()
            pairs = (ctypes.c_uint64 * (2 * (len(raw) + 1)))()
            count = native(raw, pairs, len(raw) + 1)
            actual = [raw[pairs[2*i]:pairs[2*i]+pairs[2*i+1]].decode() for i in range(count)]
            expected = [piece for piece, _ in reference.pre_tokenize_str(text)]
            self.assertEqual(actual, expected, repr(text[:100]))
            self.assertEqual(''.join(actual), text)
            pieces += count
        receipt = dict(texts=len(texts), pieces=pieces, all_unicode_scalars=True,
                       tokenizer_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
                       c_fragment_sha256=hashlib.sha256(fragment.encode()).hexdigest(),
                       pass_=True)
        (scratch / 'r4-tokenizer-proof.json').write_text(json.dumps(receipt, indent=2))
        print('PASS: actual C Unicode pre-tokenizer', json.dumps(receipt))


if __name__ == '__main__':
    unittest.main()
