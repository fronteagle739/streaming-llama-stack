import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from slstack import journal


class OverlappedSha256Tests(unittest.TestCase):
    def check(self, size, block_bytes=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.gguf'
            data = os.urandom(size)
            path.write_bytes(data)
            seen = []
            if block_bytes:
                original = journal._blocks
                journal._blocks = lambda p: original(p, block_bytes=block_bytes, depth=2)
            try:
                digest = journal.sha256(path, progress=lambda done, total: seen.append((done, total)))
            finally:
                if block_bytes:
                    journal._blocks = original
            self.assertEqual(digest, hashlib.sha256(data).hexdigest())
            if size:
                self.assertEqual(seen[-1], (size, size))
                self.assertEqual([d for d, _ in seen], sorted(d for d, _ in seen))
            else:
                self.assertEqual(seen, [])

    def test_digest_matches_a_plain_hash_across_block_boundaries(self):
        self.check(3 * 1024 * 1024 + 12345, block_bytes=1024 * 1024)

    def test_small_blocks_and_a_shallow_queue_keep_byte_order(self):
        self.check(1_000_003, block_bytes=4096)

    def test_empty_file(self):
        self.check(0)

    def test_reader_failure_is_raised(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(OSError):
                list(journal._blocks(Path(folder) / 'missing.gguf'))

    def test_consumer_stopping_early_does_not_hang(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.gguf'
            path.write_bytes(os.urandom(2 * 1024 * 1024))
            blocks = journal._blocks(path, block_bytes=4096, depth=1)
            next(blocks)
            blocks.close()  # runs the generator's finally: the reader must exit


if __name__ == '__main__':
    unittest.main()
