"""Small CPU gradient oracle; this is not the real-model Jeff acceptance test."""
from types import SimpleNamespace
import io
import struct
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Run with the installed media/training Python')
class Dsv41OutputMathTests(unittest.TestCase):
    def model(self):
        from cosmos_windows.gguf_qat.streamed_model_deepseek41 import OutputLoRA, StreamedDeepSeek41Output
        torch.set_num_threads(1)
        cfg = SimpleNamespace(hidden=4, vocab=5, n_layer=2)
        model = StreamedDeepSeek41Output.__new__(StreamedDeepSeek41Output)
        torch.nn.Module.__init__(model)
        base = torch.zeros((5, 4), dtype=torch.float16)
        model.g = SimpleNamespace(dequant_2d_hf=lambda name, dtype: base)
        model.cfg, model.layer, model.device = cfg, 2, 'cpu'
        model.lora = OutputLoRA(cfg, 2, 4, [2], lambda layer, projection: (5, 4), 'cpu')
        model.head, model.closed_prefix = None, True
        return model, base

    def test_existing_lora_factors_receive_gradients_and_fused_loss_drops(self):
        model, base = self.model()
        hidden = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]])
        labels = torch.tensor([[-100, 2, 3]])
        optimizer = torch.optim.SGD(model.lora.parameters(), lr=0.3)
        losses = []
        for step in range(20):
            loss = model.suffix_loss(hidden, labels, 2, 3)
            self.assertTrue(torch.isfinite(loss))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.lora.parameters()))
            if step == 0:
                self.assertGreater(float(model.lora.B['l2_head'].grad.abs().sum()), 0)
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0] - 0.05)
        self.assertTrue(torch.equal(base, torch.zeros_like(base)))

    def test_shifted_supervision_ignores_masked_prompt_positions(self):
        model, _ = self.model()
        hidden = torch.ones((1, 3, 4))
        labels = torch.tensor([[-100, -100, 2]])
        first = model.suffix_loss(hidden, labels, 2, 3)
        hidden[:, 0] = 10000
        self.assertTrue(torch.equal(first, model.suffix_loss(hidden, labels, 2, 3)))

    def test_prefix_must_be_complete_and_provider_released(self):
        model, _ = self.model()
        with self.assertRaises(ValueError):
            model.suffix_loss(torch.zeros((1, 2, 4)), torch.tensor([[-100, 1]]), 1, 2)
        model.closed_prefix = False
        with self.assertRaises(ValueError):
            model.suffix_loss(torch.zeros((1, 2, 4)), torch.tensor([[-100, 1]]), 2, 2)

    def test_unsuccessful_native_provider_exit_never_releases_the_training_head(self):
        model, _ = self.model()
        model.closed_prefix = False
        model.url, model.headers = 'http://127.0.0.1:19001', {}
        model.owner = SimpleNamespace(wait=lambda timeout: 7)
        responses = [io.BytesIO(struct.pack('<8f', *([1.] * 8))), io.BytesIO(b'{"closing":true}')]
        with patch('urllib.request.urlopen', side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, 'exited unsuccessfully'):
                model.prefix_hidden(torch.tensor([[1, 2]]), 2)
        self.assertFalse(model.closed_prefix)
        self.assertIsNone(model.head)


if __name__ == '__main__':
    unittest.main()
