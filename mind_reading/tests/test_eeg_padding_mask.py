import copy
import unittest

import torch
from torch import nn

from framework.eeg_encoder import EEGEncoder
from framework.eegtext_model import EEGTextModel


class EEGPaddingMaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder = EEGEncoder()

    def test_eval_matches_unpadded_samples(self):
        encoder = self.encoder.eval()
        lengths = torch.tensor([encoder.min_samples, 64, 97, 128])
        eeg = torch.randn(4, encoder.n_chans, 160)
        mask = torch.arange(160)[None, :] < lengths[:, None]
        with torch.no_grad():
            actual = encoder(eeg, mask)
            expected = torch.cat([
                encoder(eeg[i:i + 1, :, :length])
                for i, length in enumerate(lengths.tolist())
            ])
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    def test_training_padding_does_not_change_statistics_or_gradients(self):
        encoder = self.encoder.train()
        # Dropout 的随机采样随张量形状变化；此处关闭它来单独验证 padding。
        for layer in encoder.modules():
            if isinstance(layer, nn.Dropout):
                layer.p = 0
        other = copy.deepcopy(encoder)
        lengths = torch.tensor([64, 97])
        eeg = torch.randn(2, encoder.n_chans, 97, requires_grad=True)
        extended = torch.cat([eeg.detach(), torch.randn(2, encoder.n_chans, 63)], -1)
        mask = torch.arange(97)[None, :] < lengths[:, None]
        extended_mask = torch.arange(160)[None, :] < lengths[:, None]
        actual = encoder(eeg, mask)
        expected = other(extended, extended_mask)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        for name, value in encoder.named_buffers():
            torch.testing.assert_close(value, dict(other.named_buffers())[name])
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())
        self.assertEqual(eeg.grad[0, :, 64:].count_nonzero().item(), 0)
        self.assertGreater(eeg.grad[0, :, :64].abs().sum().item(), 0)

    def test_model_passes_mask_and_rejects_invalid_masks(self):
        model = EEGTextModel(
            self.encoder, nn.Identity(), text_feature_dim=8,
        ).eval()
        eeg = torch.randn(2, self.encoder.n_chans, 96)
        mask = torch.arange(96)[None, :] < torch.tensor([64, 96])[:, None]
        with torch.no_grad():
            output = model(eeg, torch.randn(2, 8), eeg_attention_mask=mask)
            torch.testing.assert_close(output["eeg_feature"], self.encoder(eeg, mask))
        for invalid in (mask[:, :-1], torch.zeros_like(mask), ~mask):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                self.encoder(eeg, invalid)


if __name__ == "__main__":
    unittest.main()
