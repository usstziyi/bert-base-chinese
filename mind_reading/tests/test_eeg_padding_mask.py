import copy
import unittest

import torch
from torch import nn

from mind_reading.mindnet.eeg_encoder import EEGEncoder
from mind_reading.mindnet.eegtext_model import EEGTextModel


class EEGPaddingMaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder = EEGEncoder(drop_prob=0)

    def test_eval_matches_unpadded_samples(self):
        encoder = self.encoder.eval()
        lengths = torch.tensor([encoder.min_samples, 86, 97, 128])
        eeg = torch.randn(4, encoder.n_chans, 160)
        mask = torch.arange(160)[None, :] < lengths[:, None]
        with torch.no_grad():
            actual = encoder(eeg, mask)
            expected = torch.cat([
                encoder(eeg[i:i + 1, :, :length])
                for i, length in enumerate(lengths.tolist())
            ])
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    def test_training_padding_does_not_change_outputs_or_gradients(self):
        # 双精度降低不同序列长度下的浮点累加误差。
        encoder = self.encoder.double().train()
        other = copy.deepcopy(encoder)
        lengths = torch.tensor([86, 97])
        eeg = torch.randn(2, encoder.n_chans, 97, dtype=torch.float64, requires_grad=True)
        extended = torch.cat([eeg.detach(), torch.randn(2, encoder.n_chans, 63, dtype=eeg.dtype)], -1)
        extended.requires_grad_()
        mask = torch.arange(97)[None, :] < lengths[:, None]
        extended_mask = torch.arange(160)[None, :] < lengths[:, None]
        actual = encoder(eeg, mask)
        expected = other(extended, extended_mask)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        for name, value in encoder.named_buffers():
            torch.testing.assert_close(value, dict(other.named_buffers())[name])
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(eeg.grad, extended.grad[:, :, :97], atol=1e-5, rtol=1e-4)
        self.assertEqual(extended.grad[:, :, 97:].count_nonzero().item(), 0)
        for parameter, other_parameter in zip(encoder.parameters(), other.parameters()):
            torch.testing.assert_close(parameter.grad, other_parameter.grad, atol=1e-5, rtol=1e-4)
        self.assertTrue(torch.isfinite(eeg.grad).all())
        self.assertEqual(eeg.grad[0, :, 86:].count_nonzero().item(), 0)
        self.assertGreater(eeg.grad[0, :, :86].abs().sum().item(), 0)

    def test_model_passes_mask_and_rejects_invalid_masks(self):
        model = EEGTextModel(
            self.encoder, nn.Identity(), text_feature_dim=8,
        ).eval()
        eeg = torch.randn(2, self.encoder.n_chans, 96)
        mask = torch.arange(96)[None, :] < torch.tensor([86, 96])[:, None]
        with torch.no_grad():
            output = model(eeg, torch.randn(2, 8), eeg_attention_mask=mask)
            torch.testing.assert_close(output["eeg_feature"], self.encoder(eeg, mask))
        for invalid in (mask[:, :-1], torch.zeros_like(mask), ~mask):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                self.encoder(eeg, invalid)

    def test_transformer_lengths_and_all_valid_masks(self):
        encoder = self.encoder.eval()
        for length in (1, 2, 75, 250, 1500):
            with self.subTest(length=length), torch.no_grad():
                eeg = torch.randn(2, 128, length)
                self.assertEqual(encoder(eeg).shape, (2, encoder.feature_dim))
                torch.testing.assert_close(encoder(eeg, torch.ones(2, length)), encoder(eeg))

    def test_invalid_hyperparameters_and_empty_masked_sample(self):
        for options in ({"d_model": 0}, {"nhead": 0}, {"num_layers": 0},
                        {"dim_feedforward": -1}, {"d_model": 7, "nhead": 2},
                        {"n_chans": 0}, {"num_layers": 1.5}, {"drop_prob": 1.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                EEGEncoder(**options)
        eeg = torch.randn(2, 128, 96)
        mask = torch.arange(96)[None, :] < torch.tensor([0, 96])[:, None]
        with self.assertRaises(ValueError):
            self.encoder(eeg, mask)
        invalid = torch.ones(2, 96)
        invalid[0, 0] = 0.5
        with self.assertRaises(ValueError):
            self.encoder(eeg, invalid)

    def test_nonfinite_padding_is_ignored_in_forward_and_backward(self):
        eeg = torch.randn(2, 128, 9)
        mask = torch.arange(9)[None, :] < torch.tensor([1, 5])[:, None]
        reference = self.encoder(eeg, mask).detach()
        for value in (float("nan"), float("inf")):
            padded = eeg.masked_fill(~mask[:, None, :], value).requires_grad_()
            self.encoder.zero_grad(set_to_none=True)
            actual = self.encoder(padded, mask)
            torch.testing.assert_close(actual, reference)
            actual.square().sum().backward()
            self.assertTrue(torch.isfinite(padded.grad).all())
            self.assertEqual(padded.grad.masked_select(~mask[:, None, :]).count_nonzero().item(), 0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.encoder.parameters()))

    def test_temporal_order_matters_and_odd_feature_dimension_is_supported(self):
        encoder = EEGEncoder(n_chans=4, d_model=9, nhead=3, drop_prob=0).eval()
        eeg = torch.randn(2, 4, 12)
        with torch.no_grad():
            output = encoder(eeg)
            reversed_output = encoder(eeg.flip(-1))
        self.assertEqual(output.shape, (2, 9))
        self.assertFalse(torch.allclose(output, reversed_output))


if __name__ == "__main__":
    unittest.main()
