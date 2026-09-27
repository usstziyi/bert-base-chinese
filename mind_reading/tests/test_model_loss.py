import unittest
from unittest.mock import patch

import torch
from torch import nn

from framework.eegtext_model import EEGTextModel


class ModelLossTests(unittest.TestCase):
    def test_scalar_loss_matches_forward_and_backpropagates(self):
        model = EEGTextModel(
            nn.Linear(4, 8), nn.Linear(5, 8),
            eeg_feature_dim=8, text_feature_dim=8,
            freeze_text_encoder=False,
        ).eval()
        eeg, text = torch.randn(3, 4), torch.randn(3, 5)
        output = model(eeg, text)
        with torch.no_grad():
            expected = model.contrastive_loss(
                output["eeg_embedding"], output["text_embedding"],
            )
        with patch.object(model, "forward", side_effect=AssertionError("Unexpected forward")):
            loss = model.contrastive_loss(output["eeg_embedding"], output["text_embedding"])
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        for module in (model.eeg_encoder, model.text_encoder,
                       model.eeg_projection, model.text_projection):
            self.assertTrue(any(p.grad is not None for p in module.parameters()))


if __name__ == "__main__":
    unittest.main()
