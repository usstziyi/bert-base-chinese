import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from mind_reading.mindnet.eegtext_model import EEGTextModel


class ModelLossTests(unittest.TestCase):
    def test_scalar_loss_matches_forward_and_backpropagates(self):
        model = EEGTextModel(
            nn.Linear(4, 8), nn.Linear(5, 8),
            eeg_feature_dim=8, text_feature_dim=8,
            freeze_text_encoder=False,
        ).eval()
        eeg, text = torch.randn(3, 4), torch.randn(3, 5)
        word_ids = torch.tensor([0, 1, 0])
        output = model(eeg, text)
        with torch.no_grad():
            expected = model.contrastive_loss(
                output["eeg_embedding"], output["text_embedding"], word_ids,
            )
        with patch.object(model, "forward", side_effect=AssertionError("Unexpected forward")):
            loss = model.contrastive_loss(output["eeg_embedding"], output["text_embedding"], word_ids)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        for module in (model.eeg_encoder, model.text_encoder,
                       model.eeg_projection, model.text_projection):
            self.assertTrue(any(p.grad is not None for p in module.parameters()))

    def model(self):
        return EEGTextModel(nn.Identity(), nn.Identity(), eeg_feature_dim=3,
                            text_feature_dim=3, temperature=0.2)

    def test_unique_labels_recover_original_clip_loss(self):
        model = self.model()
        eeg, text = F.normalize(torch.randn(4, 3), dim=1), F.normalize(torch.randn(4, 3), dim=1)
        logits = eeg @ text.T / model.temperature
        expected = (F.cross_entropy(logits, torch.arange(4)) +
                    F.cross_entropy(logits.T, torch.arange(4))) / 2
        torch.testing.assert_close(model.contrastive_loss(eeg, text, torch.arange(4)), expected)

    def test_repeated_words_match_uniform_positive_targets_in_both_directions(self):
        model = self.model()
        eeg = F.normalize(torch.tensor([[1., 0., 0.], [0., 1., 0.],
                                        [0., 0., 1.], [0., 2., 1.]]), dim=1)
        text = F.normalize(torch.tensor([[1., 1., 0.], [0., 1., 1.],
                                         [1., 0., 1.], [0., 1., 1.]]), dim=1)
        word_ids = torch.tensor([10, 20, 30, 20])
        targets = torch.tensor([[1., 0., 0., 0.], [0., .5, 0., .5],
                                [0., 0., 1., 0.], [0., .5, 0., .5]])
        logits = eeg @ text.T / model.temperature
        expected = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets.T)) / 2
        torch.testing.assert_close(model.contrastive_loss(eeg, text, word_ids), expected)
        # 各方向不相同，确保测试覆盖了双向平均。
        self.assertFalse(torch.allclose(F.cross_entropy(logits, targets),
                                       F.cross_entropy(logits.T, targets.T)))

    def test_off_diagonal_positive_receives_attraction_gradient(self):
        logits = torch.zeros(3, 3, requires_grad=True)
        ids = torch.tensor([1, 2, 1])
        self.model().multi_positive_loss(logits, ids[:, None] == ids[None, :]).backward()
        self.assertLess(logits.grad[0, 2].item(), 0)  # 同字另一出现：提高相似度。
        self.assertGreater(logits.grad[0, 1].item(), 0)  # 不同字：降低相似度。

    def test_permutation_and_branch_exchange_leave_loss_unchanged(self):
        model = self.model()
        eeg, text = F.normalize(torch.randn(4, 3), dim=1), F.normalize(torch.randn(4, 3), dim=1)
        ids, order = torch.tensor([0, 1, 0, 2]), torch.tensor([2, 0, 3, 1])
        expected = model.contrastive_loss(eeg, text, ids)
        torch.testing.assert_close(model.contrastive_loss(eeg[order], text[order], ids[order]), expected)
        torch.testing.assert_close(model.contrastive_loss(text, eeg, ids), expected)

    def test_single_label_and_single_sample_are_finite(self):
        model = self.model()
        embeddings = torch.tensor([[1., 0., 0.]]).repeat(3, 1).requires_grad_()
        loss = model.contrastive_loss(embeddings, embeddings, torch.zeros(3, dtype=torch.long))
        torch.testing.assert_close(loss, torch.tensor(3.).log())
        loss.backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        torch.testing.assert_close(model.contrastive_loss(embeddings[:1], embeddings[:1],
                                                        torch.tensor([0])), torch.tensor(0.))

    def test_invalid_labels_and_empty_embeddings_are_rejected(self):
        model, embeddings = self.model(), torch.randn(3, 3)
        for ids in ([0, 1, 0], torch.tensor([0, 1]), torch.zeros(3, 1, dtype=torch.long),
                    torch.tensor([0., 1., 0.]), torch.tensor([True, False, True])):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, 'word_ids'):
                model.contrastive_loss(embeddings, embeddings, ids)
        with self.assertRaises(ValueError):
            model.contrastive_loss(embeddings[:0], embeddings[:0], torch.tensor([], dtype=torch.long))
        with self.assertRaises(ValueError):
            model.multi_positive_loss(torch.randn(2, 3), torch.zeros(2, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            model.multi_positive_loss(torch.randn(2, 3), torch.ones(2, 3))

    def test_extreme_logits_are_numerically_stable(self):
        logits = torch.tensor([[10000., -10000., 10000.], [-10000., 10000., -10000.]],
                              requires_grad=True)
        mask = torch.tensor([[True, False, True], [False, True, False]])
        loss = self.model().multi_positive_loss(logits, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
