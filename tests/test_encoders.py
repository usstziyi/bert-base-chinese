import unittest
from unittest.mock import patch

import torch
from transformers import BertConfig, BertModel

from framework.eeg_encoder import EEGEncoder
from framework.text_encoder import BertSentenceEncoder
from framework.eegtext_model import EEGTextModel


def tokenize(texts, **kwargs):
    ids = torch.tensor([[2, 4 + i % 2, 3, 0] for i in range(len(texts))])
    return {
        "input_ids": ids,
        "attention_mask": ids.ne(0).long(),
        "special_tokens_mask": ((ids < 4)).long(),
    }


def text_encoder(frozen):
    bert = BertModel(BertConfig(
        vocab_size=12, hidden_size=16, num_hidden_layers=4,
        num_attention_heads=2, intermediate_size=32, output_hidden_states=True,
    ))
    with patch("framework.text_encoder.AutoModel.from_pretrained", return_value=bert), patch(
        "framework.text_encoder.AutoTokenizer.from_pretrained", return_value=tokenize
    ):
        return BertSentenceEncoder(device=torch.device("cpu"), frozen=frozen)


class EncoderIntegrationTests(unittest.TestCase):
    def test_multimodal_training_and_freezing(self):
        for frozen in (True, False):
            with self.subTest(frozen=frozen):
                text = text_encoder(frozen)
                eeg = EEGEncoder(n_chans=4, n_times=128)
                model = EEGTextModel(
                    eeg, text, eeg_feature_dim=eeg.feature_dim,
                    text_feature_dim=text.feature_dim, freeze_text_encoder=frozen,
                ).train()
                output = model.compute_loss(torch.randn(2, 4, 128), ["你好", "再见"])
                self.assertEqual(output["eeg_feature"].shape, (2, eeg.feature_dim))
                self.assertEqual(output["text_feature"].shape, (2, 16))
                self.assertTrue(torch.isfinite(output["loss"]))
                torch.testing.assert_close(output["eeg_embedding"].norm(dim=-1), torch.ones(2))
                output["loss"].backward()
                self.assertIsNotNone(eeg.backbone.conv_temporal.weight.grad)
                self.assertIsNotNone(model.text_projection.projection[0].weight.grad)
                self.assertEqual(any(p.grad is not None for p in text.bert.parameters()), not frozen)
                self.assertEqual(text.bert.training, not frozen)

    def test_text_pooling_and_inference(self):
        encoder = text_encoder(True)
        inputs = tokenize(["你好", "再见"])
        with torch.no_grad():
            hidden = encoder.bert(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).hidden_states
        expected = torch.stack(hidden[-4:]).mean(0)[:, 1]
        torch.testing.assert_close(encoder(["你好", "再见"]), expected)
        torch.testing.assert_close(encoder(**inputs), expected)
        torch.testing.assert_close(encoder.encode_texts(["你好", "再见"]), expected)
        with self.assertRaises(ValueError):
            encoder([])

    def test_external_freeze_and_eeg_shapes(self):
        text = text_encoder(False)
        text.requires_grad_(False).train()
        self.assertFalse(text.bert.training)
        eeg = EEGEncoder(n_chans=4, n_times=1500, F2=24).eval()
        for length in (1000, 1500):
            self.assertEqual(eeg(torch.randn(2, 4, length)).shape, (2, 24))
        with self.assertRaises(ValueError):
            eeg(torch.randn(2, 3, 1500))


if __name__ == "__main__":
    unittest.main()
