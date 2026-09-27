import unittest
from unittest.mock import patch

import torch
from transformers import BertConfig, BertModel

from mindnet.eeg_encoder import EEGEncoder
from mindnet.text_encoder import TextEncoder
from mindnet.eegtext_model import EEGTextModel


def tokenize(texts, **kwargs):
    ids = torch.tensor([[2, 4 + i % 2, 3, 0] for i in range(len(texts))])
    return {
        "input_ids": ids,
        "attention_mask": ids.ne(0).long(),
        "special_tokens_mask": ((ids < 4)).long(),
    }


def text_encoder():
    bert = BertModel(BertConfig(
        vocab_size=12, hidden_size=16, num_hidden_layers=4,
        num_attention_heads=2, intermediate_size=32, output_hidden_states=True,
    ))
    with patch("mindnet.text_encoder.AutoModel.from_pretrained", return_value=bert), patch(
        "mindnet.text_encoder.AutoTokenizer.from_pretrained", return_value=tokenize
    ):
        return TextEncoder()


class EncoderIntegrationTests(unittest.TestCase):
    def test_model_controls_freezing_and_infers_dimensions(self):
        eeg = EEGEncoder(n_chans=4)
        text = text_encoder()
        model = EEGTextModel(
            eeg, text, freeze_eeg_encoder=True, freeze_text_encoder=False,
        ).train()
        self.assertFalse(eeg.training)
        self.assertTrue(text.bert.training)
        self.assertTrue(all(p.requires_grad for p in text.parameters()))
        running_mean = eeg.backbone.bnorm_temporal.running_mean.clone()
        output = model(torch.randn(2, 4, 128), ["你好", "再见"])
        model.contrastive_loss(output["eeg_embedding"], output["text_embedding"]).backward()
        torch.testing.assert_close(eeg.backbone.bnorm_temporal.running_mean, running_mean)
        self.assertTrue(all(p.grad is None for p in eeg.parameters()))
        self.assertTrue(any(p.grad is not None for p in text.parameters()))
        model.set_encoder_freeze(model.eeg_encoder, False)
        self.assertTrue(eeg.training)
        self.assertTrue(all(p.requires_grad for p in eeg.parameters()))
        model.eval().train()
        self.assertTrue(eeg.training)
        model.set_encoder_freeze(text, True)
        model.eval().train()
        self.assertFalse(text.bert.training)
        self.assertTrue(all(not p.requires_grad for p in text.parameters()))
        model.eval()
        model.set_encoder_freeze(text, False)
        self.assertFalse(text.bert.training)
        self.assertTrue(all(p.requires_grad for p in text.parameters()))
        model.train()
        self.assertTrue(text.bert.training)

    def test_device_is_owned_by_caller(self):
        # 即使系统报告 GPU 可用，构造编码器也不能自行迁移设备。
        with patch("torch.cuda.is_available", return_value=True):
            text = text_encoder()
        self.assertEqual(text.device.type, "cpu")
        model = EEGTextModel(EEGEncoder(n_chans=4), text).to("cpu")
        self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))
        if torch.cuda.is_available():
            model.to("cuda")
            output = model(torch.randn(2, 4, 128, device="cuda"), ["你好", "再见"])
            loss = model.contrastive_loss(output["eeg_embedding"], output["text_embedding"])
            self.assertEqual(loss.device.type, "cuda")
            loss.backward()
            model.to("cpu")
            self.assertEqual(text.device.type, "cpu")

    def test_custom_pooling_and_invalid_configuration(self):
        eeg = EEGEncoder(
            n_chans=4, m1=15, m2=31, s=3,
        ).eval()
        with self.assertRaises(ValueError):
            eeg(torch.randn(2, 4, eeg.min_samples - 1))
        self.assertEqual(eeg(torch.randn(2, 4, eeg.min_samples)).shape, (2, eeg.feature_dim))
        text = text_encoder()
        with self.assertRaises(ValueError):
            EEGTextModel(eeg, text, temperature=0)
        with self.assertRaises(ValueError):
            EEGTextModel(eeg, text, text_feature_dim=768)

    def test_multimodal_training_and_freezing(self):
        for frozen in (True, False):
            with self.subTest(frozen=frozen):
                text = text_encoder()
                eeg = EEGEncoder(n_chans=4)
                model = EEGTextModel(
                    eeg, text, eeg_feature_dim=eeg.feature_dim,
                    text_feature_dim=text.feature_dim, freeze_text_encoder=frozen,
                ).eval().train()
                self.assertEqual(text.training, not frozen)
                self.assertTrue(model.text_projection.training)
                eeg_input = torch.randn(2, 4, 128)
                output = model(eeg_input, ["你好", "再见"])
                self.assertEqual(output["eeg_feature"].shape, (2, eeg.feature_dim))
                self.assertEqual(output["text_feature"].shape, (2, 16))
                self.assertEqual(output["text_feature"].requires_grad, not frozen)
                loss = model.contrastive_loss(output["eeg_embedding"], output["text_embedding"])
                self.assertTrue(torch.isfinite(loss))
                torch.testing.assert_close(output["eeg_embedding"].norm(dim=-1), torch.ones(2))
                loss.backward()
                self.assertIsNotNone(eeg.backbone.conv_temporal.weight.grad)
                self.assertIsNotNone(model.text_projection.projection[0].weight.grad)
                self.assertEqual(any(p.grad is not None for p in text.bert.parameters()), not frozen)
                self.assertEqual(text.bert.training, not frozen)

    def test_text_pooling_and_inference(self):
        encoder = text_encoder().eval()
        inputs = tokenize(["你好", "再见"])
        with torch.no_grad():
            hidden = encoder.bert(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).hidden_states
        expected = torch.stack(hidden[-4:]).mean(0)[:, 1]
        torch.testing.assert_close(encoder(["你好", "再见"]), expected)
        torch.testing.assert_close(encoder(**inputs), expected)
        with self.assertRaises(ValueError):
            encoder([])

    def test_external_freeze_and_eeg_shapes(self):
        text = text_encoder()
        text.requires_grad_(False).train()
        self.assertTrue(text.bert.training)
        text(**tokenize(["你好"]))
        self.assertTrue(text.bert.training)
        eeg = EEGEncoder(n_chans=4, k=24).eval()
        for length in (1000, 1500):
            self.assertEqual(eeg(torch.randn(2, 4, length)).shape, (2, 24))
        with self.assertRaises(ValueError):
            eeg(torch.randn(2, 3, 1500))


if __name__ == "__main__":
    unittest.main()
