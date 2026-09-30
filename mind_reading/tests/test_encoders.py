import unittest
from unittest.mock import patch

import torch
from transformers import BertConfig, BertModel

from mind_reading.mindnet.eeg_encoder import EEGEncoder
from mind_reading.mindnet.text_encoder import TextEncoder
from mind_reading.mindnet.eegtext_model import EEGTextModel


def tokenize(texts, **kwargs):
    rows = [[2] + [4 + ord(char) % 8 for char in text] + [3] for text in texts]
    length = max(map(len, rows))
    ids = torch.tensor([row + [0] * (length - len(row)) for row in rows])
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
    with patch("mind_reading.mindnet.text_encoder.AutoModel.from_pretrained", return_value=bert), patch(
        "mind_reading.mindnet.text_encoder.AutoTokenizer.from_pretrained", return_value=tokenize
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
        frozen_weights = eeg.input_projection.weight.detach().clone()
        output = model(torch.randn(2, 4, 128), ["你好", "再见"])
        model.contrastive_loss(output["eeg_embedding"], output["text_embedding"], torch.tensor([0, 1])).backward()
        torch.testing.assert_close(eeg.input_projection.weight, frozen_weights)
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
            loss = model.contrastive_loss(output["eeg_embedding"], output["text_embedding"], torch.tensor([0, 1]))
            self.assertEqual(loss.device.type, "cuda")
            loss.backward()
            model.to("cpu")
            self.assertEqual(text.device.type, "cpu")

    def test_custom_transformer_and_invalid_configuration(self):
        eeg = EEGEncoder(
            n_chans=4, d_model=24, nhead=3, num_layers=1, dim_feedforward=48,
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
                eeg_input = torch.randn(3, 4, 128)
                output = model(eeg_input, ["你好", "再见", "你好"])
                self.assertEqual(output["eeg_feature"].shape, (3, eeg.feature_dim))
                self.assertEqual(output["text_feature"].shape, (3, 16))
                self.assertEqual(output["text_feature"].requires_grad, not frozen)
                loss = model.contrastive_loss(output["eeg_embedding"], output["text_embedding"],
                                              torch.tensor([0, 1, 0]))
                self.assertTrue(torch.isfinite(loss))
                torch.testing.assert_close(output["eeg_embedding"].norm(dim=-1), torch.ones(3))
                loss.backward()
                self.assertIsNotNone(eeg.input_projection.weight.grad)
                self.assertIsNotNone(eeg.transformer.layers[0].self_attn.in_proj_weight.grad)
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
        expected = torch.stack(hidden[-4:]).mean(0)[:, 1:3].mean(1)
        torch.testing.assert_close(encoder(["你好", "再见"]), expected)
        torch.testing.assert_close(encoder(**inputs), expected)
        with self.assertRaises(ValueError):
            encoder([])

    def test_single_char_and_multi_char_layer_selection_in_mixed_batch(self):
        encoder = text_encoder().eval()
        texts = ['我', '王子', '!']
        inputs = tokenize(texts)
        with torch.no_grad():
            outputs = encoder.bert(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            last_four_mean = torch.stack(outputs.hidden_states[-4:]).mean(0)
            expected = torch.stack([outputs.last_hidden_state[0, 1],
                                    last_four_mean[1, 1:3].mean(0), outputs.last_hidden_state[2, 1]])
            actual = encoder(texts)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(encoder(**inputs), expected)
            torch.testing.assert_close(encoder('我'), expected[:1])
            torch.testing.assert_close(encoder(['我', '!']), expected[[0, 2]])
            self.assertFalse(torch.allclose(actual[0], last_four_mean[0, 1]))

    def test_single_char_only_does_not_request_intermediate_layers(self):
        encoder = text_encoder().eval()
        with patch.object(encoder.bert, 'forward', wraps=encoder.bert.forward) as forward:
            encoder(['我', '你'])
            self.assertFalse(forward.call_args.kwargs['output_hidden_states'])
            encoder(['我', '王子'])
            self.assertTrue(forward.call_args.kwargs['output_hidden_states'])

    def test_multi_char_single_token_uses_raw_length_or_explicit_mask(self):
        encoder = text_encoder().eval()
        # 模拟多字符文本被 tokenizer 编成单个 token，不能据此认定原文是单字。
        inputs = tokenize(['我', '你'])
        with torch.no_grad():
            outputs = encoder.bert(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            expected = torch.stack([outputs.last_hidden_state[0, 1],
                                    torch.stack(outputs.hidden_states[-4:]).mean(0)[1, 1]])
            with patch.object(encoder, 'tokenizer', return_value=inputs):
                torch.testing.assert_close(encoder(['我', 'hello']), expected)
            torch.testing.assert_close(encoder(**inputs, single_char_mask=torch.tensor([True, False])), expected)
        for invalid in (torch.tensor([True]), torch.tensor([1, 0]), torch.ones(2, 1, dtype=torch.bool)):
            with self.subTest(mask=invalid), self.assertRaisesRegex(ValueError, 'single_char_mask'):
                encoder(**inputs, single_char_mask=invalid)

    def test_mixed_layer_selection_preserves_gradients(self):
        encoder = text_encoder().eval()
        encoder(['我', '王子'])[:, 0].sum().backward()
        gradients = [parameter.grad for parameter in encoder.bert.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(any(gradient.abs().sum() > 0 for gradient in gradients))

    def test_external_freeze_and_eeg_shapes(self):
        text = text_encoder()
        text.requires_grad_(False).train()
        self.assertTrue(text.bert.training)
        text(**tokenize(["你好"]))
        self.assertTrue(text.bert.training)
        eeg = EEGEncoder(n_chans=4, d_model=24).eval()
        for length in (1000, 1500):
            self.assertEqual(eeg(torch.randn(2, 4, length)).shape, (2, 24))
        with self.assertRaises(ValueError):
            eeg(torch.randn(2, 3, 1500))


if __name__ == "__main__":
    unittest.main()
