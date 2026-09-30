"""多正样本字词对齐的训练、跨 batch 验证和 checkpoint 集成测试。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from mind_reading import train
from mind_reading.mindnet.eegtext_model import EEGTextModel


class WordTextEncoder(nn.Module):
    feature_dim = 2
    max_length = 8

    def __init__(self):
        super().__init__()
        self.register_buffer('vectors', torch.eye(2))
        self.tokenizer = SimpleNamespace(name_or_path='test-word-encoder')

    def forward(self, texts):
        return self.vectors[[{'我': 0, '你': 1}[text] for text in texts]]


def make_loader(batch_size=2):
    # 第三个样本的文本表示与第一个相同，验证 argmax 会命中另一次出现。
    samples = [{'eeg': torch.eye(2)[word_id, :, None], 'text': text, 'word_id': word_id}
               for word_id, text in [(0, '我'), (1, '你'), (0, '我')]]
    return DataLoader(samples, batch_size=batch_size)


def make_model(identity_projections=False):
    model = EEGTextModel(nn.Flatten(start_dim=1), WordTextEncoder(), eeg_feature_dim=2,
                         projection_dim=2, projection_hidden_dim=4, temperature=0.2)
    if identity_projections:
        model.eeg_projection = nn.Identity()
        model.text_projection = nn.Identity()
    return model


class TrainingTests(unittest.TestCase):
    def test_chunked_validation_matches_full_multi_positive_loss_and_word_recall(self):
        model = make_model(identity_projections=True)
        loader = make_loader()
        features, ids = torch.tensor([[1., 0.], [0., 1.], [1., 0.]]), torch.tensor([0, 1, 0])
        expected = model.contrastive_loss(features, features, ids).item()
        for chunk_size in (1, 2, 3, 8):
            with self.subTest(chunk_size=chunk_size), contextlib.redirect_stdout(io.StringIO()):
                metrics = train.validate(model, loader, torch.device('cpu'),
                                         SimpleNamespace(batch_size=chunk_size, eeg_scale=1.))
                self.assertAlmostEqual(metrics['val_loss'], expected, places=6)
                self.assertEqual(metrics['eeg_to_text_r1'], 1.)
                self.assertEqual(metrics['text_to_eeg_r1'], 1.)

    def test_training_passes_word_ids_and_updates_projection_weights(self):
        torch.manual_seed(7)
        model = make_model()
        loader = make_loader(batch_size=3)
        weights_before = model.eeg_projection.projection[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with patch.object(model, 'contrastive_loss', wraps=model.contrastive_loss) as loss_call, \
                contextlib.redirect_stdout(io.StringIO()):
            loss = train.train_epoch(model, loader, optimizer, torch.device('cpu'),
                                     SimpleNamespace(eeg_scale=1., max_grad_norm=1.))
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        torch.testing.assert_close(loss_call.call_args.args[2], torch.tensor([0, 1, 0]))
        self.assertFalse(torch.equal(weights_before, model.eeg_projection.projection[0].weight))

    def test_training_skips_batches_without_negative_words(self):
        model = make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        args = SimpleNamespace(eeg_scale=1., max_grad_norm=1.)
        with patch.object(model, 'contrastive_loss', wraps=model.contrastive_loss) as loss_call, \
                contextlib.redirect_stdout(io.StringIO()):
            train.train_epoch(model, make_loader(), optimizer, torch.device('cpu'), args)
        self.assertEqual(loss_call.call_count, 1)  # singleton 尾批被跳过。
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'distinct word_ids'):
            train.train_epoch(model, make_loader(batch_size=1), optimizer, torch.device('cpu'), args)
        same_word_batch = {'eeg': torch.ones(3, 2, 1), 'text': ['我'] * 3,
                           'word_id': torch.zeros(3, dtype=torch.long)}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'distinct word_ids'):
            train.train_epoch(model, [same_word_batch], optimizer, torch.device('cpu'), args)

    def test_main_runs_one_epoch_and_saves_numeric_metrics(self):
        loader = make_loader()
        with tempfile.TemporaryDirectory() as directory:
            with patch('sys.argv', ['train', '--epochs', '1', '--batch-size', '2', '--run-num', '2',
                                    '--device', 'cpu', '--eeg-scale', '1', '--no-include-padding',
                                    '--output-dir', directory]), \
                    patch.object(train, 'make_loaders', return_value=(loader, loader, 2)), \
                    patch.object(train, 'TextEncoder', side_effect=WordTextEncoder), \
                    contextlib.redirect_stdout(io.StringIO()):
                train.main()
            output_dir, = Path(directory).iterdir()
            config = json.loads((output_dir / 'config.json').read_text(encoding='utf-8'))
            metrics = json.loads((output_dir / 'history.jsonl').read_text(encoding='utf-8'))
            self.assertFalse(config['include_padding'])
            self.assertEqual(config['n_chans'], 2)
            self.assertEqual(config['contrastive_objective'], 'symmetric_multi_positive_word_identity')
            for name in ('val_loss', 'eeg_to_text_r1', 'text_to_eeg_r1'):
                self.assertIsInstance(metrics[name], float)
            for name in ('best.pt', 'last.pt'):
                checkpoint = torch.load(output_dir / name, weights_only=True)
                self.assertEqual(checkpoint['metrics'], metrics)


if __name__ == '__main__':
    unittest.main()
