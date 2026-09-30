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
    def test_fixed_eval_groups_repeat_without_consuming_training_rng(self):
        samples = make_loader().dataset * 4
        source_generator = torch.Generator().manual_seed(19)
        source = DataLoader(samples, batch_size=4, shuffle=True, generator=source_generator)
        rng_before = source_generator.get_state().clone()
        eval_loader = train.make_batch_eval_loader(source, batch_size=4, seed=7)
        other = train.make_batch_eval_loader(source, batch_size=4, seed=7)
        self.assertEqual(eval_loader.dataset.indices, other.dataset.indices)
        self.assertEqual(sorted(eval_loader.dataset.indices), list(range(len(samples))))
        self.assertNotEqual(eval_loader.dataset.indices, list(range(len(samples))))
        self.assertTrue(eval_loader.drop_last)
        self.assertEqual(eval_loader.batch_size, 4)
        first_pass = [batch['word_id'].tolist() for batch in eval_loader]
        self.assertEqual(first_pass, [batch['word_id'].tolist() for batch in eval_loader])
        torch.testing.assert_close(source_generator.get_state(), rng_before)

    def test_batch_evaluation_uses_eval_without_mutating_weights_buffers_or_gradients(self):
        model = make_model().train()
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        state_before = {key: value.clone() for key, value in model.state_dict().items()}
        gradients_before = [parameter.grad.clone() for parameter in model.parameters()]
        observations = []

        def inspect_forward(module, inputs):
            observations.append((module.training, torch.is_grad_enabled()))

        hook = model.register_forward_pre_hook(inspect_forward)
        self.addCleanup(hook.remove)
        loader = train.make_batch_eval_loader(make_loader(batch_size=3), batch_size=3, seed=42)
        args = SimpleNamespace(eeg_scale=1.)
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = train.evaluate_batch_loss(model, loader, torch.device('cpu'), args)
            repeated = train.evaluate_batch_loss(model, loader, torch.device('cpu'), args)
        self.assertEqual(metrics, repeated)
        self.assertEqual(observations, [(False, False), (False, False)])
        self.assertEqual(metrics['samples'], 3)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state_before[key])
        for parameter, gradient in zip(model.parameters(), gradients_before):
            torch.testing.assert_close(parameter.grad, gradient)

    def test_batch_loss_matches_manual_batch_candidates_and_reports_excluded_samples(self):
        model = make_model(identity_projections=True)
        # 固定批次为：同字/不同字/同字 + 一个尾部样本，只对前三条计算损失。
        loader = DataLoader(make_loader().dataset + [make_loader().dataset[1]], batch_size=3, drop_last=True)
        features, ids = torch.tensor([[1., 0.], [0., 1.], [1., 0.]]), torch.tensor([0, 1, 0])
        expected = model.contrastive_loss(features, features, ids).item()
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = train.evaluate_batch_loss(model, loader, torch.device('cpu'), SimpleNamespace(eeg_scale=1.))
        self.assertAlmostEqual(metrics['loss'], expected, places=6)
        self.assertEqual(metrics['samples'], 3)
        self.assertEqual(metrics['excluded_samples'], 1)
        self.assertEqual(metrics['skipped_batches'], 0)

    def test_eval_reports_skipped_single_word_batches_and_rejects_unusable_split(self):
        samples = make_loader().dataset
        loader = DataLoader([samples[0], samples[2], samples[0], samples[1], samples[0]],
                            batch_size=2, drop_last=True)
        model = make_model(identity_projections=True)
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = train.evaluate_batch_loss(model, loader, torch.device('cpu'), SimpleNamespace(eeg_scale=1.))
        self.assertEqual(metrics['samples'], 2)
        self.assertEqual(metrics['skipped_batches'], 1)
        self.assertEqual(metrics['excluded_samples'], 3)
        with self.assertRaisesRegex(ValueError, 'full batch'):
            train.make_batch_eval_loader(make_loader(), batch_size=4, seed=42)
        with self.assertRaises(ValueError):
            train.make_batch_eval_loader(make_loader(), batch_size=1, seed=42)
        same_word_loader = DataLoader([samples[0]] * 4, batch_size=2, drop_last=True)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'distinct word_ids'):
            train.evaluate_batch_loss(model, same_word_loader, torch.device('cpu'), SimpleNamespace(eeg_scale=1.))

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
        loader = DataLoader(make_loader().dataset + [make_loader().dataset[1]], batch_size=2)
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
            self.assertEqual(config['checkpoint_selection_metric'], 'val_batch_loss')
            self.assertEqual(config['batch_evaluation']['model_mode'], 'eval')
            for name in ('val_loss', 'eeg_to_text_r1', 'text_to_eeg_r1',
                         'train_eval_batch_loss', 'val_batch_loss'):
                self.assertIsInstance(metrics[name], float)
            self.assertAlmostEqual(metrics['train_eval_batch_loss'], metrics['val_batch_loss'], places=6)
            self.assertEqual(metrics['train_eval_batch_samples'], 4)
            self.assertEqual(metrics['val_batch_samples'], 4)
            for name in ('best.pt', 'last.pt'):
                checkpoint = torch.load(output_dir / name, weights_only=True)
                self.assertEqual(checkpoint['metrics'], metrics)
                self.assertEqual(checkpoint['best_val_batch_loss'], metrics['val_batch_loss'])

    def test_best_checkpoint_uses_val_batch_loss_instead_of_full_library_loss(self):
        loader = DataLoader(make_loader().dataset + [make_loader().dataset[1]], batch_size=2)
        batch_results = [{'loss': loss, 'samples': 4, 'excluded_samples': 0, 'skipped_batches': 0}
                         for loss in (0.2, 0.3, 0.1, 0.4)]
        full_results = [{'val_loss': loss, 'eeg_to_text_r1': 0., 'text_to_eeg_r1': 0.}
                        for loss in (1.2, 1.0)]
        with tempfile.TemporaryDirectory() as directory:
            with patch('sys.argv', ['train', '--epochs', '2', '--batch-size', '2', '--run-num', '2',
                                    '--device', 'cpu', '--eeg-scale', '1', '--output-dir', directory]), \
                    patch.object(train, 'make_loaders', return_value=(loader, loader, 2)), \
                    patch.object(train, 'TextEncoder', side_effect=WordTextEncoder), \
                    patch.object(train, 'evaluate_batch_loss', side_effect=batch_results), \
                    patch.object(train, 'validate', side_effect=full_results), \
                    contextlib.redirect_stdout(io.StringIO()):
                train.main()
            output_dir, = Path(directory).iterdir()
            best = torch.load(output_dir / 'best.pt', weights_only=True)
            last = torch.load(output_dir / 'last.pt', weights_only=True)
            self.assertEqual(best['epoch'], 1)
            self.assertEqual(last['epoch'], 2)
            self.assertEqual(best['metrics']['val_batch_loss'], 0.3)
            self.assertEqual(last['best_val_batch_loss'], 0.3)


if __name__ == '__main__':
    unittest.main()
