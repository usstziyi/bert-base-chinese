"""display 字符与 ROWS/ROWE EEG 的字典对齐回归测试。"""
import tempfile
import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import openpyxl
import pandas as pd
import torch
from torch.utils.data import Subset

from mind_reading.dataset import ChineseEEGDataset, create_eegtext_dataloader, load_text, load_eeg
from mind_reading.dataset.load_text import _load_display_words
from mind_reading.dataset.load_eeg import _row_intervals, split_row_eeg
from mind_reading.dataset.eegtext_dataloader import eeg_text_collate_fn


class FakeRaw:
    def __init__(self):
        self.ch_names = ['C1', 'C2']
        self.n_times = 2000
        self.data = np.stack([np.arange(2000), np.arange(2000) + 10000]).astype(float)
        self.closed = False

    def get_data(self, start, stop):
        return self.data[:, start:stop]

    def close(self):
        self.closed = True


class EEGTextDataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        folder = self.root / 'derivatives/novels/segmented_novel/test'
        folder.mkdir(parents=True)
        text_path_patch = patch('mind_reading.dataset.load_text.SEG_NOVEL_PATH', folder.parent)
        text_path_patch.start()
        self.addCleanup(text_path_patch.stop)
        eeg_path_patch = patch('mind_reading.dataset.load_eeg.EEG_PREPROC_DIR',
                               self.root / 'derivatives' / 'preproc')
        eeg_path_patch.start()
        self.addCleanup(eeg_path_patch.stop)
        self.display_path = folder / 'segmented_Chinense_novel_run_1_display.xlsx'
        self.records = [('10', 0, 0, 1)]
        for text, positions in [('你B6*—‘好’，', range(8)), ('天地人', range(3)), ('天地人', range(3))]:
            self.records.extend((text, p, 0, 1) for p in positions)
        self.write_display()
        # 不提供普通正文 xlsx，确保只依赖 display 表。
        events = [('ROWE', 0), ('CH10', 1), ('ROWS', 2), ('ROWE', 95),
                  ('ROWS', 100), ('ROWE', 924), ('ROWS', 1000), ('ROWE', 1194),
                  ('ROWS', 1300), ('ROWE', 1569)]
        eeg_dir = self.root / 'derivatives/preproc/filtered_0.5_30/sub-04/ses-test/eeg'
        eeg_dir.mkdir(parents=True)
        self.events_path = eeg_dir / 'sub-04_ses-test_task-reading_run-01_events.tsv'
        pd.DataFrame(events, columns=['trial_type', 'sample']).to_csv(self.events_path, sep='\t', index=False)

    def write_display(self):
        wb = openpyxl.Workbook()
        wb.active.append(['Chinese_text', 'index', 'main_row', 'row_num'])
        for record in self.records:
            wb.active.append(record)
        wb.save(self.display_path)
        wb.close()

    def dataset(self, **kwargs):
        raw = FakeRaw()
        with patch('mind_reading.dataset.load_eeg.mne.io.read_raw_brainvision', return_value=raw):
            dataset = ChineseEEGDataset(novel_name='test', run_num=1, **kwargs)
        self.assertTrue(raw.closed)
        return dataset

    def test_display_only_auto_discovery_and_all_characters(self):
        words = load_text('test', None)
        self.assertIsInstance(words, list)
        self.assertEqual(len(words), 15)
        self.assertEqual([w['row_num'] for w in words], [1] + [2] * 8 + [3] * 3 + [4] * 3)
        self.assertTrue(words[0]['is_chapter'])
        chars = words[1:9]
        self.assertEqual([c['text'] for c in chars], list('你B6*—‘好’'))
        self.assertEqual([c['char_idx'] for c in chars], list(range(8)))
        self.assertTrue(all(c['row_num'] == 2 and c['run_num'] == 1 for c in chars))

    def test_flat_eeg_words_fixed_100_points_trim_and_right_pad(self):
        raw = FakeRaw()
        self.display_path.unlink()  # EEG 加载不需要小说 display 文件。
        with patch('mind_reading.dataset.load_eeg.mne.io.read_raw_brainvision', return_value=raw):
            eeg_words = load_eeg('test', run_num=1,
                                 row_char_counts={(1, 1): 1, (1, 2): 8, (1, 3): 3, (1, 4): 3})
        self.assertIsInstance(eeg_words, list)
        self.assertEqual(len(eeg_words), 15)
        self.assertEqual([w['row_num'] for w in eeg_words], [1] + [2] * 8 + [3] * 3 + [4] * 3)
        self.assertEqual([w['is_padding'] for w in eeg_words],
                         [True] + [False] * 8 + [False, True, True] + [False, False, True])
        for eeg_word in eeg_words:
            self.assertIsInstance(eeg_word['is_padding'], bool)
            self.assertEqual(eeg_word['eeg'].shape, (2, 100))
            self.assertNotIn('is_chapter', eeg_word)
            self.assertNotIn('n_chars', eeg_word)
        np.testing.assert_array_equal(eeg_words[0]['eeg'][:, :94], raw.data[:, 2:96])
        self.assertFalse(eeg_words[0]['eeg'][:, 94:].any())
        np.testing.assert_array_equal(np.concatenate([w['eeg'] for w in eeg_words[1:9]], axis=1),
                                      raw.data[:, 100:900])  # 丢弃本行末尾 25 点。
        np.testing.assert_array_equal(eeg_words[9]['eeg'], raw.data[:, 1000:1100])
        np.testing.assert_array_equal(eeg_words[10]['eeg'][:, :95], raw.data[:, 1100:1195])
        self.assertFalse(eeg_words[10]['eeg'][:, 95:].any())
        self.assertFalse(eeg_words[11]['eeg'].any())  # 缺少整片时也右补零。
        np.testing.assert_array_equal(eeg_words[12]['eeg'], raw.data[:, 1300:1400])
        np.testing.assert_array_equal(eeg_words[14]['eeg'][:, :70], raw.data[:, 1500:1570])
        self.assertFalse(eeg_words[14]['eeg'][:, 70:].any())
        self.assertTrue(raw.closed)

    def test_dataset_keeps_symbols_and_pairs_by_row_and_index(self):
        dataset = self.dataset()
        self.assertEqual(len(dataset), 14)
        self.assertEqual([dataset[i]['text'] for i in range(8)], list('你B6*—‘好’'))
        self.assertEqual(dataset[6]['text'], '好')
        self.assertEqual(dataset[6]['row_num'], 2)
        self.assertEqual(dataset[6]['char_idx'], 6)
        self.assertEqual(dataset[6]['eeg'].shape, (2, 100))
        self.assertEqual(dataset[6]['eeg'][0, 0].item(), 700)
        with_chapters = self.dataset(include_chapters=True)
        self.assertEqual(len(with_chapters), 15)
        self.assertTrue(with_chapters[0]['is_chapter'])
        self.assertEqual(with_chapters[0]['text'], '1')  # 直接取章节记录 index=0

    def test_subset_batch_padding_and_metadata(self):
        dataset = self.dataset()
        batches = list(create_eegtext_dataloader(Subset(dataset, [10, 1, 11]), batch_size=2))
        self.assertEqual(batches[0]['eeg'].shape, (2, 2, 100))
        self.assertEqual(batches[0]['text'], ['人', 'B'])
        self.assertEqual(batches[0]['row_num'].tolist(), [3, 2])
        self.assertEqual(batches[0]['char_idx'].tolist(), [2, 1])
        self.assertEqual(batches[0]['is_padding'].dtype, torch.bool)
        self.assertEqual(batches[0]['is_padding'].tolist(), [True, False])
        self.assertFalse(batches[0]['eeg'][0].any())
        self.assertEqual(batches[1]['eeg'].shape, (1, 2, 100))
        self.assertEqual(batches[1]['row_num'].tolist(), [4])

    def test_dataset_can_exclude_partial_and_full_padding_without_shifting_pairs(self):
        dataset = self.dataset(include_padding=False)
        self.assertFalse(dataset.include_padding)
        self.assertEqual(len(dataset), 11)
        self.assertTrue(all(not dataset[i]['is_padding'] for i in range(len(dataset))))
        self.assertEqual([s['word_idx'] for s in dataset.samples], list(range(1, 10)) + [12, 13])
        self.assertEqual([(dataset[i]['row_num'], dataset[i]['char_idx'], dataset[i]['text'])
                          for i in range(8, 11)], [(3, 0, '天'), (4, 0, '天'), (4, 1, '地')])
        self.assertEqual(dataset[9]['eeg'][0, 0].item(), 1300)
        # 章节本身含补零，include_chapters=True 也不能绕过 padding 过滤。
        self.assertEqual(len(self.dataset(include_chapters=True, include_padding=False)), 11)

    def test_word_ids_follow_text_identity_across_rows_filtering_and_batches(self):
        dataset = self.dataset()
        self.assertEqual(dataset[8]['text'], dataset[11]['text'])
        self.assertEqual(dataset[8]['word_id'], dataset[11]['word_id'])
        self.assertNotEqual(dataset[8]['word_id'], dataset[9]['word_id'])
        filtered = self.dataset(include_padding=False)
        self.assertEqual(filtered[8]['word_id'], dataset[8]['word_id'])
        batches = list(create_eegtext_dataloader(Subset(dataset, [8, 9, 11]), batch_size=2))
        self.assertEqual(batches[0]['word_id'].dtype, torch.long)
        self.assertEqual(batches[0]['word_id'][0], batches[1]['word_id'][0])

    def test_dataset_rejects_empty_result_after_padding_filtering(self):
        events = pd.read_csv(self.events_path, sep='\t')
        for start, stop in [(2, 95), (100, 924), (1000, 1194), (1300, 1569)]:
            events.loc[events['sample'] == stop, 'sample'] = start + 49
        events.to_csv(self.events_path, sep='\t', index=False)
        with self.assertRaisesRegex(ValueError, 'after chapter/padding filtering'):
            self.dataset(include_padding=False)

    def test_padding_flag_uses_length_rather_than_signal_values(self):
        raw = FakeRaw()
        raw.data.fill(0)
        events = pd.read_csv(self.events_path, sep='\t')
        events.loc[events['sample'] == 1194, 'sample'] = 1299
        events.to_csv(self.events_path, sep='\t', index=False)
        with patch('mind_reading.dataset.load_eeg.mne.io.read_raw_brainvision', return_value=raw):
            words = load_eeg('test', run_num=1,
                             row_char_counts={(1, 1): 1, (1, 2): 8, (1, 3): 3, (1, 4): 3})
        self.assertEqual([w['is_padding'] for w in words[9:12]], [False, False, False])

    def test_main_row_selection_and_repeated_text_rows(self):
        self.records = [('你好人\n背景\n下行', index, 0, 3) for index in range(3)] + [
                        ('你好人', 0, 0, 1), ('你好人', 1, 0, 1), ('你好人', 2, 0, 1)]
        self.write_display()
        words = _load_display_words(self.display_path, 1)
        self.assertEqual([w['row_num'] for w in words], [1, 1, 1, 2, 2, 2])
        self.assertEqual([w['text'] for w in words], list('你好人你好人'))

    def test_flat_order_across_runs_and_independent_discovery(self):
        shutil.copyfile(self.display_path, self.display_path.with_name(self.display_path.name.replace('run_1_', 'run_2_')))
        shutil.copyfile(self.events_path, self.events_path.with_name(self.events_path.name.replace('run-01_', 'run-02_')))
        words = load_text('test', None)
        self.assertEqual(len(words), 30)
        self.assertEqual(words[15]['run_num'], 2)
        self.assertEqual(words[15]['row_num'], 1)
        self.assertEqual(words[15]['char_idx'], 0)
        with patch('mind_reading.dataset.load_eeg.mne.io.read_raw_brainvision', side_effect=[FakeRaw(), FakeRaw()]):
            dataset = ChineseEEGDataset('test', run_num=None, include_chapters=True)
        self.assertEqual(len(dataset.eeg_words), 30)
        self.assertEqual(dataset[1]['word_id'], dataset[16]['word_id'])
        self.assertEqual([(w['run_num'], w['row_num'], w['char_idx']) for w in words],
                         [(w['run_num'], w['row_num'], w['char_idx']) for w in dataset.eeg_words])
        for path in self.display_path.parent.glob('*.xlsx'):
            path.unlink()
        with patch('mind_reading.dataset.load_eeg.mne.io.read_raw_brainvision', side_effect=[FakeRaw(), FakeRaw()]):
            eeg_words = load_eeg('test', run_num=None,
                                 row_char_counts={(w['run_num'], w['row_num']): w['n_chars'] for w in words})
        self.assertEqual(len(eeg_words), 30)
        self.assertEqual((eeg_words[15]['run_num'], eeg_words[15]['row_num'], eeg_words[15]['char_idx']), (2, 1, 0))

    def test_incomplete_or_shuffled_words_are_rejected(self):
        words = load_text('test', 1)
        for invalid in (words[:3] + words[4:], [words[0], words[2], words[1]] + words[3:]):
            with patch('mind_reading.dataset.eegtext_dataset.load_text', return_value=invalid), \
                    patch('mind_reading.dataset.load_eeg.mne.io.read_raw_brainvision') as reader:
                with self.assertRaisesRegex(ValueError, 'incomplete or unordered'):
                    ChineseEEGDataset('test', run_num=1)
                reader.assert_not_called()

    def test_invalid_display_and_row_mismatch_fail_explicitly(self):
        self.records[1] = ('你', 8, 0, 1)
        self.write_display()
        with self.assertRaisesRegex(ValueError, 'row 3: invalid display record'):
            load_text('test', 1)
        self.records[1] = ('你B6*—‘好’，', 0, 0, 1)
        self.write_display()
        events = pd.read_csv(self.events_path, sep='\t').iloc[:-2]
        events.to_csv(self.events_path, sep='\t', index=False)
        with self.assertRaisesRegex(ValueError, 'row mismatch'):
            self.dataset()
        self.display_path.unlink()
        with self.assertRaises(FileNotFoundError):
            load_text('test', 1)

    def test_short_rows_pad_without_removing_slots(self):
        words = split_row_eeg(np.ones((2, 105)), 3)
        self.assertEqual(words.shape, (3, 2, 100))
        self.assertTrue((words[0] == 1).all())
        self.assertTrue((words[1, :, :5] == 1).all())
        self.assertFalse(words[1, :, 5:].any())
        self.assertFalse(words[2].any())

    def test_dataset_pads_missing_segments_within_each_row(self):
        events = pd.read_csv(self.events_path, sep='\t')
        events.loc[events['sample'] == 1194, 'sample'] = 1104
        events.to_csv(self.events_path, sep='\t', index=False)
        dataset = self.dataset()
        np.testing.assert_array_equal(dataset[8]['eeg'], FakeRaw().data[:, 1000:1100])
        np.testing.assert_array_equal(dataset[9]['eeg'][:, :5], FakeRaw().data[:, 1100:1105])
        self.assertFalse(dataset[9]['eeg'][:, 5:].any())
        self.assertFalse(dataset[10]['eeg'].any())
        self.assertEqual(dataset[11]['eeg'][0, 0].item(), 1300)

    def test_bad_event_pairing_is_rejected(self):
        for events in ([('CH01', 1), ('ROWE', 5)],
                       [('CH01', 1), ('ROWS', 5), ('ROWS', 6)],
                       [('CH01', 1), ('ROWS', 5)],
                       [('ROWS', 1), ('ROWE', 5)],
                       [('CH01', 1), ('ROWS', 5), ('ROWE', 4)]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                _row_intervals(pd.DataFrame(events, columns=['trial_type', 'sample']), 'test')

    def test_row_counts_must_match_events_and_be_positive(self):
        counts = {(1, 1): 1, (1, 2): 8, (1, 3): 3, (1, 4): 3}
        for invalid in ({(1, 1): 1}, {**counts, (2, 1): 1}, {**counts, (1, 2): 0}):
            with self.subTest(counts=invalid), \
                    patch('mind_reading.dataset.load_eeg.mne.io.read_raw_brainvision') as reader:
                with self.assertRaises(ValueError):
                    load_eeg('test', run_num=1, row_char_counts=invalid)
                reader.assert_not_called()

    def test_recording_end_limits_read_and_right_pads(self):
        raw = FakeRaw()
        raw.n_times = 1540
        with patch('mind_reading.dataset.load_eeg.mne.io.read_raw_brainvision', return_value=raw):
            words = load_eeg('test', run_num=1,
                             row_char_counts={(1, 1): 1, (1, 2): 8, (1, 3): 3, (1, 4): 3})
        np.testing.assert_array_equal(words[-1]['eeg'][:, :40], raw.data[:, 1500:1540])
        self.assertFalse(words[-1]['eeg'][:, 40:].any())
        self.assertTrue(words[-1]['is_padding'])
        self.assertTrue(raw.closed)

    def test_invalid_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            eeg_text_collate_fn([])
        with self.assertRaisesRegex(ValueError, 'shapes'):
            eeg_text_collate_fn([{'eeg': torch.zeros(2, 90)}, {'eeg': torch.zeros(2, 91)}])


if __name__ == '__main__':
    unittest.main()
