import unittest

import torch
from torch.utils.data import Subset

from dataset import create_eegtext_dataloader


class EEGTextDataLoaderTests(unittest.TestCase):
    def test_subset_dynamic_padding_and_metadata(self):
        samples = [
            dict(eeg=torch.full((2, length), float(i + 1)), text=f"句子{i}",
                 novel_name="test", run_idx=0, run_num=1,
                 segment_idx=i, sfreq=100.0)
            for i, length in enumerate((2, 1601, 3))
        ]
        loader = create_eegtext_dataloader(
            Subset(samples, [1, 0, 2]), batch_size=2,
        )
        batches = list(loader)
        self.assertEqual(len(batches), 2)
        batch = batches[0]
        self.assertEqual(batch["eeg"].shape, (2, 2, 1601))
        self.assertEqual(batch["text"], ["句子1", "句子0"])
        self.assertEqual(batch["segment_idx"].tolist(), [1, 0])
        self.assertEqual(batch["attention_mask"].sum(dim=-1).tolist(), [1601, 2])
        self.assertEqual(batch["attention_mask"].tolist(),
                         [[True] * 1601, [True, True] + [False] * 1599])
        torch.testing.assert_close(batch["eeg"][0], samples[1]["eeg"])
        torch.testing.assert_close(batch["eeg"][1, :, :2], samples[0]["eeg"])
        torch.testing.assert_close(batch["eeg"][1, :, 2:], torch.zeros(2, 1599))
        self.assertEqual(batches[1]["text"], ["句子2"])
        self.assertEqual(batches[1]["eeg"].shape, (1, 2, 3))
        self.assertEqual(batches[1]["attention_mask"].sum(dim=-1).tolist(), [3])
        self.assertTrue(batches[1]["attention_mask"].all())


if __name__ == "__main__":
    unittest.main()
