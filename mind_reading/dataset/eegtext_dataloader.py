"""EEG–文本批处理与 DataLoader 创建入口。"""

from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def eeg_text_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """将 EEG 补零到当前 batch 的最长长度，保留每个样本的完整时间序列。"""
    if len(batch) == 0:
        raise ValueError("batch must not be empty")

    # (B,) 原始有效长度
    lengths = torch.tensor([sample["eeg_length"] for sample in batch], dtype=torch.long)
    # (C, T) → (T, C)，按 batch 最大长度补零，再恢复 (B, C, T_max)。
    eeg = pad_sequence(
        [sample["eeg"].transpose(0, 1) for sample in batch],
        batch_first=True,
        padding_value=0.0,
    ).transpose(1, 2).contiguous()
    max_len = eeg.shape[-1]

    attention_mask = (
        torch.arange(max_len).unsqueeze(0)  # (1, max_len)
        < lengths.unsqueeze(1)              # (B, 1)
    )                                       # (B, max_len) bool; 填充位为 False

    return {
        "eeg": eeg, #(B, C, T_max)
        "eeg_lengths": lengths, # EEG 原始有效长度 (B,)
        "attention_mask": attention_mask, #(B, max_len)
        "text": [sample["text"] for sample in batch],
        "novel_name": [sample["novel_name"] for sample in batch],
        "run_idx": torch.tensor([sample["run_idx"] for sample in batch], dtype=torch.long),
        "run_num": torch.tensor([sample["run_num"] for sample in batch], dtype=torch.long),
        "segment_idx": torch.tensor([sample["segment_idx"] for sample in batch], dtype=torch.long),
        "sfreq": torch.tensor([sample["sfreq"] for sample in batch], dtype=torch.float32)
    }


def create_eegtext_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """将 Dataset 或 Subset 封装为 EEG–文本加载器。

    EEG 补零到当前 batch 的最大长度，输出 (B, C, T_batch)；文本保留为字符串列表。
    训练时可设置 shuffle=True，并传入 generator 控制随机顺序。
    Windows 下 num_workers 默认取 0，避免复制预加载的数据。
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        collate_fn=eeg_text_collate_fn,
    )
