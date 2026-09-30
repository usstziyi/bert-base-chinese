"""逐字 EEG–文本批处理：固定长度 EEG 堆叠为 (B,C,S)。"""
import torch
from torch.utils.data import DataLoader, Dataset


def eeg_text_collate_fn(batch):
    """堆叠固定长度 EEG，并汇总文本与字符位置信息。"""
    if not batch:
        raise ValueError('batch must not be empty')
    shape = batch[0]['eeg'].shape
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError('Each EEG sample must have shape (channels, samples)')
    if any(sample['eeg'].shape != shape for sample in batch):
        raise ValueError('Character EEG shapes must match within a batch')

    output = {
        'eeg': torch.stack([s['eeg'] for s in batch]), # (B,C,100)
    }
    for key in ('text', 'novel_name', 'subject'):
        output[key] = [s[key] for s in batch]
    for key in ('run_idx', 'run_num', 'row_num', 'row_idx', 'char_idx', 'n_chars'):
        output[key] = torch.tensor([s[key] for s in batch], dtype=torch.long)
    output['is_chapter'] = torch.tensor([s['is_chapter'] for s in batch], dtype=torch.bool)
    return output


def create_eegtext_dataloader(dataset: Dataset, *, batch_size=32, shuffle=False,
                              num_workers=0, pin_memory=False, drop_last=False,
                              generator=None):
    """支持 Dataset/Subset；Windows 默认 num_workers=0 避免复制预加载片段。"""
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
