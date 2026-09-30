"""在项目根目录运行 python -m mind_reading.dataset.debug_eegtext_dataset。"""
from torch.utils.data import Subset
from itertools import groupby

if __package__:
    from .eegtext_dataloader import create_eegtext_dataloader
    from .eegtext_dataset import ChineseEEGDataset
else:
    from eegtext_dataloader import create_eegtext_dataloader
    from eegtext_dataset import ChineseEEGDataset


if __name__ == '__main__':
    dataset = ChineseEEGDataset(novel_name='LittlePrince', subject='sub-04', run_num=7)
    print(f'Character samples: {len(dataset)}, samples per character: {dataset.samples_per_char}')
    for run_num, items in groupby(dataset.eeg_words, key=lambda word: word['run_num']):
        items = list(items)
        print(f'Run {run_num}: rows={len({w["row_num"] for w in items})}, '
              f'body characters={sum(not w["is_chapter"] for w in items)}')
    train_indices = [i for i, s in enumerate(dataset.samples) if s['run_idx'] != 6]
    val_indices = [i for i, s in enumerate(dataset.samples) if s['run_idx'] == 6]
    for name, indices in [('train', train_indices), ('val', val_indices)]:
        loader = create_eegtext_dataloader(Subset(dataset, indices), batch_size=32)
        batch = next(iter(loader))
        print(f'{name}: samples={len(indices)}, batches={len(loader)}, EEG={tuple(batch["eeg"].shape)}')
        print('characters:', batch['text'])
        print('row numbers:', batch['row_num'].tolist())
