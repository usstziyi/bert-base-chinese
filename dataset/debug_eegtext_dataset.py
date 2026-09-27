"""快速检查 EEG–文本数据集与 DataLoader 的输出。"""

from torch.utils.data import Subset

if __package__:
    from .eegtext_dataloader import create_eegtext_dataloader
    from .eegtext_dataset import ChineseEEGDataset
else:
    from eegtext_dataloader import create_eegtext_dataloader
    from eegtext_dataset import ChineseEEGDataset


if __name__ == "__main__":
    dataset = ChineseEEGDataset(
        novel_name="LittlePrince",
        subject="sub-04",
        filtered="filtered_0.5_30",
    )

    print(f"Dataset size: {len(dataset)}")

    # run 1~6 做训练集，run 7 做验证集（与 train/train.py 的默认切分一致）。
    train_indices = [i for i, s in enumerate(dataset.samples) if s["run_idx"] + 1 != 7]
    val_indices = [i for i, s in enumerate(dataset.samples) if s["run_idx"] + 1 == 7]

    train_loader = create_eegtext_dataloader(
        Subset(dataset, train_indices),
        batch_size=32,
        # shuffle=True,
    )
    val_loader = create_eegtext_dataloader(
        Subset(dataset, val_indices),
        batch_size=32,
        # shuffle=False,
    )

    print(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    for split, loader in (("train", train_loader), ("val", val_loader)):
        for i, batch in enumerate(loader):
            print(f"{split} batch {i}: eeg={tuple(batch['eeg'].shape)}, text={len(batch['text'])}")


    # batch 是 collate 后的 dict；单样本原始形状由 eeg_lengths 还原为 (C, T_i)。
    for batch in train_loader:
        n_channels = batch["eeg"].shape[1]
        print(batch["eeg"].shape)
        for i, length in enumerate(batch["eeg_lengths"]):
            print(f"train sample {i}: eeg=({n_channels}, {int(length)}) -> {batch["eeg"].shape[-1]}")
        break




