"""快速检查 EEG–文本数据集与 DataLoader 的输出。"""

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

    loader = create_eegtext_dataloader(
        dataset,
        batch_size=32,
        shuffle=False,
    )

    batch = next(iter(loader))

    print("\n========== One batch ==========")
    print("EEG batch shape:", batch["eeg"].shape)
    for i, text in enumerate(batch["text"]):
        print(f"{i}[{len(text)}]: {text}")
