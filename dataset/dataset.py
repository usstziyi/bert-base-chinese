from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

if __package__:
    from .load_eeg import load_eeg
    from .load_text import load_text
else:
    from load_eeg import load_eeg
    from load_text import load_text



class ChineseEEGDataset(Dataset):

    def __init__(
        self,
        novel_name: str = "LittlePrince",
        filtered: str = "filtered_0.5_30",
        subject: str = "sub-04",
        run_num: Optional[int] = 7,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()


        self.novel_name = novel_name
        self.filtered = filtered
        self.subject = subject
        self.run_num = run_num
        self.dtype = dtype

        # ------------------------------------------------------------
        # 1. Call the two existing loader APIs
        # ------------------------------------------------------------
        self.eeg_data = load_eeg(
            novel_name=novel_name,
            filtered=filtered,
            subject=subject,
            run_num=run_num,
        )

        self.text_data = load_text(
            novel_name=novel_name,
            run_num=run_num,
        )

        # ------------------------------------------------------------
        # 2. Basic run-level checks
        # ------------------------------------------------------------
        if len(self.eeg_data) != len(self.text_data):
            raise ValueError(
                "Number of EEG runs and text runs does not match: "
                f"EEG={len(self.eeg_data)}, text={len(self.text_data)}"
            )

        self.samples: List[Dict[str, int]] = []

        for run_idx, (run_eeg, run_texts) in enumerate(
            zip(self.eeg_data, self.text_data)
        ):
            eeg_segments = run_eeg["eeg_segments"]

            if len(eeg_segments) != len(run_texts):
                raise ValueError(
                    f"Run {run_idx + 1:02d}: EEG/text count mismatch: "
                    f"EEG segments={len(eeg_segments)}, "
                    f"text sentences={len(run_texts)}"
                )

            for segment_idx in range(len(eeg_segments)):
                self.samples.append(
                    {
                        "run_idx": run_idx,
                        "segment_idx": segment_idx,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_index = self.samples[index]

        run_idx = sample_index["run_idx"]
        segment_idx = sample_index["segment_idx"]

        run_eeg = self.eeg_data[run_idx]

        # numpy.ndarray: (n_channels, n_times)
        eeg_segment = run_eeg["eeg_segments"][segment_idx]

        # Convert to Tensor without changing the channel/time ordering.
        eeg = torch.as_tensor(
            eeg_segment,
            dtype=self.dtype,
        )

        text = self.text_data[run_idx][segment_idx]

        return {
            "eeg": eeg,
            "text": text,
            "novel_name": self.novel_name,
            "run_idx": run_idx,
            "run_num": run_idx + 1,
            "segment_idx": segment_idx,
            "sfreq": float(run_eeg["sfreq"]),
        }

# 这个函数是给 PyTorch DataLoader 用的 collate 函数（批处理整理函数），
# 核心任务是：把 batch 里长度不一的 EEG 片段统一填充（pad）/截断到固定长度
# max_len（默认 1500），输出形状恒为 (B, C, max_len)。
def eeg_text_collate_fn(
    batch: List[Dict[str, Any]],
    max_len: int = 1500,
) -> Dict[str, Any]:


    if len(batch) == 0:
        raise ValueError("batch must not be empty")

    # All samples should have the same number of EEG channels.
    n_channels = batch[0]["eeg"].shape[0]

    for i, sample in enumerate(batch):
        eeg = sample["eeg"]

        if eeg.ndim != 2:
            raise ValueError(
                f"Sample {i} EEG must have shape (C, T), "
                f"but got {tuple(eeg.shape)}"
            )

        if eeg.shape[0] != n_channels:
            raise ValueError(
                "All EEG samples in one batch must have the same "
                f"number of channels, but sample 0 has {n_channels} "
                f"and sample {i} has {eeg.shape[0]}"
            )

    # 统一到固定长度：超出 max_len 的截断（保留前 max_len 个时间点），
    # 不足 max_len 的在时间维末尾补 0。全程保持 (C, T) 布局，无需转置。
    eeg_list: List[torch.Tensor] = []
    length_list: List[int] = []

    for sample in batch:
        # 截断
        eeg = sample["eeg"][:, :max_len]  # (C, T_i'), T_i' <= max_len

        # 在 pad 之前记录有效长度 = min(原始长度, max_len)，
        # pad 之后 shape[1] 恒为 max_len，就丢失真实长度了。
        length_list.append(eeg.shape[1])

        pad_t = max_len - eeg.shape[1]
        if pad_t > 0:
            eeg = F.pad(eeg, (0, pad_t))  # 在最后一维（时间维）右侧补 0

        # pad之后的 EEG 片段，形状为 (C, max_len)，
        eeg_list.append(eeg)

    eeg = torch.stack(eeg_list, dim=0)  # (B, C, max_len)

    lengths = torch.tensor(length_list, dtype=torch.long)  # (B,)

    attention_mask = (
        torch.arange(max_len).unsqueeze(0)  # (1, max_len)
        < lengths.unsqueeze(1)              # (B, 1)
    )                                       # (B, max_len) bool; 填充位为 False

    return {
        "eeg": eeg,
        "eeg_lengths": lengths, # EEG 有效长度（min(原始长度, max_len)）
        "attention_mask": attention_mask,
        "text": [sample["text"] for sample in batch],
        "novel_name": [sample["novel_name"] for sample in batch],
        "run_idx": torch.tensor(
            [sample["run_idx"] for sample in batch],
            dtype=torch.long,
        ),
        "run_num": torch.tensor(
            [sample["run_num"] for sample in batch],
            dtype=torch.long,
        ),
        "segment_idx": torch.tensor(
            [sample["segment_idx"] for sample in batch],
            dtype=torch.long,
        ),
        "sfreq": torch.tensor(
            [sample["sfreq"] for sample in batch],
            dtype=torch.float32,
        ),
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = ChineseEEGDataset(
        novel_name="LittlePrince",
        subject="sub-04",
        filtered="filtered_0.5_30",
    )

    print(f"Dataset size: {len(dataset)}")

    # sample = dataset[1]

    # print("\n========== Single sample ==========")
    # print("EEG shape:", sample["eeg"].shape)
    # print("Text:", sample["text"])
    # print("Run:", sample["run_num"])
    # print("Segment:", sample["segment_idx"])
    # print("Sampling rate:", sample["sfreq"])

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=eeg_text_collate_fn,
    )

    batch = next(iter(loader))

    print("\n========== One batch ==========")
    print("EEG batch shape:", batch["eeg"].shape)
    for i, text in enumerate(batch["text"]):
        print(f"{i}[{len(text)}]: {text}")

    # batch = next(iter(loader))

    # device = torch.device(
    #     "cuda" if torch.cuda.is_available() else "cpu"
    # )

    # eeg = batch["eeg"].to(device)
    # eeg_mask = batch["attention_mask"].to(device)

    # encoder = EEGEncoder(
    #     eeg_channels=eeg.shape[1],
    #     input_samples=eeg.shape[-1],
    #     projection_dim=768,
    # ).to(device)

    # eeg_features = encoder(
    #     eeg=eeg,
    #     attention_mask=eeg_mask,
    # )

    # print(eeg.shape)
    # print(eeg_features.shape)
