from typing import Any, Dict, List, Optional

import torch
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
        eeg_length = eeg.shape[1]

        text = self.text_data[run_idx][segment_idx]

        return {
            "eeg": eeg,
            "eeg_length": eeg_length,
            "text": text,
            "novel_name": self.novel_name,
            "run_idx": run_idx,
            "run_num": run_idx + 1,
            "segment_idx": segment_idx,
            "sfreq": float(run_eeg["sfreq"]),
        }


def main():
    print("hello,world")

if __name__ == "__main__":
    main()