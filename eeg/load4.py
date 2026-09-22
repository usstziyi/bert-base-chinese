import mne
import pandas as pd
from pathlib import Path

# ============================================================
# 1. 路径
# ============================================================
eeg_dir = Path(
    r"D:\AI\ChineseEEG\derivatives\preproc\filtered_0.5_30"
    r"\sub-04\ses-LittlePrince\eeg"
)

prefix = "sub-04_ses-LittlePrince_task-reading_run-01"

vhdr_path = eeg_dir / f"{prefix}_eeg.vhdr"
events_path = eeg_dir / f"{prefix}_events.tsv"

# ============================================================
# 2. 加载 EEG 和 events.tsv
# ============================================================
raw = mne.io.read_raw_brainvision(
    vhdr_path,
    preload=False,
    verbose=False
)

events_df = pd.read_csv(
    events_path,
    sep="\t"
)

print(raw)
print(f"采样率: {raw.info['sfreq']} Hz")
print(f"通道数: {len(raw.ch_names)}")
print(f"时长: {raw.times[-1]:.2f} s")

# ============================================================
# 3. 找到当前 run 第一个 CHxx
# ============================================================
chapter_df = events_df[
    events_df["trial_type"].str.match(r"^CH\d+$", na=False)
]

if chapter_df.empty:
    raise ValueError("当前 run 中没有找到 CHxx")

first_chapter_sample = int(
    chapter_df.iloc[0]["sample"]
)

# ============================================================
# 4. 保留第一个章节后的 ROWS / ROWE
# ============================================================
rows_df = events_df[
    events_df["trial_type"].isin(["ROWS", "ROWE"])
    & (events_df["sample"] > first_chapter_sample)
].reset_index(drop=True)

# ============================================================
# 5. ROWS -> ROWE 配对
# ============================================================
segments = []
start_sample = None

for _, row in rows_df.iterrows():
    event_type = row["trial_type"]
    sample = int(row["sample"])

    if event_type == "ROWS":
        start_sample = sample

    elif event_type == "ROWE" and start_sample is not None:
        segments.append({
            "start_sample": start_sample,
            "end_sample": sample
        })

        start_sample = None

segments_df = pd.DataFrame(segments)

# ============================================================
# 6. 提取 EEG
# ============================================================
eeg_segments = [
    raw.get_data(
        start=row.start_sample,
        stop=row.end_sample + 1
    )
    for row in segments_df.itertuples()
]

# ============================================================
# 7. 查看结果
# ============================================================
sfreq = raw.info["sfreq"]

segments_df["n_samples"] = (
    segments_df["end_sample"]
    - segments_df["start_sample"]
    + 1
)

segments_df["duration"] = (
    segments_df["n_samples"] / sfreq
)

print(f"\n共提取 {len(eeg_segments)} 个 EEG segment")
print(segments_df.head(10))

for i, eeg in enumerate(eeg_segments[:10]):
    print(
        f"{i}: "
        f"shape={eeg.shape}, "
        f"duration={eeg.shape[1] / sfreq:.3f}s"
    )