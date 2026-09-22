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

sfreq = raw.info["sfreq"]

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
# 6. 计算 segment 信息
# ============================================================
segments_df["n_samples"] = (
    segments_df["end_sample"]
    - segments_df["start_sample"]
    + 1
)

segments_df["duration"] = (
    segments_df["n_samples"] / sfreq
)

# ============================================================
# 7. 提取 EEG
# ============================================================
eeg_segments = [
    raw.get_data(
        start=row.start_sample,
        stop=row.end_sample + 1
    )
    for row in segments_df.itertuples()
]

print(f"\n共提取 {len(eeg_segments)} 个 EEG segment")
print(segments_df.head(10))

for i, eeg in enumerate(eeg_segments[:10]):
    print(
        f"{i}: "
        f"shape={eeg.shape}, "
        f"duration={eeg.shape[1] / sfreq:.3f}s"
    )

# ============================================================
# 8. 创建 TEXT Annotations
# ============================================================
segment_annotations = mne.Annotations(
    onset=segments_df["start_sample"].to_numpy() / sfreq,

    # annotation 表示 ROWS 到 ROWE 的时间间隔
    duration=(
        segments_df["end_sample"].to_numpy()
        - segments_df["start_sample"].to_numpy()
    ) / sfreq,

    description=["TEXT"] * len(segments_df),

    orig_time=raw.annotations.orig_time
)

print("\n========== Segment Annotations ==========")
print(segment_annotations)

# # ============================================================
# # 9. 保留原始 annotations，并添加 TEXT annotations
# # ============================================================
# raw_with_segments = raw.copy()

# raw_with_segments.set_annotations(
#     raw.annotations + segment_annotations
# )

# print("\n========== Raw + TEXT Annotations ==========")
# print(raw_with_segments.annotations)

# annotations = raw_with_segments.annotations

# ann_df = pd.DataFrame({
#     "onset": annotations.onset,
#     "duration": annotations.duration,
#     "description": annotations.description,
# })

# print("\n========== Annotations ==========")
# print(ann_df.head(20))

# print("\n========== 各类型标注数量 ==========")
# print(ann_df["description"].value_counts())


print("\n========== 检查 ROWS / ROWE / TEXT 数量 ==========")
n_rows = (rows_df["trial_type"] == "ROWS").sum()
n_rowe = (rows_df["trial_type"] == "ROWE").sum()
n_segments = len(segments_df)

assert n_rows == n_rowe, (
    f"ROWS/ROWE 数量不一致: ROWS={n_rows}, ROWE={n_rowe}"
)

assert n_segments == n_rows, (
    f"Segment 数量异常: segments={n_segments}, ROWS={n_rows}"
)

print(
    f"检查通过: "
    f"ROWS={n_rows}, "
    f"ROWE={n_rowe}, "
    f"TEXT={n_segments}"
)

# ============================================================
# 10. 可视化整个 run + TEXT 区间
# ============================================================
start_time = 100     # 起点（秒）
end_time = 110      # 终点（秒），None 表示到数据结尾，不能超过数据时长

raw_with_segments = raw.copy()

if start_time > 0 or end_time is not None:
    raw_with_segments.crop(
        tmin=start_time,
        tmax=end_time
    )

raw_with_segments.plot(
    duration=20,       # 每个窗口显示 20 秒
    n_channels=10,     # 同时显示 30 个通道
    scalings={"eeg": 100e-6},  # 纵轴每格 100 µV
    block=True         # 保持窗口打开
)