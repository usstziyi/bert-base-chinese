import mne
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# 1. 路径
# ============================================================
eeg_dir = Path(
    r"D:\AI\ChineseEEG\derivatives\preproc\filtered_0.5_30"
    r"\sub-04\ses-LittlePrince\eeg"
)

vhdr_path = eeg_dir / "sub-04_ses-LittlePrince_task-reading_run-01_eeg.vhdr"
events_path = eeg_dir / "sub-04_ses-LittlePrince_task-reading_run-01_events.tsv"

# ============================================================
# 2. 加载 EEG
# ============================================================
raw = mne.io.read_raw_brainvision(
    vhdr_path,
    preload=False,
    verbose=False
)

# ============================================================
# 3. EEG 基本信息
# ============================================================
print("\n========== Raw ==========")
print(raw)

print("\n========== Info ==========")
print(raw.info)

print("\n========== Sampling rate ==========")
print(raw.info["sfreq"], "Hz")

print("\n========== Number of channels ==========")
print(len(raw.ch_names))

print("\n========== Channel names ==========")
print(raw.ch_names)

print("\n========== Number of samples ==========")
print(raw.n_times)

print("\n========== Duration ==========")
print(raw.times[-1], "seconds")

# ============================================================
# 4. Annotations
# ============================================================
print("\n========== Annotations ==========")
print(raw.annotations)

print("\n========== Annotations description ==========")
desc = raw.annotations.description
unique, counts = np.unique(desc, return_counts=True)

for u, c in zip(unique, counts):
    print(f"{u}: {c}")

# ============================================================
# 5. MNE Events
# ============================================================
print("\n========== Event ID ==========")
events, event_id = mne.events_from_annotations(raw)
for name, code in event_id.items():
    print(f"{name} -> {code}")

# 第一列：事件发生的 采样点索引
# 第二列：前一个事件的 ID（基本不用，几乎总是 0，是历史遗留）
# 第三列：事件的 编码值 ，对应event_id字典里的 code
for event in events[:10]:
    print(event)



# ============================================================
# 6. 加载 events.tsv
# ============================================================
events_df = pd.read_csv(
    events_path,
    sep="\t"
)

print("\n========== events.tsv ==========")
print(events_df.head(10))

print("\n========== 事件映射 ==========")
print(
    events_df[["trial_type", "value"]]   # 1. 只取这两列，得到一个新 DataFrame
        .drop_duplicates()               # 2. 去掉重复的行，只留唯一的 (trial_type, value) 组合
        .sort_values("value")            # 3. 按 value 列升序排序
)

# ============================================================
# 7. 找到当前 run 的第一个章节事件 CHxx
# ============================================================
chapter_df = events_df[
    events_df["trial_type"].str.match(r"^CH\d+$", na=False)
]

if chapter_df.empty:
    raise ValueError("当前 run 中没有找到 CHxx 章节事件")

first_chapter_sample = int(chapter_df.iloc[0]["sample"])
first_chapter_name = chapter_df.iloc[0]["trial_type"]

print(
    f"当前 run 第一个章节事件: {first_chapter_name}, "
    f"sample = {first_chapter_sample}"
)

# ============================================================
# 8. 只保留第一个章节之后的 ROWS / ROWE
# ============================================================
rows_df = events_df[
    events_df["trial_type"].isin(["ROWS", "ROWE"])
    & (events_df["sample"] > first_chapter_sample)
].reset_index(drop=True)


print("\n========== ROWS / ROWE ==========")
print(rows_df.head(20))

# ============================================================
# 9. 配对 ROWS -> ROWE
# ============================================================
segments = []
current_start_sample = None

for _, row in rows_df.iterrows():
    event_type = row["trial_type"]
    sample = int(row["sample"])

    if event_type == "ROWS":
        current_start_sample = sample

    elif event_type == "ROWE" and current_start_sample is not None:
        segments.append({
            "start_sample": current_start_sample,
            "end_sample": sample
        })
        current_start_sample = None

# ============================================================
# 10. 提取 ROWS -> ROWE 之间的 EEG
# ============================================================
eeg_segments = []
sfreq = raw.info["sfreq"]

for seg in segments:
    start_sample = seg["start_sample"]
    end_sample = seg["end_sample"]

    data = raw.get_data(
        start=start_sample,
        stop=end_sample + 1
    )

    eeg_segments.append(data)

# ============================================================
# 11. 查看结果
# ============================================================
print(f"\n共提取 {len(eeg_segments)} 个 EEG segment")


for i, data in enumerate(eeg_segments[:10]):
    duration = data.shape[1] / sfreq

    print(
        f"Segment {i}: "
        f"shape={data.shape}, "
        f"duration={duration:.3f}s"
    )