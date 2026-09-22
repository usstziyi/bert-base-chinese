import numpy as np
import matplotlib.pyplot as plt
import mne

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
# 11. 设置 EEG 电极位置
# ============================================================

# 如果当前 raw 没有 montage，就使用匹配的标准电极坐标
# 通道名为 E1~E128（EGI 128 导系统），需使用 GSN-HydroCel-128
if raw.get_montage() is None:
    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")

    raw.set_montage(
        montage,
        on_missing="ignore"
    )

print("montage:", raw.get_montage())


# ============================================================
# 12. 提取前 10 秒 EEG
# ============================================================

sfreq = raw.info["sfreq"]

# 只选择 EEG 通道
picks = mne.pick_types(
    raw.info,
    eeg=True,
    meg=False,
    eog=False,
    stim=False,
    exclude="bads"
)

# 构建 EEG-only 的 Info，供 plot_topomap 使用
# （旧版 MNE 的 plot_topomap 不支持 picks 参数）
eeg_info = mne.pick_info(raw.info, picks)

eeg_data = raw.get_data(
    picks=picks,
    start=0,
    stop=int(10 * sfreq)
)

print("EEG data shape:", eeg_data.shape)
# shape:
# (n_channels, 10 * sfreq)

# ============================================================
# 13. 每 1 秒计算一次各通道平均电位
# ============================================================
window_duration = 1.0
n_windows = 10

window_samples = int(window_duration * sfreq)

topomap_data = []

for i in range(n_windows):

    start_sample = i * window_samples
    end_sample = (i + 1) * window_samples

    segment = eeg_data[
        :,
        start_sample:end_sample
    ]

    # 对这一秒内的时间维求平均
    # shape:
    # (n_channels, n_samples)
    # ->
    # (n_channels,)
    mean_voltage = segment.mean(axis=1)

    topomap_data.append(mean_voltage)

topomap_data = np.array(topomap_data)

print("Topomap data shape:", topomap_data.shape)
# (10, n_channels)


# ============================================================
# 14. 统一 10 张图的颜色范围
# ============================================================

# 转成 µV，更方便观察
topomap_data_uv = topomap_data * 1e6

# 使用所有 10 秒的最大绝对值，保证所有图颜色尺度一致
vmax = np.max(np.abs(topomap_data_uv))
vmin = -vmax

print(f"color range: {vmin:.3f} ~ {vmax:.3f} µV")


# ============================================================
# 15. 横向绘制 10 张 EEG Topomap
# ============================================================

fig, axes = plt.subplots(
    1,
    10,
    figsize=(24, 3)
)

im = None

for i, ax in enumerate(axes):

    im, _ = mne.viz.plot_topomap(
        topomap_data_uv[i],
        eeg_info,
        axes=ax,
        show=False,
        vlim=(vmin, vmax),
        cmap="RdBu_r",
        contours=6,
        sensors=True
    )

    ax.set_title(
        f"{i}-{i+1} s",
        fontsize=11
    )


# ============================================================
# 16. 添加公共 colorbar
# ============================================================

cbar = fig.colorbar(
    im,
    ax=axes,
    orientation="vertical",
    fraction=0.015,
    pad=0.02
)

cbar.set_label(
    "Mean EEG amplitude (µV)"
)

fig.suptitle(
    "EEG Topography — First 10 Seconds",
    fontsize=16
)

plt.show()