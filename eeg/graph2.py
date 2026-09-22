import mne
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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
# 11. 设置 montage
# ============================================================
montage = mne.channels.make_standard_montage(
    "GSN-HydroCel-128"
)

raw.set_montage(montage)

print("montage:", raw.get_montage())
# ============================================================
# 12. 计算并绘制 PSD
# ============================================================
spectrum = raw.compute_psd(
    method="welch",
    fmin=0.5,
    fmax=30,
    picks="eeg"
)

psd_data, freqs = spectrum.get_data(
    return_freqs=True
)

print("\n========== PSD ==========")
print(spectrum)
print("PSD shape:", psd_data.shape)
print("Frequency shape:", freqs.shape)
print(
    f"Frequency range: "
    f"{freqs[0]:.2f} ~ {freqs[-1]:.2f} Hz"
)


spectrum.plot(spatial_colors=True)
# 阻塞程序，直到手动关闭窗口
plt.show(block=True)

# ============================================================
# 11. 计算平均PSD并绘制
# ============================================================
spectrum.plot(average=True,color="red")
plt.show(block=True)


# ============================================================
# 12. 频段功率头皮地形图（PSD topomap）
# 颜色表示该频段在该头皮位置上的功率大小
# 说明在你这个 run 的整个记录期间，8–13 Hz 功率在这些区域相对更明显
# 意思是在那个电极附近，Delta 频段的 PSD 较高
# ============================================================


spectrum = raw.compute_psd(
    method="welch",
    fmin=0.5,
    fmax=30,
    picks="eeg"
)

bands = {
    'Delta (1-4 Hz)':   (1, 4),
    'Theta (4-8 Hz)':   (4, 8),
    'Alpha (8-13 Hz)':  (8, 13),
    'Beta (13-30 Hz)':  (13, 30),
}
spectrum.plot_topomap(
    bands=bands,
    colorbar=True,
    size=2,
    show=True
)

plt.show(block=True)

