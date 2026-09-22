import numpy as np
import matplotlib.pyplot as plt
import mne

import mne
import pandas as pd
from pathlib import Path
from matplotlib.widgets import Slider

# 中文标签需要指定字体，否则会显示成方块
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

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
# 只取 EEG
raw_eeg = raw.copy().pick("eeg")

# 前 10 秒
raw_10s = raw_eeg.copy().crop(
    tmin=10,
    tmax=11
)


# Raw -> Evoked
evoked = mne.EvokedArray(
    raw_10s.get_data(),
    raw_10s.info,
    tmin=0
)

data_uv = evoked.data * 1e6
times_ms = evoked.times * 1000

t_init = 0.0  # 初始时刻（秒）：刺激 onset

fig = plt.figure(figsize=(8, 7))
# 创建一个 3 行 1 列 的网格布局，第 1 行是地形图，第 2 行是波形，第 3 行是滑块
gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 1.8, 0.35], hspace=0.45)

ax_topo = fig.add_subplot(gs[0])
ax_wave = fig.add_subplot(gs[1])
ax_slider = fig.add_subplot(gs[2])

# --- 第 2 层：波形 + 游标 ---
ax_wave.plot(times_ms, data_uv.T, color='k', lw=0.4, alpha=0.5)
ax_wave.axvline(0, color='gray', ls='--', lw=1)              # 刺激 onset 参考线
cursor = ax_wave.axvline(t_init * 1000, color='r', lw=1.5)   # 当前时刻游标
ax_wave.set_xlim(times_ms[0], times_ms[-1])
ax_wave.set_xlabel('Time (ms)')
ax_wave.set_ylabel('Amplitude (uV)')
ax_wave.set_title('所有EEG通道波形（灰虚线=刺激onset，红线=当前时刻）', fontsize=10)


def _update(t):
    """滑块回调：把游标移到时刻 t（秒），并重画该时刻的地形图。"""
    idx = int(np.argmin(np.abs(evoked.times - t)))
    t_actual = evoked.times[idx]

    # 游标横坐标单位是 ms，和上面波形的横轴保持一致
    cursor.set_xdata([t_actual * 1000, t_actual * 1000])

    # 每个时刻都是重建 artists，最省事、也不会残留上一个时刻的等值线
    # 用 evoked.plot_topomap 的两个好处：
    #   1) vlim 按 uV 处理，MNE 内部按 scalings 从 V 自动换算，不用自己 *1e6
    #   2) 自动把时刻写进 axes 标题（如 "120 ms"），不用自己 set_title
    # 注意：一旦传了 axes，colorbar 就必须是 False，
    #       否则 MNE 要求 axes 个数 = 时刻个数 + 1 个 colorbar 位
    ax_topo.clear()
    evoked.plot_topomap(
        times=[t_actual],
        ch_type='eeg',
        cmap='RdBu_r',
        contours=6,
        # vlim=(-10, 10),      # 固定色标范围，单位 uV，拖动时各时刻才能直接对比
        size=1,
        time_unit='ms',
        time_format='%.0f ms',   # MNE 默认是 "%01d ms"（截断），这里改成四舍五入
        axes=ax_topo,
        colorbar=False,
        show=False,
    )
    # 不用再 fig.canvas.draw_idle()：
    # 传了 axes 时 evoked.plot_topomap 结尾自己会 fig.canvas.draw()（MNE 源码里那行），
    # 而游标在它之前就移好了，会被同一次重绘一起带出去


# 先画初始时刻，再从它在地形图 axes 上留下的 image 建 colorbar；
# 后续更新只是 clear + 重画 ax_topo，colorbar 是独立 axes，不受影响
_update(t_init)
cbar = fig.colorbar(ax_topo.images[0], ax=ax_topo, fraction=0.046, pad=0.04)
cbar.set_label('Amplitude (uV)')

# --- 第 3 层：时间滑块 ---
slider = Slider(
    ax_slider,
    'Time (s)',
    evoked.times[0],
    evoked.times[-1],
    valinit=t_init,
    valfmt='%1.3f',
)
slider.on_changed(_update)

fig.suptitle('EEG地形图（拖动滑块联动查看）', fontsize=13)

# 交互窗口靠 plt.show() 保持，关掉窗口后再保存图片
plt.show()