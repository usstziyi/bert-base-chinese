import numpy as np
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
# 测试 EEG 数据单位
# ============================================================
# 取第一个 EEG 通道前 10 个采样点
data_v = raw.get_data(
    picks=[0],
    start=0,
    stop=10
)[0]


# 转成 µV
data_uv = data_v * 1e6

print("========== EEG unit test ==========")

print("\n原始 MNE 数据（单位 V）:")
print(data_v)

print("\n转换成 µV:")
print(data_uv)

print("\n前 10 个点逐个对照:")
for i, (v, uv) in enumerate(zip(data_v, data_uv)):
    print(
        f"{i:2d}: "
        f"{v:.9f} V"
        f"  =  {uv:.3f} µV"
    )