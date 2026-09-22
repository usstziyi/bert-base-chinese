import mne
from pathlib import Path
import pandas as pd


# ============================================================
# 1. EEG 文件路径
# ============================================================

eeg_dir = Path(
    r"D:\AI\ChineseEEG\derivatives\preproc\filtered_0.5_30"
    r"\sub-04\ses-LittlePrince\eeg"
)

vhdr_path = (
    eeg_dir
    / "sub-04_ses-LittlePrince_task-reading_run-01_eeg.vhdr"
)


# ============================================================
# 2. 加载 BrainVision EEG
# ============================================================

raw = mne.io.read_raw_brainvision(
    vhdr_path,
    preload=False,
    verbose=False
)


# ============================================================
# 3. 查看基本信息
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

print("\n========== Annotations ==========")
annotations = raw.annotations
ann_df = pd.DataFrame({
    "onset": annotations.onset,
    "duration": annotations.duration,
    "description": annotations.description,
})

print(ann_df.head(20))

print("\n========== Annotations description ==========")
import numpy as np
desc = raw.annotations.description
unique, counts = np.unique(desc, return_counts=True)
for u, c in zip(unique, counts):
    print(u,":", c)



print("\n========== Events ==========")
events, event_id = mne.events_from_annotations(raw)

print("\nevent_id:")
for name, code in event_id.items():
    print(f"{name} -> {code}")