"""按外部提供的每行字符数，从 ROWS 起切成 100 点片段，行尾多退少补。"""
from pathlib import Path
import re
import mne
import numpy as np
import pandas as pd

ChineseEEG_ROOT = Path(__file__).resolve().parents[2] / 'data' / 'ChineseEEG'
EEG_PREPROC_DIR = ChineseEEG_ROOT / 'derivatives' / 'preproc'
SAMPLES_PER_CHAR = 100


def _row_intervals(events, context):
    """首个 CH 后严格配对 ROWS/ROWE；sample 使用零基坐标。"""
    if not {'sample', 'trial_type'}.issubset(events.columns):
        raise ValueError(f'{context}: events must contain sample and trial_type')
    samples = np.asarray(pd.to_numeric(events['sample'], errors='raise'), dtype=np.float64)
    if not np.isfinite(samples).all() or (samples < 0).any() or (samples != np.floor(samples)).any():
        raise ValueError(f'{context}: event samples must be nonnegative integers')
    if (np.diff(samples) < 0).any():
        raise ValueError(f'{context}: events are not ordered by sample')
    chapter = events['trial_type'].str.fullmatch(r'CH\d+', na=False).to_numpy()
    if not chapter.any():
        raise ValueError(f'{context}: no CH chapter marker found')
    first_chapter = samples[chapter][0]
    intervals, start = [], None
    for kind, sample in zip(events['trial_type'], samples):
        if sample <= first_chapter:
            continue
        sample = int(sample)
        if kind == 'ROWS':
            if start is not None:
                raise ValueError(f'{context}: repeated ROWS before ROWE at sample {sample}')
            start = sample
        elif kind == 'ROWE':
            if start is None:
                raise ValueError(f'{context}: ROWE without ROWS at sample {sample}')
            # 保留 ROWE 所在采样点，与原加载器闭区间一致。
            intervals.append((start, sample + 1))
            start = None
    if start is not None:
        raise ValueError(f'{context}: ROWS at {start} has no ROWE')
    return intervals


def split_row_eeg(eeg, n_segments):
    """从左向右切成 (N,C,100)，超出目标的尾部丢弃，不足在右侧补零。"""
    if eeg.ndim != 2 or isinstance(n_segments, bool) or not isinstance(n_segments, int) or n_segments <= 0:
        raise ValueError('Expected (channels, time) and a positive integer n_segments')
    target = n_segments * SAMPLES_PER_CHAR
    valid = min(eeg.shape[1], target)
    fixed = np.zeros((eeg.shape[0], target), dtype=np.float32)
    fixed[:, :valid] = eeg[:, :valid]
    # (C,T)->(N,C,100)
    segments = fixed.reshape(eeg.shape[0], n_segments, SAMPLES_PER_CHAR).transpose(1, 0, 2)
    return segments


def load_eeg(novel_name='LittlePrince', filtered='filtered_0.5_30', subject='sub-04',
             run_num=7, *, row_char_counts):
    """返回按 run、行、片段顺序排列的扁平 eeg_words 列表。

    row_char_counts 为 {(run_num, row_num): 高亮字符数}，需包含全部章节和正文行。
    每行从 ROWS 起保留至多 字符数×100 点，不越过 ROWE 或录制末尾；
    多余尾部丢弃，缺失部分右补零，恰好返回对应数量的 (C,100) 片段，单位为 V。
    每个字典包含 eeg、run_num/row_num/char_idx、novel_name 和 subject。
    本模块不读取文本；run_num=None 从 EEG 事件文件发现连续 run。
    """
    if any(isinstance(count, bool) or not isinstance(count, int) or count <= 0
           for count in row_char_counts.values()):
        raise ValueError('row_char_counts must contain positive integer counts')
    folder = EEG_PREPROC_DIR / filtered / subject / f'ses-{novel_name}' / 'eeg'
    if run_num is None:
        pattern = re.compile(re.escape(f'{subject}_ses-{novel_name}_task-reading_run-') + r'(\d+)_events\.tsv')
        runs = sorted(int(match[1]) for path in folder.glob('*_events.tsv')
                      if (match := pattern.fullmatch(path.name)))
        if not runs:
            raise FileNotFoundError(f'No EEG runs found in {folder}')
        if runs != list(range(1, len(runs) + 1)):
            raise ValueError('EEG run numbers must be consecutive from 1')
    else:
        if isinstance(run_num, bool) or not isinstance(run_num, int) or run_num <= 0:
            raise ValueError('run_num must be a positive integer or None')
        runs = range(1, run_num + 1)


    
    run_intervals = []
    for run in runs:
        prefix = f'{subject}_ses-{novel_name}_task-reading_run-{run:02d}'
        # 句子边界
        intervals = _row_intervals(pd.read_csv(folder / f'{prefix}_events.tsv', sep='\t'), prefix)
        if not intervals:
            raise ValueError(f'{prefix}: no ROWS/ROWE intervals found')
        run_intervals.append((run, prefix, intervals))
    expected_keys = {(run, row) for run, _, intervals in run_intervals for row in range(1, len(intervals) + 1)}
    if set(row_char_counts) != expected_keys:
        raise ValueError('EEG/text row mismatch: row_char_counts must match all run/row keys')

    eeg_words = []
    for run, prefix, intervals in run_intervals:
        raw = mne.io.read_raw_brainvision(folder / f'{prefix}_eeg.vhdr', preload=False, verbose=False)
        try:
            # row_num 从 1 开始
            for row_num, (start, stop) in enumerate(intervals, 1):
                if start >= raw.n_times:
                    raise ValueError(f'{prefix}, row {row_num}: ROWS outside recording')
                available = min(stop, raw.n_times) - start # 可用采样点数
                n_segments = row_char_counts[(run, row_num)]
                count = min(available, n_segments * SAMPLES_PER_CHAR) # 取少
                data = raw.get_data(start=start, stop=start + count)
                # (C,T)->(N,C,100)
                segments = split_row_eeg(data, n_segments)
                for char_idx in range(n_segments):
                    eeg_words.append(dict(
                        eeg=segments[char_idx], 
                        row_num=row_num, 
                        char_idx=char_idx,
                        novel_name=novel_name, 
                        subject=subject,
                        run_num=run,
                    ))
        finally:
            raw.close()
    return eeg_words

