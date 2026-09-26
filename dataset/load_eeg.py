import os
import mne
import pandas as pd
from pathlib import Path

ChineseEEG_ROOT = "data/ChineseEEG"
EEG_PREPROC_DIR = os.path.join(ChineseEEG_ROOT, "derivatives", "preproc")


def load_eeg(novel_name="LittlePrince", filtered="filtered_0.5_30", subject="sub-04", run_num=7):
    eeg_data = []
    for run in range(1, run_num + 1):
        eeg_dir = os.path.join(EEG_PREPROC_DIR, filtered, subject,f"ses-{novel_name}", "eeg")
        prefix=f"{subject}_ses-{novel_name}_task-reading_run-{run:02d}"
        vhdr_path = os.path.join(eeg_dir, f"{prefix}_eeg.vhdr")
        events_path = os.path.join(eeg_dir, f"{prefix}_events.tsv")

        # print(vhdr_path)
        # print(events_path)
        # print("-----------------")

        raw = mne.io.read_raw_brainvision(
            vhdr_path,
            preload=True,
            verbose=False
        )

        events_df = pd.read_csv(
            events_path,
            sep="\t"
        )

        sfreq = raw.info["sfreq"]

        chapter_df = events_df[
            events_df["trial_type"].str.match(r"^CH\d+$", na=False)
        ]

        if chapter_df.empty:
            raise ValueError("当前 run 中没有找到 CHxx")

        first_chapter_sample = int(
            chapter_df.iloc[0]["sample"]
        )

        # 筛选出「行起始事件（ROWS）」或「行结束事件（ROWE）」，且事件发生时间晚于首个章节标记的事件行
        # 重置索引以保证后续遍历 segments 时的下标连续性
        rows_df = events_df[
            events_df["trial_type"].isin(["ROWS", "ROWE"])  # 事件类型限定为行起止标记
            & (events_df["sample"] > first_chapter_sample)  # 仅保留章节开始后的事件
        ].reset_index(drop=True)

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

        segments_df["n_samples"] = (
            segments_df["end_sample"]
            - segments_df["start_sample"]
            + 1
        )

        segments_df["duration"] = (
            segments_df["n_samples"] / sfreq
        )

        # 预加载后整个 run 已在内存，一次性取出，再按采样点区间做内存切片
        raw_data = raw.get_data()  # (n_channels, n_samples_all)

        eeg_segments = [
            raw_data[:, row.start_sample:row.end_sample + 1]
            for row in segments_df.itertuples()
        ]  # list of (n_channels, n_samples)、各行长度不等


        # print("\n========== 检查 ROWS / ROWE / TEXT 数量 ==========")
        # n_rows = (rows_df["trial_type"] == "ROWS").sum()
        # n_rowe = (rows_df["trial_type"] == "ROWE").sum()
        # n_segments = len(segments_df)

        # assert n_rows == n_rowe, (
        #     f"ROWS/ROWE 数量不一致: ROWS={n_rows}, ROWE={n_rowe}"
        # )

        # assert n_segments == n_rows, (
        #     f"Segment 数量异常: segments={n_segments}, ROWS={n_rows}"
        # )

        # print(
        #     f"检查通过: "
        #     f"ROWS={n_rows}, "
        #     f"ROWE={n_rowe}, "
        #     f"TEXT={n_segments}"
        # )

        run_eeg_data = {}
        run_eeg_data["eeg_segments"] = eeg_segments
        run_eeg_data["info"] = raw.info
        run_eeg_data["sfreq"] = sfreq
        eeg_data.append(run_eeg_data)
    
    return eeg_data

   


if __name__ == "__main__":
    print("Loading EEG data for LittlePrince...")
    eeg_data = load_eeg(novel_name="LittlePrince", run_num=7, subject="sub-04", filtered="filtered_0.5_30")
    print(len(eeg_data))
    for run_eeg_data in eeg_data:
        # print(run_eeg_data["info"])
        # print(run_eeg_data["sfreq"])
        print(len(run_eeg_data["eeg_segments"]))

    print("Loading EEG data for GarnettDream...")
    eeg_data = load_eeg(novel_name="GarnettDream", run_num=18, subject="sub-04", filtered="filtered_0.5_30")
    print(len(eeg_data))
    for run_eeg_data in eeg_data:
        # print(run_eeg_data["info"])
        # print(run_eeg_data["sfreq"])
        print(len(run_eeg_data["eeg_segments"]))

       
