import os
import mne
import pandas as pd
from pathlib import Path




EEG_ROOT = "D:/AI/ChineseEEG"


def load_eeg(novel_name="LittlePrince", filtered="filtered_0.5_30", subject="sub-04", run_num=7):
    eeg_data = []
    for run in range(1, run_num + 1):
        eeg_dir = os.path.join(EEG_ROOT, "derivatives", "preproc", filtered, subject,f"ses-{novel_name}", "eeg")
        prefix=f"{subject}_ses-{novel_name}_task-reading_run-{run:02d}"
        vhdr_path = os.path.join(eeg_dir, f"{prefix}_eeg.vhdr")
        events_path = os.path.join(eeg_dir, f"{prefix}_events.tsv")

        # print(vhdr_path)
        # print(events_path)
        # print("-----------------")

        raw = mne.io.read_raw_brainvision(
            vhdr_path,
            preload=False,
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

        rows_df = events_df[
            events_df["trial_type"].isin(["ROWS", "ROWE"])
            & (events_df["sample"] > first_chapter_sample)
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

        eeg_segments = [
            raw.get_data(
                start=row.start_sample,
                stop=row.end_sample + 1
            )
            for row in segments_df.itertuples()
        ] # numpy array

        # print(f"\n共提取 {len(eeg_segments)} 个 EEG segment")

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
    eeg_data = load_eeg()
    print(len(eeg_data))
    for run_eeg_data in eeg_data:
        # print(run_eeg_data["info"])
        # print(run_eeg_data["sfreq"])
        print(len(run_eeg_data["eeg_segments"]))

       
