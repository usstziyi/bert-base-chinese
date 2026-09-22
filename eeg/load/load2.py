import pandas as pd

events_path = (
    r"D:\AI\ChineseEEG\derivatives\preproc\filtered_0.5_30"
    r"\sub-04\ses-LittlePrince\eeg"
    r"\sub-04_ses-LittlePrince_task-reading_run-01_events.tsv"
)

events_df = pd.read_csv(
    events_path,
    sep="\t"
)

print(events_df.head(20))

print("\n事件映射：")
print(
    events_df[["trial_type", "value"]]
    .drop_duplicates()
    .sort_values("value")
)