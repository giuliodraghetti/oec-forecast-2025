import pandas as pd, csv, pathlib
df = pd.read_parquet("data/processed/forecasts_oct25.parquet")
df.to_csv("submission_oct25.csv", index=False, quoting=csv.QUOTE_ALL)
print("submission_oct25.csv written")
