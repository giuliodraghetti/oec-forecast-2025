import pandas as pd, numpy as np, pathlib
P = pathlib.Path("data/processed")
df = pd.read_parquet(P/"trade_monthly.parquet")

df = df.sort_values(["Reporter","Partner","HS4","TradeFlow","date"])
for k in range(1,13):
    df[f"lag{k}"] = df.groupby(["Reporter","Partner","HS4","TradeFlow"])["Value"].shift(k)
df["yoy_growth"] = np.log1p(df["Value"]) - np.log1p(df["lag12"])
df["month"] = df.date.dt.month.astype("category")
df.dropna(inplace=True)
df.to_parquet(P/"features.parquet", index=False)
print("features saved")
