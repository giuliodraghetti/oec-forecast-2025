import zipfile, pathlib, pandas as pd

RAW = pathlib.Path("data/raw")
OUT = pathlib.Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def read_zip(z):
    with zipfile.ZipFile(z) as zf:
        csv = zf.namelist()[0]
        return pd.read_csv(zf.open(csv), dtype={"hs4": str})

df = pd.concat([read_zip(z) for z in RAW.glob("*.zip")], ignore_index=True)

df = df.rename(columns={
    "state": "Reporter",
    "partner_iso3": "Partner",
    "hs4": "HS4",
    "trade_flow": "TradeFlow",
    "value_usd": "Value"
})
df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
df.to_parquet(OUT / "trade_monthly.parquet", index=False)
print("Saved", OUT / "trade_monthly.parquet")
