import statsmodels.api as sm, pandas as pd, pathlib, pickle, tqdm
P = pathlib.Path("data/processed")
M = pathlib.Path("models/sarimax"); M.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(P/"features.parquet")

for key, grp in tqdm.tqdm(df.groupby(["Reporter","Partner","HS4","TradeFlow"])):
    y = grp.set_index("date")["Value"]
    if len(y) < 24: continue
    try:
        mod = sm.tsa.SARIMAX(y, order=(1,1,0), seasonal_order=(0,1,1,12),
                             enforce_stationarity=False).fit(disp=False)
        pickle.dump(mod, open(M/f"{'__'.join(key)}.pkl","wb"))
    except: pass
print("SARIMAX done")
