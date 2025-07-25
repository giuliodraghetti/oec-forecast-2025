import pandas as pd, numpy as np, pathlib, pickle, lightgbm as lgb, json, tqdm, math
P = pathlib.Path("data/processed"); df = pd.read_parquet(P/"features.parquet")
df = df[(df.date >= "2025-01-01") & (df.date <= "2025-08-31")]
saridir = pathlib.Path("models/sarimax")
gbm = lgb.Booster(model_file="models/gbm.txt")
features = ["yoy_growth","month"]+[f"lag{k}" for k in range(1,13)]
def smape(a,f): return 100*np.mean(2*np.abs(f-a)/(np.abs(a)+np.abs(f)+1e-9))
errs = {"sarimax":[], "gbm":[]}
for _,r in tqdm.tqdm(df.iterrows(), total=len(df)):
    key = (r.Reporter,r.Partner,r.HS4,r.TradeFlow)
    try:
        yhat_s = pickle.load(open(saridir / f"{'__'.join(key)}.pkl","rb")).forecast(1)[0]
    except: yhat_s = np.nan
    yhat_g = gbm.predict(pd.DataFrame([r[features]]))[0]
    errs["sarimax"].append((r.Value,yhat_s))
    errs["gbm"].append((r.Value,yhat_g))
w = {}
for m in errs:
    vals = np.array([v for v, _ in errs[m] if not np.isnan(_)])
    preds = np.array([p for _, p in errs[m] if not np.isnan(p)])
    w[m] = 1/(smape(vals,preds)+1e-6)
tot = w["sarimax"]+w["gbm"]
w = {k:v/tot for k,v in w.items()}
json.dump(w, open("models/weights.json","w"))
print("weights", w)
