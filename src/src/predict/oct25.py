import pandas as pd, numpy as np, pathlib, pickle, lightgbm as lgb, json
from dateutil.relativedelta import relativedelta
P = pathlib.Path("data/processed")
df = pd.read_parquet(P/"features.parquet")
last = df[df.date == df.date.max()].copy()
oct_date = last.date.max() + relativedelta(months=2)
saridir = pathlib.Path("models/sarimax")
gbm = lgb.Booster(model_file="models/gbm.txt")
w = json.load(open("models/weights.json"))
rows=[]
for _,r in last.iterrows():
    key=(r.Reporter,r.Partner,r.HS4,r.TradeFlow)
    try: y_s=pickle.load(open(saridir/f\"{'__'.join(key)}.pkl\",\"rb\")).forecast(2)[-1]
    except: y_s=np.nan
    feat=r.copy()
    feat["month"]=oct_date.month
    for k in range(12,1,-1): feat[f\"lag{k}\"]=feat[f\"lag{k-1}\"]
    feat["lag1"]=r.Value
    feat["yoy_growth"]=0
    y_g=gbm.predict(pd.DataFrame([feat[["yoy_growth","month"]+[f\"lag{k}\" for k in range(1,13)]]]))[0]
    val=w["sarimax"]*(y_s if not np.isnan(y_s) else y_g)+w["gbm"]*y_g
    rows.append(dict(Country1=r.Reporter,Country2=r.Partner,ProductCode=r.HS4,
                     TradeFlow=r.TradeFlow,Value=max(0,round(val))))
pd.DataFrame(rows).to_parquet(P/"forecasts_oct25.parquet",index=False)
print("oct forecasts saved")
