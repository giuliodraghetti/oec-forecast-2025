import lightgbm as lgb, pandas as pd, pathlib, numpy as np
P = pathlib.Path("data/processed")
df = pd.read_parquet(P/"features.parquet")
train = df[df.date <= "2024-12-31"]
valid = df[(df.date > "2024-12-31") & (df.date <= "2025-08-31")]
features = ["yoy_growth","month"]+[f"lag{k}" for k in range(1,13)]
lgb_train = lgb.Dataset(train[features], train.Value)
lgb_valid = lgb.Dataset(valid[features], valid.Value, reference=lgb_train)
model = lgb.train(dict(objective="poisson", learning_rate=0.05,
                       num_leaves=255, metric="mae"),
                  lgb_train, valid_sets=[lgb_valid],
                  early_stopping_rounds=100)
pathlib.Path("models").mkdir(exist_ok=True)
model.save_model("models/gbm.txt")
print("GBM saved")
