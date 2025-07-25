# oec-forecast-2025
End‑to‑end CPU pipeline: ingest → features → SARIMAX & LightGBM → ensemble → CSV.

## Quick run

```bash
conda env create -f env/conda.yml
conda activate oec-forecast-2025

# put the four raw ZIPs under data/raw/
python -m src.data.ingest
python -m src.features.build_features
python -m src.models.sarimax
python -m src.models.gbm
python -m src.models.ensemble
python -m src.predict.oct25
python -m src.submission.make_csv

---

### LICENSE (MIT boilerplate)

```text
MIT License … (fill in team details)
