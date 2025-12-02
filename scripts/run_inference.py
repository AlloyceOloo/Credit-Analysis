import argparse
from pathlib import Path
from src.inference.predict import load_models, predict
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", required=True)
parser.add_argument("--models_dir", default="models")
parser.add_argument("--out_csv", default="artifacts/predictions.csv")
args = parser.parse_args()

input_path = Path(args.input_csv)
if input_path.suffix == ".parquet":
    df = pd.read_parquet(input_path)
else:
    df = pd.read_csv(input_path)

lgbm_model, lgbm_features, scorecard_artifact = load_models(args.models_dir)
preds = predict(df, lgbm_model, lgbm_features, scorecard_artifact)

Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
preds.to_csv(args.out_csv, index=False)
print(f"Saved predictions to {args.out_csv}")
