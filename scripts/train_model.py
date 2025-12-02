# scripts/train_model.py
import argparse
from src.models.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features/features.parquet")
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--artifacts_dir", default="artifacts")
    args = parser.parse_args()
    train(args.features, args.out_dir, args.artifacts_dir)
