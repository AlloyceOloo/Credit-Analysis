import argparse
from pathlib import Path
import pandas as pd

from src.features.fe_build import build_features


def main(src: str, out: str):
    p_src = Path(src)
    if not p_src.exists():
        raise FileNotFoundError(f"{p_src} not found")

    df = pd.read_parquet(p_src)
    df_feat = build_features(df)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(out, index=False)

    print(f"Created {len(df_feat)} feature rows at {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/processed/customers.parquet")
    parser.add_argument("--out", default="data/features/features.parquet")
    args = parser.parse_args()
    main(args.src, args.out)
