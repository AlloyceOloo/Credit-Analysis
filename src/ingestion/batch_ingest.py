"""Simple batch ingest: CSV -> DQ checks -> Parquet (raw & processed)
Usage:
 python src/ingestion/batch_ingest.py --src data/synthetic/customers.csv
"""
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path


REQUIRED_COLUMNS = [
	'customer_id', 'age', 'income', 'debt', 'credit_limit', 'revolving_balance',
	'num_past_due', 'days_past_due', 'dti', 'utilization', 'target'
]


def run_checks(df: pd.DataFrame) -> dict:
	checks = {}
	checks['n_rows'] = len(df)
	checks['missing_pct'] = df.isna().mean().to_dict()
	checks['income_min'] = float(df['income'].min())
	checks['income_max'] = float(df['income'].max())
	# simple range check
	checks['income_ok'] = checks['income_min'] >= 0
	checks['dti_range_ok'] = float(df['dti'].min()) >= 0
	checks['util_range_ok'] = ((df['utilization'] >= 0) & (df['utilization'] <=
	10)).all()
	checks['target_balance'] = int(df['target'].sum())
	
	return checks


def ingest(src: str, out_raw: str = 'data/raw/customers.parquet', out_proc: str = 'data/processed/customers.parquet') -> dict:
	src = Path(src)

	if not src.exists():
		raise FileNotFoundError(f'{src} not found')

	df = pd.read_csv(src)
	# quick schema validation
	missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
	
	if missing_cols:
		raise ValueError(f'Missing columns: {missing_cols}')

	# run checks
	checks = run_checks(df)
	# write raw
	Path(out_raw).parent.mkdir(parents=True, exist_ok=True)
	df.to_parquet(out_raw, index=False)
	# basic processing: cast types, add ingestion_ts
	df_proc = df.copy()
	df_proc['ingestion_ts'] = pd.Timestamp.utcnow()
	# compute a simple risk_score for quick use
	df_proc['risk_score_raw'] = (df_proc['dti'] * 0.5 + df_proc['utilization'] *
	0.4 + df_proc['num_past_due'] * 0.2)
	Path(out_proc).parent.mkdir(parents=True, exist_ok=True)
	df_proc.to_parquet(out_proc, index=False)
	# persist checks metadata
	meta = {
		"n_rows": int(len(df_proc)),
    	"columns": df_proc.columns.tolist(),
    	"nulls": {k: bool(v) for k, v in df_proc.isnull().any().to_dict().items()},
    	"out_raw": str(out_raw),
    	"out_proc": str(out_proc)
	}

	Path(out_proc).with_suffix('.meta.json').write_text(json.dumps(meta, indent=2))
	
	return meta

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--src', type=str, default='data/synthetic/customers.csv')
	parser.add_argument('--out_raw', type=str, default='data/raw/customers.parquet')
	parser.add_argument('--out_proc', type=str, default='data/processed/customers.parquet')
	args = parser.parse_args()
	meta = ingest(args.src, args.out_raw, args.out_proc)
	print('Ingestion finished. Metadata:')
	print(json.dumps(meta, indent=2))