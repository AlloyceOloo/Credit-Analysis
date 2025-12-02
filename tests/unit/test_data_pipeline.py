import os
import pandas as pd
from pathlib import Path
from scripts.generate_synthetic_data import generate
from src.ingestion.batch_ingest import ingest


def test_generate_small():
	df = generate(n=100, seed=123)
	assert len(df) == 100
	assert 'customer_id' in df.columns
	assert df['income'].min() >= 0

def test_ingest_roundtrip(tmp_path):
	out_csv = tmp_path / 'customers.csv'
	df = generate(n=200, seed=7)
	df.to_csv(out_csv, index=False)
	out_raw = tmp_path / 'raw.parquet'
	out_proc = tmp_path / 'proc.parquet'
	meta = ingest(str(out_csv), str(out_raw), str(out_proc))
	assert Path(meta['out_raw']).exists()
	assert Path(meta['out_proc']).exists()
	proc_df = pd.read_parquet(meta['out_proc'])
	assert 'ingestion_ts' in proc_df.columns
	assert 'risk_score_raw' in proc_df.columns
	