"""Generate synthetic credit dataset as CSV.
Usage:
python scripts/generate_synthetic_data.py --n 10000 --out data/synthetic/customers.csv
"""
import argparse
import numpy as np
import pandas as pd




def sigmoid(x):
	return 1 / (1 + np.exp(-x))




def generate(n=10000, seed=42):
	rng = np.random.default_rng(seed)
	# demographics
	age = rng.integers(18, 75, n)
	income = rng.normal(50000, 20000, n).clip(8000, None)
	# credit lines and balances
	credit_limit = rng.normal(10000, 5000, n).clip(500, None)
	revolving_balance = (rng.random(n) * credit_limit).clip(0)
	debt = rng.normal(10000, 8000, n).clip(0)
	# behavioral
	num_past_due = rng.poisson(0.5, n)
	days_past_due = (num_past_due * rng.integers(1, 120, n)).clip(0)
	# derived signals
	dti = debt / (income + 1e-9)
	utilization = revolving_balance / (credit_limit + 1e-9)


	# base log-odds for default
	base = -3.5
	logit = (
		base
		+ 4.0 * dti
		+ 3.5 * utilization
		+ 0.8 * (num_past_due > 0).astype(float)
		+ 0.01 * (70 - (age - 18))
		+ rng.normal(0, 0.5, n)
	)
	p_default = sigmoid(logit)
	target = (rng.random(n) < p_default).astype(int)


	df = pd.DataFrame({
		'customer_id': [f'CUST{100000+i}' for i in range(n)],
		'age': age,
		'income': income.round(2),
		'debt': debt.round(2),
		'credit_limit': credit_limit.round(2),
		'revolving_balance': revolving_balance.round(2),
		'num_past_due': num_past_due,
		'days_past_due': days_past_due,
		'dti': dti.round(4),
		'utilization': utilization.round(4),
		'target': target,
	})
	return df




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--n', type=int, default=10000)
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--out', type=str, default='data/synthetic/customers.csv')
	args = parser.parse_args()
	df = generate(n=args.n, seed=args.seed)
	# ensure directories exist
	import pathlib
	pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(args.out, index=False)
	print(f'Wrote {len(df)} rows to {args.out}')