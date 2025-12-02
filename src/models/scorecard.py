import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


# ------------------------------------------------------------
# Compute score for a single row (must be top-level for pickling)
# ------------------------------------------------------------

def pd_to_score(row: pd.Series, scorecard_dict: dict) -> float:
    """Compute final score for a single row."""
    total = 0
    for feat, bins in scorecard_dict.items():
        value = row[feat]
        for b in bins:
            low, high = b["bin"]
            if low <= value < high:
                total += b["points"]
                break
    return total


# ------------------------------------------------------------
# Build the scorecard dictionary
# ------------------------------------------------------------

def make_scorecard_dict(features, woe_bins, points):
    """
    Combine all WOE bins + assigned points into a serializable dict.

    features: list of feature names
    woe_bins: list[list[{"bin": (low, high), "woe": float}]]
    points:   list[list[int]]
    """
    scorecard = {}
    for feat, feat_bins, feat_points in zip(features, woe_bins, points):
        combined = []
        for b, p in zip(feat_bins, feat_points):
            combined.append({
                "bin": b["bin"],
                "woe": b["woe"],
                "points": p
            })
        scorecard[feat] = combined
    return scorecard


# ------------------------------------------------------------
# Main scorecard fitter
# ------------------------------------------------------------

def fit_scorecard(X: pd.DataFrame, y: pd.Series, n_bins: int = 5):
    """
    Fit a WOE-based scorecard from numeric X and binary y.

    Returns:
        scorecard_dict
        discretizer
    """

    # 1. Fit binner
    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="quantile",
    )
    X_binned = discretizer.fit_transform(X)

    features = list(X.columns)
    y_arr = y.values

    woe_bin_list = []
    points_list = []

    # 2. Per feature, compute WOE + score points
    for idx, feat in enumerate(features):

        edges = discretizer.bin_edges_[idx]
        x_col = X_binned[:, idx]

        feat_bins = []
        feat_points = []

        # build bins
        for b in range(len(edges) - 1):
            low, high = edges[b], edges[b + 1]

            mask = (x_col == b)

            good = ((y_arr == 0) & mask).sum()
            bad = ((y_arr == 1) & mask).sum()

            # protect from division by zero
            good = good if good > 0 else 0.5
            bad = bad if bad > 0 else 0.5

            woe = float(np.log(good / bad))

            # Score = -20 * WOE (classic WOE score scaling)
            pts = int(-20 * woe)

            feat_bins.append({
                "bin": (float(low), float(high)),
                "woe": woe,
            })
            feat_points.append(pts)

        woe_bin_list.append(feat_bins)
        points_list.append(feat_points)

    # 3. Build final scorecard dict
    scorecard_dict = make_scorecard_dict(features, woe_bin_list, points_list)

    return scorecard_dict, discretizer
