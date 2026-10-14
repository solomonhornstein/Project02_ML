"""
cross_validation.py - Settles V1 vs. V2 with many more test games.

TIME-SERIES CROSS-VALIDATION: instead of one test window, we test on each
month from May through September. For each month, the model trains only on
games BEFORE that month, so it never sees the future:
    train Mar-Apr      -> test May
    train Mar-May      -> test Jun
    ...
    train Mar-Aug      -> test Sep
That gives ~2,000 test games instead of a few hundred.

PAIRED COMPARISON: both versions predict the exact same games, so we compare
them game by game: (V2's log loss - V1's log loss) for each game. The average
difference tells us which is better; its standard error tells us how much of
that could be luck. If the difference is more than about 2 standard errors
from zero, it's unlikely to be luck.

AI USE NOTE (required by DA Honor Code): This script was drafted with help from
Claude (Anthropic).
"""

import numpy as np
import pandas as pd

from data_loader import DataLoader
from game_model import GameModel
from pitcher_features import PitcherFeatureBuilder
from pitcher_loader import PitcherLoader

V1_FEATURES = ["diff_run_diff", "diff_last10"]
V2_FEATURES = V1_FEATURES + ["diff_starter_fip"]
PRIOR_GAMES = 70
PRIOR_INNINGS = [5, 10, 30]

# (first day of test month, first day after it)
FOLDS = [("2026-05-01", "2026-06-01"), ("2026-06-01", "2026-07-01"),
         ("2026-07-01", "2026-08-01"), ("2026-08-01", "2026-09-01"),
         ("2026-09-01", "2026-11-01")]


def game_log_loss(y, p):
    """Log loss for each individual game (lower is better)."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def predict_folds(features, cols):
    """Train before each month, predict that month. Returns one row per test game."""
    out = []
    for start, end in FOLDS:
        data = features[features["game_date"] < pd.Timestamp(end)]
        model = GameModel(features=cols, split_date=start)
        train, test = model.split(data)
        model.train(train)
        p = model.win_prob(test)
        out.append(pd.DataFrame({
            "game_id": test["game_id"].values, "month": start[:7],
            "home_win": test["home_win"].values, "p": p,
            "loss": game_log_loss(test["home_win"].values, p),
            "correct": ((p >= 0.5).astype(int) == test["home_win"].values),
        }))
    return pd.concat(out, ignore_index=True)


games = DataLoader().load_games()
starters = PitcherLoader().load_starters(games)

v1 = None
summary, monthly = [], {}
for k in PRIOR_INNINGS:
    features = PitcherFeatureBuilder(prior_games=PRIOR_GAMES, prior_innings=k).build(games, starters)
    features["game_date"] = pd.to_datetime(features["game_date"])
    if v1 is None:   # V1 features don't depend on prior_innings
        v1 = predict_folds(features, V1_FEATURES)
    v2 = predict_folds(features, V2_FEATURES)

    paired = v1.merge(v2, on=["game_id", "month", "home_win"], suffixes=("_v1", "_v2"))
    diff = paired["loss_v2"] - paired["loss_v1"]          # negative = V2 better
    se = diff.std() / np.sqrt(len(diff))
    summary.append({
        "prior_innings": k, "test_games": len(paired),
        "v1_log_loss": paired["loss_v1"].mean(), "v2_log_loss": paired["loss_v2"].mean(),
        "v1_accuracy": paired["correct_v1"].mean(), "v2_accuracy": paired["correct_v2"].mean(),
        "v2_minus_v1": diff.mean(), "std_error": se, "z": diff.mean() / se,
    })
    monthly[k] = paired.groupby("month").agg(
        games=("game_id", "size"), v1_log_loss=("loss_v1", "mean"),
        v2_log_loss=("loss_v2", "mean"))

summary = pd.DataFrame(summary)
print("Pooled results over all test months (lower log loss is better):")
print(summary.round(4).to_string(index=False))
print("\nv2_minus_v1 < 0 means V2 is better. |z| above ~2 means unlikely to be luck.")

best = summary.sort_values("v2_log_loss").iloc[0]
k = int(best["prior_innings"])
print(f"\nMonth by month (prior_innings={k}):")
table = monthly[k]
table["v2_better"] = table["v2_log_loss"] < table["v1_log_loss"]
print(table.round(4).to_string())
print(f"\nV2 better in {table['v2_better'].sum()} of {len(table)} months")

summary.to_csv("data/cross_validation_results.csv", index=False)