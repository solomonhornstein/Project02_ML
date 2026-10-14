"""
experiments.py - Tunes the model's settings without "peeking" at the test set.

Data is split three ways by date:
  TRAIN:       games before July 15         -> fit each candidate model
  VALIDATION:  July 15 to August 14         -> choose the best settings
  TEST:        August 15 onward             -> check the winner ONCE at the end
If we picked the best of many settings by their TEST scores, the winner would
look better partly by luck. Keeping the test set untouched avoids that.

AI USE NOTE (required by DA Honor Code): This script was drafted with help from
Claude (Anthropic).
"""

import pandas as pd

from data_loader import DataLoader
from feature_builder import FeatureBuilder
from game_model import GameModel

FEATURE_SETS = {
    "all three": ["diff_win_pct", "diff_run_diff", "diff_last10"],
    "run diff + last10": ["diff_run_diff", "diff_last10"],
    "run diff only": ["diff_run_diff"],
    "win pct only": ["diff_win_pct"],
}
PRIOR_GAMES = [1, 20, 40, 70]  # 1 = essentially no regression (0 would divide by zero)

VALIDATION_START = "2026-07-15"
TEST_START = "2026-08-15"

games = DataLoader().load_games()
rows = []

for k in PRIOR_GAMES:
    features = FeatureBuilder(prior_games=k).build(games)
    features["game_date"] = pd.to_datetime(features["game_date"])
    before_test = features[features["game_date"] < pd.Timestamp(TEST_START)]

    for name, cols in FEATURE_SETS.items():
        # Split the pre-test games into train / validation.
        model = GameModel(features=cols, split_date=VALIDATION_START)
        train, valid = model.split(before_test)
        model.train(train)
        res = model.evaluate(train, valid).loc["Logistic regression"]
        rows.append({"prior_games": k, "features": name, **res.to_dict()})

results = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
print("Validation results (sorted by log loss, lower is better):")
print(results.round(4).to_string(index=False))
results.to_csv("data/experiment_results.csv", index=False)

# Retrain the winner on everything before the test period, then test ONCE.
best = results.iloc[0]
print(f"\nBest: prior_games={best['prior_games']}, features='{best['features']}'")

features = FeatureBuilder(prior_games=int(best["prior_games"])).build(games)
final = GameModel(features=FEATURE_SETS[best["features"]], split_date=TEST_START)
train, test = final.split(features)
final.train(train)
print("\nFinal test-set results for the best settings:")
print(final.evaluate(train, test).round(4))