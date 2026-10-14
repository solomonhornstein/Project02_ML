"""
experiments_v2.py - Does the starting pitcher feature actually help?

FAIR COMPARISON: both versions use the SAME games (suspended game counted once)
and the SAME date splits. The only difference is whether diff_starter_fip is
included, so any change in log loss comes from the pitcher feature itself.

Same three-way split as experiments.py:
  TRAIN:       games before July 15         -> fit each candidate model
  VALIDATION:  July 15 to August 14         -> choose the best settings
  TEST:        August 15 onward             -> check the finalists ONCE

AI USE NOTE (required by DA Honor Code): This script was drafted with help from
Claude (Anthropic).
"""

import pandas as pd

from data_loader import DataLoader
from game_model import GameModel
from pitcher_features import PitcherFeatureBuilder
from pitcher_loader import PitcherLoader

V1_FEATURES = ["diff_run_diff", "diff_last10"]
FEATURE_SETS = {
    "V1 features": V1_FEATURES,
    "V1 + starter FIP": V1_FEATURES + ["diff_starter_fip"],
    "starter FIP only": ["diff_starter_fip"],
}
PRIOR_GAMES = 70                       # V1's tuned value, kept fixed
PRIOR_INNINGS = [10, 30, 60, 100]      # pitcher regression strength to tune

VALIDATION_START = "2026-07-15"
TEST_START = "2026-08-15"

games = DataLoader().load_games()
starters = PitcherLoader().load_starters(games)

rows = []
all_features = {}
for k in PRIOR_INNINGS:
    features = PitcherFeatureBuilder(prior_games=PRIOR_GAMES, prior_innings=k).build(games, starters)
    features["game_date"] = pd.to_datetime(features["game_date"])
    all_features[k] = features
    before_test = features[features["game_date"] < pd.Timestamp(TEST_START)]

    for name, cols in FEATURE_SETS.items():
        if "fip" not in " ".join(cols).lower() and k != PRIOR_INNINGS[0]:
            continue  # V1 features don't depend on prior_innings; run them once
        model = GameModel(features=cols, split_date=VALIDATION_START)
        train, valid = model.split(before_test)
        model.train(train)
        res = model.evaluate(train, valid).loc["Logistic regression"]
        rows.append({"prior_innings": k if "fip" in " ".join(cols) else "-",
                     "features": name, **res.to_dict()})

results = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
print("Validation results (sorted by log loss, lower is better):")
print(results.round(4).to_string(index=False))
results.to_csv("data/experiment_results_v2.csv", index=False)

# Best V2 setting by validation log loss.
v2_rows = results[results["features"] != "V1 features"]
best = v2_rows.iloc[0]
best_k = int(best["prior_innings"])
print(f"\nBest V2: prior_innings={best_k}, features='{best['features']}'")

# Final test, run ONCE: V1 features vs. best V2, same games, same split.
features = all_features[best_k]
print("\nFinal test-set results (games from", TEST_START, "on):")
final = {}
for label, cols in [("V1 features", V1_FEATURES),
                    (f"V2: {best['features']}", FEATURE_SETS[best["features"]])]:
    model = GameModel(features=cols, split_date=TEST_START)
    train, test = model.split(features)
    model.train(train)
    table = model.evaluate(train, test)
    final[label] = table.loc["Logistic regression"]
    if label != "V1 features":
        baselines = table.drop(index="Logistic regression")
        coefs = model.coefficients()

print(pd.concat([baselines, pd.DataFrame(final).T]).round(4))
print("\nV2 coefficients (log-odds per standard deviation):")
print(coefs.round(3))