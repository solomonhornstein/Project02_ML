"""
game_model.py - Trains a model to predict P(home team wins) and evaluates it.

TIME-BASED SPLIT: we train on games before `split_date` and test on games after.
A random split would let the model learn from games that happened AFTER the
ones it's being tested on, which makes results look better than they really are.

METRICS:
  - accuracy:  % of games where the favored team won
  - log loss:  penalizes confident wrong predictions (lower is better)
  - Brier:     average squared error of the probabilities (lower is better)
Log loss and Brier matter most here, because the playoff simulator relies on
the model's PROBABILITIES, not just its picks.

AI USE NOTE (required by DA Honor Code): This class was drafted with help from
Claude (Anthropic). [Add your own notes on what you changed or wrote yourself.]

Setup:  pip install scikit-learn
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import DataLoader
from feature_builder import FeatureBuilder


class GameModel:
    """Logistic regression model of P(home team wins)."""

    DEFAULT_FEATURES = ["diff_win_pct", "diff_run_diff", "diff_last10"]

    def __init__(self, features=None, split_date="2026-08-15"):
        self.features = features or self.DEFAULT_FEATURES
        self.split_date = pd.Timestamp(split_date)
        # StandardScaler puts all features on the same scale, so the model's
        # coefficients can be compared directly to see which features matter most.
        self.model = make_pipeline(StandardScaler(), LogisticRegression())

    def split(self, df):
        """Split features into train (before split_date) and test (after)."""
        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_date"])
        train = df[df["game_date"] < self.split_date]
        test = df[df["game_date"] >= self.split_date]
        return train, test

    def train(self, train):
        self.model.fit(train[self.features], train["home_win"])

    def win_prob(self, df):
        """P(home team wins) for each row of features."""
        return self.model.predict_proba(df[self.features])[:, 1]

    def evaluate(self, train, test):
        """Compare the model against two simple baselines on the test set."""
        y = test["home_win"]
        home_rate = train["home_win"].mean()  # learned from training data only

        results = {}

        # Baseline 1: always pick the home team, at the training-set home win rate.
        p_home = pd.Series(home_rate, index=test.index)
        results["Always pick home team"] = self._metrics(y, p_home)

        # Baseline 2: pick whichever team has the better record (accuracy only).
        picks = (test["diff_win_pct"] >= 0).astype(int)
        results["Pick better record"] = {
            "accuracy": accuracy_score(y, picks), "log_loss": None, "brier": None}

        # The model.
        results["Logistic regression"] = self._metrics(y, self.win_prob(test))

        return pd.DataFrame(results).T

    def coefficients(self):
        """How much each feature moves the prediction (per standard deviation)."""
        lr = self.model.named_steps["logisticregression"]
        coefs = pd.Series(lr.coef_[0], index=self.features)
        coefs["intercept (home field)"] = lr.intercept_[0]
        return coefs

    @staticmethod
    def _metrics(y, p):
        return {"accuracy": accuracy_score(y, (p >= 0.5).astype(int)),
                "log_loss": log_loss(y, p),
                "brier": brier_score_loss(y, p)}


if __name__ == "__main__":
    games = DataLoader().load_games()
    features = FeatureBuilder(prior_games=20).build(games)

    model = GameModel()
    train, test = model.split(features)
    print(f"Training games: {len(train)}   Test games: {len(test)}\n")

    model.train(train)
    print("Test-set results:")
    print(model.evaluate(train, test).round(4))

    print("\nCoefficients (log-odds per standard deviation):")
    print(model.coefficients().round(3))