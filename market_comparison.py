"""
market_comparison.py - Compares the model's title odds with Kalshi's market odds.

WHY COMPARE TO A MARKET? Betting/prediction markets combine the opinions of
many people with money at stake, so they're a tough benchmark. If the model
lands close to the market, that's a good sign. Where it disagrees, ask why:
the market knows things the model doesn't (injuries, who's pitching, recent
roster moves), while the model only sees regular-season team stats.

NORMALIZING: Kalshi's "Yes" prices add up to more than 100% (the extra is
built-in margin, sometimes called the "vig" or "overround"). We divide each
price by the total so the market's probabilities add up to exactly 100%,
just like the model's.

SCORING: after the World Series, set CHAMPION and rerun. The script scores
the model and the market using log loss and Brier score (lower is better).

AI USE NOTE (required by DA Honor Code): This script was drafted with help from
Claude (Anthropic).

Optional chart:  pip install matplotlib
"""

import glob
import math

import pandas as pd

# ---------------------------------------------------------------------------
# Kalshi "Pro Baseball Champion" Yes prices. Update these whenever you
# capture new odds, and record when you captured them.
# ---------------------------------------------------------------------------
MODEL_VERSION = "v1"  # which model's predictions to compare
MARKET_CAPTURED = "2026-10-02"
MARKET_YES = {
    "Los Angeles Dodgers": 0.301,
    "Milwaukee Brewers": 0.172,
    "New York Yankees": 0.158,
    "Tampa Bay Rays": 0.109,
    "San Diego Padres": 0.101,
    "Atlanta Braves": 0.086,
    "Cleveland Guardians": 0.075,
    "Chicago White Sox": 0.053,
}

CHAMPION = None  # after the World Series, e.g. CHAMPION = "Milwaukee Brewers"


class MarketComparison:
    """Side-by-side comparison of model and market championship odds."""

    def __init__(self, predictions, market_yes):
        self.predictions = predictions
        self.market_yes = market_yes

    def table(self):
        """Model vs. market title odds for every team still alive."""
        df = self.predictions[["team", "league", "seed", "P(Champion)"]].rename(
            columns={"P(Champion)": "model"})
        df = df[df["model"] > 0].copy()  # drop eliminated teams

        missing = set(df["team"]) - set(self.market_yes)
        if missing:
            raise ValueError(f"No market price entered for: {sorted(missing)}")

        df["market_raw"] = df["team"].map(self.market_yes)
        total = df["market_raw"].sum()  # only teams still alive
        df["market"] = df["market_raw"] / total       # normalized to sum to 100%
        df["model_minus_market"] = df["model"] - df["market"]
        return df.sort_values("market", ascending=False).reset_index(drop=True)

    @staticmethod
    def score(table, champion):
        """Log loss and Brier score for the model and the market (lower is better)."""
        scores = {}
        for source in ["model", "market"]:
            p = table.set_index("team")[source]
            outcome = (p.index == champion).astype(int)
            scores[source] = {
                "log_loss": -math.log(max(p[champion], 1e-6)),
                "brier": ((p - outcome) ** 2).sum(),
            }
        return pd.DataFrame(scores).T

    @staticmethod
    def chart(table, path):
        """Grouped bar chart of model vs. market odds (needs matplotlib)."""
        import matplotlib.pyplot as plt

        x = range(len(table))
        width = 0.4
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([i - width / 2 for i in x], table["model"] * 100, width, label="My model")
        ax.bar([i + width / 2 for i in x], table["market"] * 100, width, label="Kalshi (normalized)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(table["team"], rotation=30, ha="right")
        ax.set_ylabel("Chance to win World Series (%)")
        ax.set_title(f"World Series odds: model vs. market ({MARKET_CAPTURED})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)


def latest_predictions():
    files = sorted(glob.glob(f"data/predictions_{MODEL_VERSION}_*.csv"))  # dates sort correctly
    if not files:
        raise FileNotFoundError("Run playoff_simulator.py first.")
    return files[-1], pd.read_csv(files[-1])


if __name__ == "__main__":
    path, predictions = latest_predictions()
    comparison = MarketComparison(predictions, MARKET_YES)
    table = comparison.table()

    print(f"Model: {path}")
    print(f"Market: Kalshi, captured {MARKET_CAPTURED} "
          f"(raw prices sum to {sum(MARKET_YES.values()):.1%}, normalized to 100%)\n")

    shown = table[["team", "seed", "model", "market", "model_minus_market"]].copy()
    shown["seed"] = table["league"] + " " + table["seed"].astype(str)
    for col in ["model", "market"]:
        shown[col] = shown[col].map(lambda v: f"{v:.1%}")
    shown["model_minus_market"] = table["model_minus_market"].map(lambda v: f"{v:+.1%}")
    print(shown.to_string(index=False))

    out = f"data/market_comparison_{MODEL_VERSION}_{MARKET_CAPTURED}.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved {out}")

    try:
        chart_path = f"data/model_vs_market_{MODEL_VERSION}_{MARKET_CAPTURED}.png"
        MarketComparison.chart(table, chart_path)
        print(f"Saved {chart_path}")
    except ImportError:
        print("(Install matplotlib to also save a chart.)")

    if CHAMPION:
        print(f"\nScores now that {CHAMPION} won (lower is better):")
        print(MarketComparison.score(table, CHAMPION).round(3))