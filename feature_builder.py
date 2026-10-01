"""
feature_builder.py - Turns raw game results into pre-game features for each game.

KEY IDEA (no data leakage): we walk through the season in date order. For each
game we FIRST record both teams' stats as they stood before the game, and only
THEN update the teams' running totals with that game's result. So every feature
uses only information that was available before first pitch.

EARLY-SEASON NOISE: every team starts with `prior_games` imaginary games of .500
baseball and zero run differential. This pulls small-sample stats toward league
average (a 3-0 team looks like ~.565, not 1.000) and fades as real games pile up.

AI USE NOTE (required by DA Honor Code): This class was drafted with help from
Claude (Anthropic). [Add your own notes on what you changed or wrote yourself.]
"""

from collections import defaultdict, deque

import pandas as pd

from data_loader import DataLoader


class FeatureBuilder:
    """Builds leak-free, home-minus-away team features for every game."""

    def __init__(self, prior_games=20, recent_window=10):
        self.prior_games = prior_games      # strength of the pull toward .500
        self.recent_window = recent_window  # how many games count as "recent form"
        self.teams = {}                     # running stats, filled in by build()

    def build(self, games):
        """Return a DataFrame with one row of pre-game features per game."""
        games = games.sort_values(["game_date", "game_num", "game_id"])
        teams = defaultdict(self._new_team)
        rows = []

        for g in games.itertuples(index=False):
            home, away = teams[g.home_name], teams[g.away_name]

            # 1) Record features BEFORE this game is played.
            h, a = self._team_features(home), self._team_features(away)
            rows.append({
                "game_id": g.game_id,
                "game_date": g.game_date,
                "home_name": g.home_name,
                "away_name": g.away_name,
                # Features are home minus away. Home-field advantage itself is
                # learned by the model's intercept, so it isn't a feature here.
                "diff_win_pct": h["win_pct"] - a["win_pct"],
                "diff_run_diff": h["run_diff"] - a["run_diff"],
                "diff_last10": h["last10"] - a["last10"],
                "home_games_played": home["games"],
                "away_games_played": away["games"],
                "home_win": g.home_win,  # the target the model will predict
            })

            # 2) THEN update both teams with this game's result.
            self._update(home, g.home_score, g.away_score)
            self._update(away, g.away_score, g.home_score)

        self.teams = dict(teams)
        return pd.DataFrame(rows)

    def team_ratings(self):
        """Each team's stats after the last game processed.
        You'll use these end-of-season numbers to predict playoff games."""
        ratings = {name: self._team_features(t) for name, t in self.teams.items()}
        return pd.DataFrame(ratings).T.sort_values("run_diff", ascending=False)

    # ---- helpers ----

    def _new_team(self):
        return {"games": 0, "wins": 0, "runs_for": 0, "runs_against": 0,
                "recent": deque(maxlen=self.recent_window)}

    def _team_features(self, t):
        k = self.prior_games
        win_pct = (t["wins"] + 0.5 * k) / (t["games"] + k)
        run_diff = (t["runs_for"] - t["runs_against"]) / (t["games"] + k)
        last10 = sum(t["recent"]) / len(t["recent"]) if t["recent"] else 0.5
        return {"win_pct": win_pct, "run_diff": run_diff, "last10": last10}

    def _update(self, t, scored, allowed):
        won = int(scored > allowed)
        t["games"] += 1
        t["wins"] += won
        t["runs_for"] += scored
        t["runs_against"] += allowed
        t["recent"].append(won)


if __name__ == "__main__":
    games = DataLoader().load_games()
    builder = FeatureBuilder()
    features = builder.build(games)
    features.to_csv("data/features_2026.csv", index=False)
    print(f"{len(features)} rows of features saved")

    # Sanity check: each feature should be POSITIVELY correlated with home_win.
    print("\nCorrelation with home_win:")
    print(features[["diff_win_pct", "diff_run_diff", "diff_last10", "home_win"]]
          .corr()["home_win"].drop("home_win").round(3))

    print("\nEnd-of-season team ratings (top 12 by run differential):")
    print(builder.team_ratings().head(12).round(3))