"""
pitcher_features.py - V2 features: V1 team stats plus starting pitcher quality.

STAT: FIP (fielding independent pitching) uses only what a pitcher controls
directly: strikeouts, walks, hit batters, and home runs. It predicts future
performance better than ERA, which depends partly on the defense behind him.
    FIP = (13*HR + 3*(BB + HBP) - 2*K) / innings + constant
The constant is the same for every pitcher, so it cancels out when we take
home starter minus away starter. We leave it out. Lower FIP = better pitcher.

LEAK-FREE: same as V1. Walk through games in date order; record each starter's
FIP as of BEFORE the game, then update his totals with that game's line.

REGRESSION: a pitcher with two starts has very noisy stats, just like a 3-0
team. Each pitcher starts with `prior_innings` imaginary innings at the league
average rate (league totals so far, so that's leak-free too).

SUSPENDED GAMES: game 824912 appears twice in the schedule (started 6/16,
finished 6/17). V2 keeps only the completion date. V1 is left unchanged.

AI USE NOTE (required by DA Honor Code): This class was drafted with help from
Claude (Anthropic). [Add your own notes on what you changed or wrote yourself.]
"""

from collections import defaultdict

import pandas as pd

from data_loader import DataLoader
from feature_builder import FeatureBuilder
from pitcher_loader import PitcherLoader

DEFAULT_LEAGUE_RATE = 1.0   # rough league FIP rate per inning before data exists
MIN_LEAGUE_INNINGS = 300    # use the default until the league has this many innings


class PitcherFeatureBuilder:
    """Builds V1 team features plus a regressed starter-FIP difference."""

    def __init__(self, prior_games=70, prior_innings=30, recent_window=10):
        self.team_builder = FeatureBuilder(prior_games=prior_games,
                                           recent_window=recent_window)
        self.prior_innings = prior_innings
        self.pitchers = {}       # pitcher_id -> running totals (filled by build)
        self.league = {}         # league-wide running totals

    def build(self, games, starters):
        # Suspended games appear twice; keep the date the game finished.
        games = (games.sort_values(["game_date", "game_num", "game_id"])
                      .drop_duplicates("game_id", keep="last"))

        features = self.team_builder.build(games)    # V1 team features

        lines = starters.set_index(["game_id", "side"])
        pitchers = defaultdict(self._new_pitcher)
        league = self._new_pitcher()
        rows = []

        for g in games.itertuples(index=False):
            home = self._line(lines, g.game_id, "home")
            away = self._line(lines, g.game_id, "away")

            # 1) Record each starter's FIP BEFORE this game.
            lg_rate = self._league_rate(league)
            h_fip = self._fip(pitchers[home["pitcher_id"]], lg_rate) if home is not None else lg_rate
            a_fip = self._fip(pitchers[away["pitcher_id"]], lg_rate) if away is not None else lg_rate
            rows.append({
                "game_id": g.game_id,
                "home_starter": home["pitcher_name"] if home is not None else None,
                "away_starter": away["pitcher_name"] if away is not None else None,
                "home_starter_fip": h_fip,
                "away_starter_fip": a_fip,
                # Negative = home team has the better (lower-FIP) starter.
                "diff_starter_fip": h_fip - a_fip,
            })

            # 2) THEN update each starter and the league with this game's lines.
            for line in (home, away):
                if line is not None:
                    self._update(pitchers[line["pitcher_id"]], line)
                    self._update(league, line)
                    pitchers[line["pitcher_id"]]["name"] = line["pitcher_name"]
                    pitchers[line["pitcher_id"]]["team"] = line["team"]

        self.pitchers, self.league = dict(pitchers), league
        return features.merge(pd.DataFrame(rows), on="game_id", how="left")

    def pitcher_ratings(self, min_starts=1):
        """Each pitcher's end-of-season regressed FIP rate (lower is better)."""
        lg_rate = self._league_rate(self.league)
        rows = [{"pitcher": p["name"], "team": p["team"], "starts": p["starts"],
                 "innings": round(p["outs"] / 3, 1), "fip": self._fip(p, lg_rate)}
                for p in self.pitchers.values() if p["starts"] >= min_starts]
        return pd.DataFrame(rows).sort_values("fip").reset_index(drop=True)

    def team_rotations(self, top_n=4):
        """Average FIP of each team's top_n starters by innings. Used for playoffs,
        where we don't know who starts each future game."""
        ratings = self.pitcher_ratings()
        top = ratings.sort_values("innings", ascending=False).groupby("team").head(top_n)
        return top.groupby("team")["fip"].mean().sort_values()

    # ---- helpers ----

    @staticmethod
    def _new_pitcher():
        return {"outs": 0, "k": 0, "bb_hbp": 0, "hr": 0, "starts": 0,
                "name": None, "team": None}

    @staticmethod
    def _line(lines, game_id, side):
        try:
            return lines.loc[(game_id, side)]
        except KeyError:
            return None   # no box score for this game: treated as league average

    @staticmethod
    def _raw(p):
        return 13 * p["hr"] + 3 * p["bb_hbp"] - 2 * p["k"]

    def _league_rate(self, league):
        innings = league["outs"] / 3
        if innings < MIN_LEAGUE_INNINGS:
            return DEFAULT_LEAGUE_RATE
        return self._raw(league) / innings

    def _fip(self, p, lg_rate):
        """Regressed FIP rate: real innings blended with prior_innings of league average."""
        innings = p["outs"] / 3
        return (self._raw(p) + self.prior_innings * lg_rate) / (innings + self.prior_innings)

    @staticmethod
    def _update(p, line):
        p["outs"] += int(line["outs"])
        p["k"] += int(line["strikeouts"])
        p["bb_hbp"] += int(line["walks"]) + int(line["hit_by_pitch"])
        p["hr"] += int(line["home_runs"])
        p["starts"] += 1


if __name__ == "__main__":
    games = DataLoader().load_games()
    starters = PitcherLoader().load_starters(games)

    builder = PitcherFeatureBuilder(prior_games=70, prior_innings=30)
    features = builder.build(games, starters)
    features.to_csv("data/features_v2_2026.csv", index=False)
    print(f"{len(features)} rows of V2 features saved")
    print(f"Games missing a starter: {features['home_starter'].isna().sum()}")

    # Sanity check. The pitcher feature should be NEGATIVELY correlated with
    # home_win, because a lower (better) home starter FIP means more home wins.
    cols = ["diff_run_diff", "diff_last10", "diff_starter_fip"]
    print("\nCorrelation with home_win:")
    print(features[cols + ["home_win"]].corr()["home_win"].drop("home_win").round(3))

    print("\nBest starters by end-of-season FIP rate (min. 15 starts):")
    print(builder.pitcher_ratings(min_starts=15).head(10).round(3).to_string(index=False))