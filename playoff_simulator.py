"""
playoff_simulator.py - Predicts and simulates the MLB postseason.

HOW IT WORKS:
  1. Train the tuned model (prior_games=70, run diff + last10) on ALL regular-
     season games. We're done testing, so using every game gives the best model.
  2. Use end-of-season team ratings to get P(home team wins) for any matchup.
  3. BRACKET PREDICTION: for each series, calculate the exact probability of
     every possible result (e.g. "Rays in 4") by walking through every way the
     series could play out with the real home/away pattern. Pick the favorite,
     advance them, and repeat through the World Series.
  4. MONTE CARLO ODDS: play out the remaining postseason N_SIMS times with
     random draws and count how often each team reaches each round.

Why both? The bracket is one specific forecast. The odds show how uncertain it
is. The predicted champion isn't always the team with the best title odds,
because the bracket follows only the single most likely path.

KEEP IT CURRENT: enter finished series in RESULTS and partial series in
IN_PROGRESS, then rerun. Commit the output BEFORE each round starts, so the
Git timestamps prove the predictions weren't made with hindsight.

AI USE NOTE (required by DA Honor Code): This class was drafted with help from
Claude (Anthropic).
"""

import random
from collections import Counter, defaultdict
from datetime import date

import pandas as pd

from data_loader import DataLoader
from feature_builder import FeatureBuilder
from game_model import GameModel

from pitcher_features import PitcherFeatureBuilder
from pitcher_loader import PitcherLoader

# ---------------------------------------------------------------------------
# Bracket from MLB.com. Team names must match your data exactly.
# ---------------------------------------------------------------------------
SEEDS = {
    "AL": {1: "Tampa Bay Rays", 2: "Cleveland Guardians", 3: "Houston Astros",
           4: "New York Yankees", 5: "Boston Red Sox", 6: "Chicago White Sox"},
    "NL": {1: "Milwaukee Brewers", 2: "Los Angeles Dodgers", 3: "Atlanta Braves",
           4: "San Diego Padres", 5: "Chicago Cubs", 6: "Philadelphia Phillies"},
}

# Finished series: series key -> winning team. Valid keys:
#   "AL_WC_3v6", "AL_WC_4v5", "AL_DS_1", "AL_DS_2", "AL_CS"   (same for NL)
#   "WS"
RESULTS = {
    "AL_WC_3v6": "Chicago White Sox",
    "AL_WC_4v5": "New York Yankees",
    "NL_WC_4v5": "San Diego Padres",
    "NL_WC_3v6": "Atlanta Braves",
    "AL_DS_1": "Tampa Bay Rays",
    "AL_DS_2": "Cleveland Guardians",
    "NL_DS_1": "Milwaukee Brewers",
    "NL_DS_2": "Los Angeles Dodgers",
}

# Series in progress: series key -> (higher seed's wins, lower seed's wins)
# Example: IN_PROGRESS = {"AL_DS_1": (1, 0)}
IN_PROGRESS = {}
PRE_PLAYOFF = False  # True = ignore all results; predict the whole postseason from the start

MODEL_VERSIONS = {
    "v1": {"prior_games": 70, "features": ["diff_run_diff", "diff_last10"]},
    "v2": {"prior_games": 70, "prior_innings": 30,
           "features": ["diff_run_diff", "diff_last10", "diff_starter_fip"]},
}
MODEL_VERSION = "v2"
PRIOR_GAMES = MODEL_VERSIONS[MODEL_VERSION]["prior_games"]
PRIOR_INNINGS = MODEL_VERSIONS[MODEL_VERSION].get("prior_innings")  # None for V1
FEATURES = MODEL_VERSIONS[MODEL_VERSION]["features"]
N_SIMS = 10_000

# Which team hosts each game: "H" = higher seed at home, "L" = lower seed at home.
HOME_PATTERNS = {3: "HHH",       # Wild Card: higher seed hosts every game
                 5: "HHLLH",     # Division Series: 2-2-1
                 7: "HHLLLHH"}   # LCS and World Series: 2-3-2

ROUND_NAMES = {"WC": "Wild Card", "DS": "Division Series",
               "CS": "Championship Series", "WS": "World Series"}


class PlayoffSimulator:
    """Bracket prediction and Monte Carlo simulation of the remaining postseason."""

    def __init__(self, seeds, model, ratings, records, results=None, in_progress=None):
        self.seeds = seeds
        self.model = model
        self.ratings = ratings          # end-of-season team features
        self.records = records          # actual regular-season win %
        self.results = results or {}
        self.in_progress = in_progress or {}
        self.seed_of = {team: (lg, s) for lg, d in seeds.items() for s, team in d.items()}
        self._cache = {}                # (home, away) -> P(home wins)
        self.picks = []                 # filled in by predict_bracket()

    # ---- game probabilities ----

    def p_home_wins(self, home, away):
        """Model's probability that `home` beats `away` at home."""
        if (home, away) not in self._cache:
            h, a = self.ratings.loc[home], self.ratings.loc[away]
            row = pd.DataFrame([{
                "diff_win_pct": h["win_pct"] - a["win_pct"],
                "diff_run_diff": h["run_diff"] - a["run_diff"],
                "diff_last10": h["last10"] - a["last10"],
                "diff_starter_fip": h.get("rotation_fip", 0) - a.get("rotation_fip", 0),
            }])
            self._cache[(home, away)] = self.model.win_prob(row)[0]
        return self._cache[(home, away)]

    def p_high_wins_game(self, high, low, n_games, game):
        """P(higher seed wins game number `game`), given who hosts it."""
        if HOME_PATTERNS[n_games][game] == "H":
            return self.p_home_wins(high, low)
        return 1 - self.p_home_wins(low, high)

    # ---- bracket prediction (exact, no randomness) ----

    def series_outcomes(self, high, low, n_games, start=(0, 0)):
        """Exact probability of every final result, as {(winner, games played): prob}.
        Walks through every possible sequence of game wins and losses."""
        need = n_games // 2 + 1
        outcomes = defaultdict(float)

        def walk(wins_h, wins_l, prob):
            if wins_h == need:
                outcomes[(high, wins_h + wins_l)] += prob
            elif wins_l == need:
                outcomes[(low, wins_h + wins_l)] += prob
            else:
                p = self.p_high_wins_game(high, low, n_games, wins_h + wins_l)
                walk(wins_h + 1, wins_l, prob * p)
                walk(wins_h, wins_l + 1, prob * (1 - p))

        walk(*start, 1.0)
        return outcomes

    def predict_series(self, key, high, low, n_games):
        """Pick the favorite in one series, record the pick, and return the winner."""
        round_code = "WS" if key == "WS" else key.split("_")[1]   # e.g. "AL_DS_1" -> "DS"
        pick = {"round": ROUND_NAMES[round_code],
                "series": key,
                "matchup": f"{self._label(high)} vs. {self._label(low)}"}

        if key in self.results:
            winner = self.results[key]
            pick.update({"prediction": f"{winner} (final)", "p_win_series": None,
                         "p_exact_result": None})
        else:
            start = self.in_progress.get(key, (0, 0))
            outcomes = self.series_outcomes(high, low, n_games, start)
            p_high = sum(p for (team, _), p in outcomes.items() if team == high)
            winner = high if p_high >= 0.5 else low
            games, p_exact = max(((g, p) for (team, g), p in outcomes.items()
                                  if team == winner), key=lambda x: x[1])
            pick.update({"prediction": f"{winner} in {games}",
                         "p_win_series": p_high if winner == high else 1 - p_high,
                         "p_exact_result": p_exact})
            if start != (0, 0):
                pick["matchup"] += f" (series at {start[0]}-{start[1]})"

        self.picks.append(pick)
        return winner

    def predict_bracket(self):
        """Fill out the full bracket by picking the favorite in every series."""
        self.picks = []
        *_, champ = self.simulate_once(play=self.predict_series)
        return pd.DataFrame(self.picks), champ

    # ---- Monte Carlo simulation (random) ----

    def play_series(self, key, high, low, n_games):
        """Simulate one series with random draws and return the winner."""
        if key in self.results:
            return self.results[key]
        need = n_games // 2 + 1
        wins_h, wins_l = self.in_progress.get(key, (0, 0))
        while wins_h < need and wins_l < need:
            p = self.p_high_wins_game(high, low, n_games, wins_h + wins_l)
            if random.random() < p:
                wins_h += 1
            else:
                wins_l += 1
        return high if wins_h == need else low

    # ---- bracket structure (shared by prediction and simulation) ----

    def simulate_league(self, lg, play):
        """Play one league's bracket with `play` (random or favorite-picking)."""
        s = self.seeds[lg]
        wc_36 = play(f"{lg}_WC_3v6", s[3], s[6], 3)
        wc_45 = play(f"{lg}_WC_4v5", s[4], s[5], 3)
        ds_1 = play(f"{lg}_DS_1", s[1], wc_45, 5)
        ds_2 = play(f"{lg}_DS_2", s[2], wc_36, 5)
        # LCS home field: the better seed. (Verify this rule on MLB.com.)
        high, low = sorted([ds_1, ds_2], key=lambda t: self.seed_of[t][1])
        pennant = play(f"{lg}_CS", high, low, 7)
        return [s[1], s[2], wc_36, wc_45], [ds_1, ds_2], pennant

    def simulate_once(self, play=None):
        play = play or self.play_series
        al_ds, al_cs, al_champ = self.simulate_league("AL", play)
        nl_ds, nl_cs, nl_champ = self.simulate_league("NL", play)
        # World Series home field: better regular-season record.
        high, low = sorted([al_champ, nl_champ], key=lambda t: -self.records[t])
        champ = play("WS", high, low, 7)
        return al_ds + nl_ds, al_cs + nl_cs, [al_champ, nl_champ], champ

    def run(self, n_sims):
        counts = {r: Counter() for r in ["Division Series", "LCS", "World Series", "Champion"]}
        for _ in range(n_sims):
            ds, cs, ws, champ = self.simulate_once()
            counts["Division Series"].update(ds)
            counts["LCS"].update(cs)
            counts["World Series"].update(ws)
            counts["Champion"][champ] += 1

        rows = []
        for team, (lg, seed) in self.seed_of.items():
            row = {"team": team, "league": lg, "seed": seed}
            for r, c in counts.items():
                row[f"P({r})"] = c[team] / n_sims
            rows.append(row)
        return pd.DataFrame(rows).sort_values("P(Champion)", ascending=False)

    def _label(self, team):
        lg, seed = self.seed_of[team]
        return f"({lg} {seed}) {team}"


def check_names(seeds, valid):
    bad = [t for d in seeds.values() for t in d.values() if t not in valid]
    if bad:
        raise ValueError(f"Unknown team names {bad}.\nValid names: {sorted(valid)}")


def pct(x):
    return "" if x is None or pd.isna(x) else f"{x:.1%}"


if __name__ == "__main__":
    random.seed(42)  # makes results reproducible from run to run

    games = DataLoader().load_games()
    if PRIOR_INNINGS is None:  # V1: team stats only
        team_builder = FeatureBuilder(prior_games=PRIOR_GAMES)
        features = team_builder.build(games)
    else:  # V2: team stats + starting pitchers
        starters = PitcherLoader().load_starters(games)
        v2_builder = PitcherFeatureBuilder(prior_games=PRIOR_GAMES, prior_innings=PRIOR_INNINGS)
        features = v2_builder.build(games, starters)
        team_builder = v2_builder.team_builder

    # Train on the whole regular season (split date after the season ends).
    model = GameModel(features=FEATURES, split_date="2027-01-01")
    train, _ = model.split(features)
    model.train(train)

    ratings = team_builder.team_ratings()
    records = {t: s["wins"] / s["games"] for t, s in team_builder.teams.items()}

    if PRIOR_INNINGS is not None:
        # Playoff starters aren't known in advance: rate each team's top-4 rotation.
        rotations = v2_builder.team_rotations(top_n=4)
        ratings["rotation_fip"] = rotations.reindex(ratings.index).fillna(rotations.mean())
        playoff_teams = [t for d in SEEDS.values() for t in d.values()]
        print("Playoff rotation FIP (top 4 starters, lower is better):")
        print(rotations.reindex(playoff_teams).sort_values().round(3), "\n")
    check_names(SEEDS, set(ratings.index))

    results = {} if PRE_PLAYOFF else RESULTS
    in_progress = {} if PRE_PLAYOFF else IN_PROGRESS
    sim = PlayoffSimulator(SEEDS, model, ratings, records, results, in_progress)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 80)

    # 1) Full bracket prediction
    bracket, champion = sim.predict_bracket()
    print("PREDICTED BRACKET")
    print("p_win_series = chance the pick wins the series; "
          "p_exact_result = chance of that exact result (e.g. 'in 4')\n")
    shown = bracket.drop(columns="series")
    for col in ["p_win_series", "p_exact_result"]:
        shown[col] = shown[col].map(pct)
    print(shown.to_string(index=False))
    print(f"\nPredicted World Series champion: {champion}\n")

    # 2) Monte Carlo odds
    table = sim.run(N_SIMS)
    print(f"POSTSEASON ODDS from {N_SIMS:,} simulations\n")
    print(table.to_string(index=False, float_format=pct))

    today = date.today()
    prefix = "preplayoff_" if PRE_PLAYOFF else ""
    bracket.to_csv(f"data/{prefix}bracket_{MODEL_VERSION}_{today}.csv", index=False)
    table.to_csv(f"data/{prefix}predictions_{MODEL_VERSION}_{today}.csv", index=False)
    print(f"\nSaved data/{prefix}bracket_{MODEL_VERSION}_{today}.csv and "
          f"data/{prefix}predictions_{MODEL_VERSION}_{today}.csv."
          " Commit them before the next games start!")