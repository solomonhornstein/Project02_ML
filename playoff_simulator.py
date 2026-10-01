"""
playoff_simulator.py - Simulates the MLB postseason thousands of times.

HOW IT WORKS:
  1. Train the tuned model (prior_games=70, run diff + last10) on ALL regular-
     season games. We're done testing, so using every game gives the best model.
  2. Use end-of-season team ratings to get P(home team wins) for any matchup.
  3. Play out every remaining series game by game with random draws, using the
     real home/away patterns. Repeat N_SIMS times and count how often each team
     reaches each round.

KEEP IT CURRENT: enter finished series in RESULTS and partial series in
IN_PROGRESS, then rerun. Only the games still to be played get simulated.
Commit the predictions CSV BEFORE each round starts, so the Git timestamps
prove the predictions weren't made with hindsight.

AI USE NOTE (required by DA Honor Code): This class was drafted with help from
Claude (Anthropic). [Add your own notes on what you changed or wrote yourself.]
"""

import random
from collections import Counter
from datetime import date

import pandas as pd

from data_loader import DataLoader
from feature_builder import FeatureBuilder
from game_model import GameModel

# ---------------------------------------------------------------------------
# FILL THESE IN from MLB.com. Team names must match your data exactly,
# e.g. "Milwaukee Brewers" (the script lists valid names if one is wrong).
# ---------------------------------------------------------------------------
SEEDS = {
    "AL": {1: "Tampa Bay Rays", 2: "Cleveland Guardians", 3: "Houston Astros",
           4: "New York Yankees", 5: "Boston Red Sox", 6: "Chicago White Sox"},
    "NL": {1: "Milwaukee Brewers", 2: "Los Angeles Dodgers", 3: "Atlanta Braves",
           4: "San Diego Padres", 5: "Chicago Cubs", 6: "Philadelphia Phillies"},
}

RESULTS = {
    "AL_WC_3v6": "Chicago White Sox",
    "AL_WC_4v5": "New York Yankees",
    "NL_WC_4v5": "San Diego Padres",
}

IN_PROGRESS = {
    "NL_WC_3v6": (1, 1),   # Braves-Phillies tied, Game 3 tonight in Atlanta
}

PRIOR_GAMES = 70
FEATURES = ["diff_run_diff", "diff_last10"]
N_SIMS = 10_000

# Which team hosts each game: "H" = higher seed at home, "L" = lower seed at home.
HOME_PATTERNS = {3: "HHH",       # Wild Card: higher seed hosts every game
                 5: "HHLLH",     # Division Series: 2-2-1
                 7: "HHLLLHH"}   # LCS and World Series: 2-3-2


class PlayoffSimulator:
    """Monte Carlo simulation of the remaining postseason."""

    def __init__(self, seeds, model, ratings, records, results=None, in_progress=None):
        self.seeds = seeds
        self.model = model
        self.ratings = ratings          # end-of-season team features
        self.records = records          # actual regular-season win %
        self.results = results or {}
        self.in_progress = in_progress or {}
        self.seed_of = {team: (lg, s) for lg, d in seeds.items() for s, team in d.items()}
        self._cache = {}                # (home, away) -> P(home wins)

    def p_home_wins(self, home, away):
        """Model's probability that `home` beats `away` at home."""
        if (home, away) not in self._cache:
            h, a = self.ratings.loc[home], self.ratings.loc[away]
            row = pd.DataFrame([{
                "diff_win_pct": h["win_pct"] - a["win_pct"],
                "diff_run_diff": h["run_diff"] - a["run_diff"],
                "diff_last10": h["last10"] - a["last10"],
            }])
            self._cache[(home, away)] = self.model.win_prob(row)[0]
        return self._cache[(home, away)]

    def play_series(self, key, high, low, n_games):
        """Simulate one series and return the winner. `high` has home field."""
        if key in self.results:                       # already decided
            return self.results[key]
        need = n_games // 2 + 1
        wins_h, wins_l = self.in_progress.get(key, (0, 0))
        game = wins_h + wins_l
        pattern = HOME_PATTERNS[n_games]
        while wins_h < need and wins_l < need:
            if pattern[game] == "H":
                p_high = self.p_home_wins(high, low)
            else:
                p_high = 1 - self.p_home_wins(low, high)
            if random.random() < p_high:
                wins_h += 1
            else:
                wins_l += 1
            game += 1
        return high if wins_h == need else low

    def simulate_league(self, lg):
        """Play one league's bracket. Returns teams reaching each round."""
        s = self.seeds[lg]
        wc_36 = self.play_series(f"{lg}_WC_3v6", s[3], s[6], 3)
        wc_45 = self.play_series(f"{lg}_WC_4v5", s[4], s[5], 3)
        ds_1 = self.play_series(f"{lg}_DS_1", s[1], wc_45, 5)
        ds_2 = self.play_series(f"{lg}_DS_2", s[2], wc_36, 5)
        # LCS home field: the better seed. (Verify this rule on MLB.com.)
        high, low = sorted([ds_1, ds_2], key=lambda t: self.seed_of[t][1])
        pennant = self.play_series(f"{lg}_CS", high, low, 7)
        return [s[1], s[2], wc_36, wc_45], [ds_1, ds_2], pennant

    def simulate_once(self):
        al_ds, al_cs, al_champ = self.simulate_league("AL")
        nl_ds, nl_cs, nl_champ = self.simulate_league("NL")
        # World Series home field: better regular-season record.
        high, low = sorted([al_champ, nl_champ], key=lambda t: -self.records[t])
        champ = self.play_series("WS", high, low, 7)
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


def check_names(seeds, valid):
    bad = [t for d in seeds.values() for t in d.values() if t not in valid]
    if bad:
        raise ValueError(f"Unknown team names {bad}.\nValid names: {sorted(valid)}")


if __name__ == "__main__":
    random.seed(42)  # makes results reproducible from run to run

    games = DataLoader().load_games()
    builder = FeatureBuilder(prior_games=PRIOR_GAMES)
    features = builder.build(games)

    # Train on the whole regular season (split date after the season ends).
    model = GameModel(features=FEATURES, split_date="2027-01-01")
    train, _ = model.split(features)
    model.train(train)

    ratings = builder.team_ratings()
    records = {t: s["wins"] / s["games"] for t, s in builder.teams.items()}
    check_names(SEEDS, set(ratings.index))

    sim = PlayoffSimulator(SEEDS, model, ratings, records, RESULTS, IN_PROGRESS)
    table = sim.run(N_SIMS)

    pd.set_option("display.width", 120)
    print(f"Postseason odds from {N_SIMS:,} simulations:\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.1%}"))

    out = f"data/predictions_{date.today()}.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved to {out}. Commit it before the next games start!")