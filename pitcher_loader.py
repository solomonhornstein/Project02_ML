"""
pitcher_loader.py - Downloads each game's starting pitchers from MLB box scores.

For every regular-season game, MLB's API has a box score listing the pitchers
each team used, in order. The first one is the starter. We save one row per
team per game with the starter's line: outs recorded, strikeouts, walks,
hit batters, home runs, and earned runs. These are the ingredients for FIP.

DOWNLOAD TIME: there are ~2,400 box scores. We download several at once
(threads) and save progress every 200 games, so if it's interrupted, just rerun
and it picks up where it left off.

OPENERS: some teams start a reliever for an inning or two (an "opener").
Those rows are kept, but the starter's outs show it. Worth deciding later
whether to treat very short starts differently.

AI USE NOTE (required by DA Honor Code): This class was drafted with help from
Claude (Anthropic). [Add your own notes on what you changed or wrote yourself.]
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import statsapi

from data_loader import DataLoader

TEST_RUN = False  # True = fetch only 5 games to check it works; then set False


class PitcherLoader:
    """Downloads and caches each game's starting pitcher lines."""

    def __init__(self, cache_path="data/starters_2026.csv", workers=8):
        self.cache_path = cache_path
        self.workers = workers  # how many box scores to download at once

    def load_starters(self, games, refresh=False):
        """Return a DataFrame with one row per team per game (the starter's line)."""
        done = pd.DataFrame()
        if os.path.exists(self.cache_path) and not refresh:
            done = pd.read_csv(self.cache_path)
            done = done.drop_duplicates(["game_id", "side"])
        have = set(done["game_id"]) if len(done) else set()
        todo = [g for g in games["game_id"].unique() if g not in have]
        print(f"{len(have)} games cached, {len(todo)} to download")

        rows, failed = [], []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._fetch_game, gid): gid for gid in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    rows.extend(fut.result())
                except Exception as e:  # keep going; failed games are retried next run
                    failed.append((futures[fut], repr(e)))
                if i % 200 == 0:
                    print(f"  {i}/{len(todo)} downloaded...")
                    done = self._save(done, rows)
                    rows = []

        done = self._save(done, rows)
        if failed:
            print(f"{len(failed)} games failed (rerun to retry). First error: {failed[0]}")

        # Attach game dates so features can be built in date order.
        # A suspended game appears on both dates; use the date it finished,
        # since the result wasn't known until then (keeps features leak-free).
        dates = games[["game_id", "game_date"]].drop_duplicates("game_id", keep="last")
        return done.merge(dates, on="game_id", how="left")

    def _save(self, done, rows):
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            done.to_csv(self.cache_path, index=False)
        return done

    @staticmethod
    def _fetch_game(game_id):
        """Download one box score and return the two starters' lines."""
        box = statsapi.get("game_boxscore", {"gamePk": int(game_id)})
        rows = []
        for side in ("home", "away"):
            team = box["teams"][side]
            starter_id = team["pitchers"][0]          # first pitcher used = starter
            player = team["players"][f"ID{starter_id}"]
            p = player["stats"]["pitching"]
            rows.append({
                "game_id": game_id,
                "side": side,
                "team": team["team"]["name"],
                "pitcher_id": starter_id,
                "pitcher_name": player["person"]["fullName"],
                "outs": PitcherLoader._ip_to_outs(p.get("inningsPitched", "0.0")),
                "strikeouts": p.get("strikeOuts", 0),
                "walks": p.get("baseOnBalls", 0),
                "hit_by_pitch": p.get("hitByPitch", 0),
                "home_runs": p.get("homeRuns", 0),
                "earned_runs": p.get("earnedRuns", 0),
            })
        return rows

    @staticmethod
    def _ip_to_outs(ip):
        """Innings pitched like '5.2' means 5 innings + 2 outs = 17 outs."""
        whole, _, frac = str(ip).partition(".")
        return int(whole or 0) * 3 + int(frac or 0)


if __name__ == "__main__":
    games = DataLoader().load_games()
    if TEST_RUN:
        loader = PitcherLoader(cache_path="data/starters_test.csv")
        starters = loader.load_starters(games.head(5), refresh=True)
    else:
        starters = PitcherLoader().load_starters(games)

    print(f"\n{len(starters)} starter rows")
    print(starters.head(10).to_string(index=False))
    print(f"\nAverage outs per start: {starters['outs'].mean():.1f}")