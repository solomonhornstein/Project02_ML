"""
data_loader.py - Fetches MLB game results from MLB's official Stats API.

AI USE NOTE (required by DA Honor Code): The structure of this class was
drafted with help from Claude (Anthropic).

Setup:  pip install MLB-StatsAPI pandas
"""

import os
from datetime import date, timedelta

import pandas as pd
import statsapi


class DataLoader:
    """Downloads one season's regular-season games and caches them to CSV."""

    def __init__(self, season=2026, cache_path="data/games_2026.csv"):
        self.season = season
        self.cache_path = cache_path

    def load_games(self, refresh=False):
        """Return a DataFrame of completed regular-season games.

        Uses the cached CSV if it exists (so you don't hit the API every run
        and your data is committed to GitHub). Pass refresh=True to re-download.
        """
        if os.path.exists(self.cache_path) and not refresh:
            return pd.read_csv(self.cache_path, parse_dates=["game_date"])

        games = self._fetch_season()
        df = pd.DataFrame(games)

        # Keep only regular-season ('R') games that actually finished.
        df = df[(df["game_type"] == "R") & (df["status"] == "Final")]

        # Keep the columns we need for modeling.
        df = df[[
            "game_id", "game_date", "doubleheader", "game_num",
            "home_name", "away_name", "home_score", "away_score",
            "home_probable_pitcher", "away_probable_pitcher",
        ]].copy()

        df["game_date"] = pd.to_datetime(df["game_date"])
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
        df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df.to_csv(self.cache_path, index=False)
        return df

    def _fetch_season(self):
        """Request the schedule one month at a time (smaller, safer requests)."""
        start = date(self.season, 3, 1)       # spring training is filtered out later
        end = min(date(self.season, 10, 1), date.today())
        all_games = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + timedelta(days=30), end)
            print(f"Fetching {chunk_start} to {chunk_end}...")
            all_games.extend(statsapi.schedule(
                start_date=chunk_start.strftime("%Y-%m-%d"),
                end_date=chunk_end.strftime("%Y-%m-%d"),
            ))
            chunk_start = chunk_end + timedelta(days=1)
        return all_games


if __name__ == "__main__":
    loader = DataLoader()
    games = loader.load_games()
    print(f"{len(games)} games loaded")
    print(games.head())
    print(f"Home team win rate: {games['home_win'].mean():.3f}")  # your first baseline