"""
results_display.py - A Pygame window showing the latest playoff predictions.

Shows the predicted bracket (every series, with the pick and its confidence)
and a bar chart of each remaining team's chance to win the World Series.
It reads the most recent prediction files saved by playoff_simulator.py, so
run that first.

CONTROLS:
  1 / 2   switch between model V1 and V2
  S       save a screenshot (data/screenshot_<version>.png) for slides
  Esc     quit

Setup:  pip install pygame   (if that fails on Python 3.14: pip install pygame-ce;
        the code is the same, both use "import pygame")

AI USE NOTE (required by DA Honor Code): This class was drafted with help from
Claude (Anthropic).
"""

import glob

import pandas as pd
import pygame

WIDTH, HEIGHT = 1280, 800
ROUNDS = ["Wild Card", "Division Series", "Championship Series", "World Series"]

# Colors
BG = (18, 22, 30)
CARD = (34, 40, 54)
CARD_FINAL = (28, 32, 42)
TEXT = (230, 233, 240)
MUTED = (130, 138, 155)
PICK = (98, 200, 140)       # predicted winner
WINNER = (120, 170, 255)    # actual winner of a finished series
BAR = (98, 200, 140)
ACCENT = (255, 196, 80)


class ResultsDisplay:
    """Draws the predicted bracket and title odds for one model version."""

    def __init__(self, version="v1"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("2026 MLB Postseason Predictions")
        self.title_font = pygame.font.SysFont(None, 40)
        self.head_font = pygame.font.SysFont(None, 26)
        self.font = pygame.font.SysFont(None, 21)
        self.small = pygame.font.SysFont(None, 18)
        self.prefix = ""          # "" = regular predictions, "preplayoff_" = from the start
        self.load(version)

    # ---- data ----

    def load(self, version):
        """Load the most recent bracket and odds files for this version."""
        self.version = version
        self.bracket, self.bracket_file = self._latest(f"data/{self.prefix}bracket_{version}_*.csv")
        self.odds, self.odds_file = self._latest(f"data/{self.prefix}predictions_{version}_*.csv")

    @staticmethod
    def _latest(pattern):
        files = sorted(glob.glob(pattern))   # dates in the names sort correctly
        if not files:
            return None, None
        return pd.read_csv(files[-1]), files[-1]

    # ---- drawing ----

    def draw(self):
        self.screen.fill(BG)
        title = f"2026 MLB Postseason Predictions  -  Model {self.version.upper()}"
        if self.prefix:
            title += "  -  From the start (generated after the fact)"
        self._text(title, self.title_font, TEXT, 30, 22)
        self._text("1 / 2: switch model    P: from-the-start view    S: screenshot    Esc: quit",
                   self.small, MUTED, 30, 58)
        self._text("Blue = actual winner", self.small, WINNER, WIDTH - 330, 58)
        self._text("Green = predicted winner", self.small, PICK, WIDTH - 185, 58)

        if self.bracket is None or self.odds is None:
            self._text(f"No {self.version} prediction files found in data/. "
                       "Run playoff_simulator.py first.", self.head_font, ACCENT, 30, 120)
        else:
            self._draw_bracket()
            self._draw_odds()
            source = f"Files: {self.bracket_file}, {self.odds_file}"
            self._text(source, self.small, MUTED, 30, HEIGHT - 26)
        pygame.display.flip()

    def _draw_bracket(self):
        col_w, card_w, card_h = 310, 292, 74
        top, area_h = 95, 420
        for c, rnd in enumerate(ROUNDS):
            x = 30 + c * col_w
            self._text(rnd, self.head_font, ACCENT, x, top)
            series = self.bracket[self.bracket["round"] == rnd]
            n = len(series)
            gap = (area_h - n * card_h) / (n + 1) if n else 0
            for i, row in enumerate(series.itertuples(index=False)):
                y = top + 30 + gap + i * (card_h + gap)
                self._draw_card(row, x, y, card_w, card_h)

    def _draw_card(self, row, x, y, w, h):
        final = row.prediction.endswith("(final)")
        pygame.draw.rect(self.screen, CARD_FINAL if final else CARD,
                         (x, y, w, h), border_radius=8)

        matchup = row.matchup.split(" (series at")[0]
        teams = matchup.split(" vs. ")
        winner = row.prediction.replace(" (final)", "").split(" in ")[0]

        for j, team in enumerate(teams):
            is_winner = team.endswith(winner)
            color = (WINNER if final else PICK) if is_winner else (MUTED if final else TEXT)
            marker = "> " if is_winner else "  "
            self._text(marker + team, self.font, color, x + 10, y + 8 + j * 22)

        if final:
            note = "Final"
        else:
            games = row.prediction.split(" in ")[-1]
            note = f"Predicted: wins in {games}  ({row.p_win_series:.0%} to win series)"
        self._text(note, self.small, ACCENT if not final else MUTED, x + 10, y + h - 20)

    def _draw_odds(self):
        top = 545
        self._text("Chance to win the World Series (10,000 simulations)",
                   self.head_font, ACCENT, 30, top)
        alive = self.odds[self.odds["P(Champion)"] > 0].sort_values(
            "P(Champion)", ascending=False)
        if alive.empty:
            return
        best = alive["P(Champion)"].max()
        bar_x, bar_max = 290, 820
        row_h = min(26, (HEIGHT - top - 80) // len(alive))
        for i, row in enumerate(alive.itertuples(index=False)):
            y = top + 36 + i * row_h
            p = alive.iloc[i]["P(Champion)"]
            self._text(f"({row.league} {row.seed}) {row.team}", self.font, TEXT, 30, y)
            length = max(2, int(bar_max * p / best))
            pygame.draw.rect(self.screen, BAR, (bar_x, y + 2, length, row_h - 8),
                             border_radius=4)
            self._text(f"{p:.1%}", self.font, TEXT, bar_x + length + 10, y)

    def _text(self, s, font, color, x, y):
        self.screen.blit(font.render(s, True, color), (x, y))

    # ---- main loop ----

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_1:
                        self.load("v1")
                    elif event.key == pygame.K_2:
                        self.load("v2")
                    elif event.key == pygame.K_p:
                        self.prefix = "" if self.prefix else "preplayoff_"
                        self.load(self.version)
                    elif event.key == pygame.K_s:
                        path = f"data/screenshot_{self.prefix}{self.version}.png"
                        pygame.image.save(self.screen, path)
                        print(f"Saved {path}")
            self.draw()
            clock.tick(30)
        pygame.quit()


if __name__ == "__main__":
    ResultsDisplay(version="v1").run()