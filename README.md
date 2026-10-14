# Project02_ML: Predicting the 2026 MLB Postseason

A machine learning model that learns from the 2026 regular season and predicts the 2026 MLB playoffs, series by series, with predictions committed to GitHub *before* the games were played. The model's World Series odds are also compared against a real prediction market (Kalshi).

---

## How it works

The project has two stages:

1. **A game-level model** estimates the probability that the home team wins a single game, using each team's stats *before* that game. A season has about 2,400 games to learn from, while there's only one champion per year, so predicting games is far more learnable than predicting champions directly.
2. **A playoff simulator** uses those game probabilities to play out every remaining series with the real home/away patterns. It does this two ways:
   - **Exact bracket prediction:** for each series, it calculates the probability of every possible result (e.g., "Rays in 6") by walking through every sequence of wins and losses, picks the favorite, and advances them.
   - **Monte Carlo odds:** it plays the rest of the postseason 10,000 times with random draws and counts how often each team reaches each round.

As a check, the two methods agree to within about a percentage point (e.g., Rays win the ALCS 57.0% by exact calculation vs. 56.9% in simulation).

## Model versions

| Version | Features | Status |
|---|---|---|
| **V1** | Team run differential per game, last-10-games record (home minus away) | Official, frozen at Git tag `v1` |
| **V2** | V1 + starting pitcher quality (FIP) | Runs alongside V1 |

Both versions use logistic regression (scikit-learn). Switch between them with `MODEL_VERSION` in `playoff_simulator.py` and `market_comparison.py`.

---

## Files

| File | What it does |
|---|---|
| `data_loader.py` | Downloads all 2026 regular-season game results from MLB's official Stats API and caches them to CSV |
| `feature_builder.py` | Builds leak-free pre-game team features (V1) |
| `game_model.py` | Logistic regression model with a time-based train/test split, evaluation metrics, and baselines |
| `experiments.py` | Tunes V1's feature set and regression strength on a validation window |
| `pitcher_loader.py` | Downloads every game's starting pitchers from MLB box scores |
| `pitcher_features.py` | Adds the leak-free starting pitcher feature (V2) |
| `experiments_v2.py` | Compares V1 and V2 on the same games and splits |
| `cross_validation.py` | Monthly time-series cross-validation comparing V1 and V2 game by game |
| `playoff_simulator.py` | Predicts the bracket and simulates the postseason |
| `market_comparison.py` | Compares the model's title odds with Kalshi's market odds |
| `results_display.py` | Pygame window showing the latest predicted bracket and title odds; switch between V1 and V2, or to the from-the-start retrospective |
| `data/` | Cached data, experiment results, and every committed prediction snapshot |

## How to run

**Setup** (Python 3.14, in the project's virtual environment):

```
pip install -r requirements.txt
```

**Run in this order.** Data is cached in `data/`, so after the first run the scripts don't need to call the API again.

1. `data_loader.py`: download game results
2. `feature_builder.py` and `game_model.py`: build V1 features and check the model
3. `experiments.py`: V1 tuning
4. `pitcher_loader.py`: download starting pitchers (takes a few minutes; resumes if interrupted)
5. `pitcher_features.py`, `experiments_v2.py`, `cross_validation.py`: build and test V2
6. `playoff_simulator.py`: predictions (set `MODEL_VERSION`, and update `RESULTS` / `IN_PROGRESS` with series already played). Set `PRE_PLAYOFF = True` to ignore all results and predict the whole postseason from the start; those files are saved with a `preplayoff_` prefix so they never replace the regular predictions. Set it back to `False` afterward.
7. `market_comparison.py`: compare with Kalshi (update `MARKET_YES` and `MARKET_CAPTURED` with current prices)
8. `results_display.py`: view the latest predictions in a Pygame window (press 1/2 to switch models, P to switch to the from-the-start retrospective, S to save a screenshot)

---

## Key methods and decisions

**Only this season's data.** Rosters change a lot between seasons. The features are each team's stats *within* the season, so the model learns how much a stat gap matters, not which teams are good.

**No data leakage.** The features builder walks through the season in date order and records each team's (and pitcher's) stats *before* each game, then updates them with that game's result. The model never sees information from the future.

**Regression to the mean.** Early-season stats are mostly noise (a 3–0 team isn't a 1.000 team). Every team starts with imaginary games of .500 baseball (`prior_games`), and every pitcher starts with imaginary league-average innings (`prior_innings`). These fade as real games pile up.

**Time-based splits.** Data is split by date, never randomly, so the model is always tested on games that happened after the ones it learned from. Settings are chosen on a validation window (July 15–August 14), and the test set (August 15 on) is checked only once at the end.

**Time-series cross-validation.** One test window of ~590 games was too small to tell V1 and V2 apart, so V2 was also tested on five monthly windows (May–September, ~2,000 games), training only on games before each month. The versions were compared game by game on the same games, with standard errors.

**Playoff rotations.** Future playoff starters aren't known, so V2 rates each team by the average FIP of its top four starters by innings.

**Retrospective from the start.** Because the model only uses regular-season data, running the simulator with no playoff results entered (`PRE_PLAYOFF = True`) reproduces exactly what it would have predicted before the Wild Card round, including the Wild Card series my committed snapshots never fully predicted. This run was generated on Oct 14, after many games had been played, so it is labeled as a retrospective, not a prediction, in the file names and in the Pygame display. The committed snapshots remain the real evidence. V1's retrospective is a fair test, since V1 was tuned on regular-season data before the Wild Card round finished. V2's is weaker evidence, because I built V2 after seeing some playoff results.

**Market comparison.** Kalshi's prices add up to more than 100% (the market's built-in margin), so they're normalized to 100% before comparing.

### Decisions I made

- **Predicting the playoffs live.** I built the model as the 2026 postseason began and committed predictions before each round, so they could be checked against real results instead of hindsight.
- **Using only the 2026 season.** I chose not to use past seasons because of significant roster changes between years.
- **Full bracket predictions, not just odds.** The first simulator only gave each team's percentage chances. I wanted actual picks for every series, including the number of games, so I added the exact bracket prediction alongside the Monte Carlo odds.
- **Benchmarking against a betting market.** I wanted to compare my model's World Series odds with Kalshi's market prices, and captured the market odds myself.
- **Continuing to improve the model.** Rather than stopping at V1, I kept V1 frozen for consistent official predictions and built V2 (starting pitchers) alongside it.
- **Versioning.** I tagged V1 in Git (`v1`) and set up `MODEL_VERSION` so both versions can run side by side without overwriting each other's output.
- **Labeling snapshots honestly.** For the ALCS snapshot, I chose to show the NLCS as not yet started, and labeled the commit to say it was generated after NLCS Game 1 had been played.
- **Entering results.** I entered the bracket seeds and series results from MLB.com, and updated them after each round.

---

## Results

**Baselines** (test set, games from Aug. 15 on): always picking the home team scores 53.6% accuracy and 0.6908 log loss. Picking the team with the better record scores 58.8% accuracy.

**V1 tuning.** More regression was consistently better (`prior_games` = 70 beat 1, 20, and 40), and run differential predicted better than win-loss record. Final V1 test log loss: **0.6803** (accuracy 56.4%), better than the home-team baseline.

V1 is less accurate than "pick the better record" because home-field advantage tips its close calls toward the home team, but it wins on log loss, which measures the quality of the probabilities that the simulator relies on.

**V2 (starting pitchers).**

| Test | V1 log loss | V2 log loss | Winner |
|---|---|---|---|
| Validation window (Jul 15–Aug 14) | 0.6833 | 0.6860 | V1 |
| Test window (Aug 15 on) | 0.6801 | 0.6758 | V2 |
| Cross-validation, 1,960 games (May–Sep) | 0.6868 | 0.6853 | V2 (better in 4 of 5 months) |

Across cross-validation, V2 improved log loss by about 0.0014 per game (z ≈ −1.4 to −1.7): **suggestive but not conclusive.** V2 lost only in May, when pitchers had few starts, and its advantage was largest in September, which makes sense, since pitcher stats become reliable as innings accumulate.

**Context:** even professional models only reach about 57–60% accuracy on single MLB games. Baseball is extremely random.

## Prediction log

Every snapshot was committed to GitHub before the games it predicts. Commit timestamps are the proof.

| Snapshot | Model | Notes |
|---|---|---|
| Before NL Wild Card Game 3 (Oct 1) | V1 | Braves vs. Phillies |
| Before Division Series (Oct 2) | V1 | First full bracket prediction |
| Before ALCS Game 1 (Oct 12) | V1, V2 | NLCS shown as pre-series but generated after NLCS Game 1 was played |
| Retrospective from the start (generated Oct 14) | V1, V2 | **Not a prediction:** generated after the fact; saved as `preplayoff_` files |

**V2 predictions before ALCS Game 1:** Rays over Guardians in 6 (57.0%), Brewers over Dodgers in 6 (53.6%), Brewers win the World Series in 6. Title odds: Brewers 34.8%, Dodgers 29.5%, Rays 21.8%, Guardians 13.9%.

### Division Series report card (V1, predicted Oct 2)

| Series | V1 pick | Confidence | Actual winner | Result |
|---|---|---|---|---|
| ALDS: Rays vs. Yankees | Yankees in 4 | 53.1% | Rays | ✗ |
| ALDS: Guardians vs. White Sox | White Sox in 4 | 52.2% | Guardians (in 5, after trailing 0–2) | ✗ |
| NLDS: Brewers vs. Padres | Brewers in 4 | 64.7% | Brewers | ✓ |
| NLDS: Dodgers vs. Braves | Dodgers in 5 | 59.3% | Dodgers | ✓ |

**V1 went 2 for 4.** That's almost exactly what the model itself expected: adding up its confidence in each pick (0.531 + 0.522 + 0.647 + 0.593) gives 2.29 expected correct picks. Both misses were near coin flips (52–53%), while both hits were the model's strongest picks. That's how a well-calibrated model should behave: being wrong on 52% picks about half the time is expected, not a failure.

V1's full pre–Division Series bracket also picked the Yankees to win the AL pennant and the Brewers to beat the Yankees in the World Series in 6.

### Model vs. Kalshi (V1, before the Division Series)

Kalshi "Pro Baseball Champion" prices captured Oct 2, normalized to 100%:

| Team | V1 model | Kalshi | Difference |
|---|---|---|---|
| Los Angeles Dodgers | 19.9% | 28.5% | −8.7 |
| Milwaukee Brewers | 23.5% | 16.3% | +7.2 |
| New York Yankees | 14.1% | 15.0% | −0.9 |
| Tampa Bay Rays | 10.6% | 10.3% | +0.2 |
| San Diego Padres | 6.5% | 9.6% | −3.1 |
| Atlanta Braves | 9.9% | 8.2% | +1.7 |
| Cleveland Guardians | 6.6% | 7.1% | −0.5 |
| Chicago White Sox | 9.1% | 5.0% | +4.0 |

The biggest disagreements were at the top. The market made the Dodgers the clear favorite, while V1 preferred the Brewers, whose run differential was the best in baseball. V1 also liked the White Sox much more than the market did; they were eliminated by Cleveland. The market can price in things V1 can't see, like injuries and playoff rotations, which is part of why V2 adds starting pitching.

**Status at submission (Oct 14, 2026):** the Championship Series were still in progress, so LCS and World Series results aren't included here. Once the champion is known, setting `CHAMPION` in `market_comparison.py` scores both models against the market.

---

## Limitations and known issues

- **No injury or lineup information.** The model only sees regular-season stats. The market knows about injuries and planned rotations; the model doesn't.
- **Rotation estimate.** V2 uses each team's top four starters by innings, which may not match who actually starts in the playoffs.
- **Small samples.** Even cross-validation leaves the V2 improvement short of statistically conclusive, and one postseason is a single outcome, so whoever "wins" the market comparison may have been partly lucky.
- **Suspended game in V1.** Game 824912 (started June 16, finished June 17) appears twice in MLB's schedule data, so V1 counts it twice. This is 1 of 2,428 games and V1 was left unchanged to stay frozen; V2 counts it once, dated by completion.
- **LCS home field** is assigned to the better seed; this rule wasn't independently verified.

## Future improvements

- Add the 2023–2025 seasons for more training data and to backtest the simulator on past postseasons
- Try random forest or gradient boosting models
- Add bullpen quality as a feature
- Add a calibration plot (do games rated 60% actually go 60%?)
- Automatically grade committed predictions against actual results

---

## Data sources and resources

- **MLB Stats API** via the [MLB-StatsAPI](https://github.com/toddrob99/MLB-StatsAPI) Python library: game results and box scores
- **MLB.com**: postseason bracket and results
- **Kalshi**: "Pro Baseball Champion" market prices (screenshots, dated in `market_comparison.py`)

- **Claude (Anthropic)**: see Collaboration and AI use below

## Collaboration and AI use

The only AI I used was Claude (Anthropic), in a single ongoing chat. I did not use FlintK12. Each `.py` file also has an AI use note at the top.

**What Claude did:**
- **Planning:** suggested the two-stage design (a game-level model plus a playoff simulator), data sources (MLB Stats API, pybaseball), and the overall pipeline, and recommended an order of steps.
- **Drafting code:** wrote the first versions of every `.py` file in the project, and later wrote edits I applied, such as adding model versioning, adding V2 to the simulator, and fixing the market normalization.
- **Explaining concepts:** data leakage, regression to the mean, time-based splits, validation vs. test sets, log loss and Brier score, Monte Carlo simulation, FIP, time-series cross-validation, standard errors, and market normalization.
- **Interpreting results:** helped me read each run's output, including why V1 lost to "pick the better record" on accuracy but won on log loss, why the V1/V2 comparison was inconclusive at first, and what the month-by-month cross-validation showed.
- **Debugging:** a missing package in my virtual environment, a duplicated suspended game in MLB's schedule data, and a script saved in the wrong folder.
- **Looking up results:** searched the web for playoff results and the bracket while I updated predictions.
- **Git and documentation:** explained committing, pushing, and tagging in PyCharm; drafted commit messages; and drafted this README from our conversation.

**What I did:**
- Chose the project idea and the decisions listed under "Decisions I made."
- Set up the PyCharm project, virtual environment, and private GitHub repository, and added my teacher as a collaborator.
- Ran every script, checked each output for problems (for example, noticing results that needed explaining), and sent results back to decide on next steps.
- Entered the playoff bracket and results from MLB.com and captured the Kalshi market odds.
- Committed and pushed every version and prediction snapshot, and tagged V1.
- Reviewed and edited this README.