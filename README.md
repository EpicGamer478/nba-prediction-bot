# NBA prediction bot

Predicts NBA moneylines, spreads and totals, backtests against historical closing lines, and renders a clean
picks page (`picks.html`) with winners, top 5 points props and a budget splitter (quarter Kelly).

## How it works

- **Data** (`fetch_data.py`): game results and closing lines from ESPN's public API, 2019-20 to today.
- **Features** (`model.py`): Elo ratings (margin-of-victory adjusted, regressed between seasons),
  recent-form averages for scoring and defense, rest days, back-to-backs, and the betting line itself.
  Every feature uses only games played before the one being predicted.
- **Models**: three versions are trained and compared:
  - `stats_only`: ridge regression on stats, never sees the line
  - `market_blend`: starts at the book's line and learns a small adjustment (ridge)
  - `market_gbm`: same, with gradient-boosted trees
- **Backtest** (`backtest.py`): walk-forward. Each month is predicted by a model trained only on earlier
  games, then bet with flat 1-unit stakes at the closing line (spreads and totals at -110). The version with
  the most accurate win probabilities (lowest Brier score) is used for tonight's prices.
- **Props** (`props.py`): points projections from box scores, blended with DraftKings prices; top 5 by expected value.
  tonight's slate, rebuild the page.

## Run

```bash
pip install pandas numpy scikit-learn scipy certifi
./daily.sh
open markets.html
```

## Caveats

Closing lines are very efficient. A model that matches them is doing well; consistently beating them is rare.
Treat the picks as an experiment and check `data/picks_log.csv` for the live record before trusting them.
