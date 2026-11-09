#!/bin/zsh
# Daily routine: pull new results + lines, grade yesterday, retrain/re-backtest, price tonight, rebuild the page.
set -e
cd "$(dirname "$0")"
python3 -u fetch_data.py            # cached, so only new days/games hit the network
python3 grade_picks.py grade        # score yesterday's leans against final results
python3 backtest.py                 # re-run walk-forward test; picks the best model version
python3 predict.py                  # retrain on everything through last night, price tonight
python3 grade_picks.py log          # remember tonight's leans for tomorrow's grading
python3 build_page.py               # writes markets.html
