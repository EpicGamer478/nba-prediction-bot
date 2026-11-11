#!/bin/zsh
# Daily routine: pull new results + lines, grade yesterday, retrain, print tonight's picks.
# Usage: ./daily.sh [budget]     e.g. ./daily.sh 100
set -e
cd "$(dirname "$0")"
python3 -u fetch_data.py 2026 > data/fetch.log   # this season's results + lines (cached)
CACHED_ONLY_OLD=1 python3 props.py fetch > data/props_fetch.log   # new box scores + prop lines
python3 grade_picks.py grade                    # score yesterday's picks
python3 backtest.py > data/backtest.log         # retrain/re-test, pick best model version
if [[ -n "$1" ]]; then python3 picks.py --budget "$1"; else python3 picks.py; fi   # winners, top 5 props, stake split
python3 parlay.py                               # double-up parlay with live Polymarket prices
python3 grade_picks.py log                      # remember tonight's picks for tomorrow's grading
python3 build_ui.py                             # writes picks.html
