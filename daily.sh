#!/bin/zsh
# Daily routine: pull new results + lines, grade yesterday, retrain, print tonight's picks.
# Usage: ./daily.sh [budget]     e.g. ./daily.sh 100
set -e
cd "$(dirname "$0")"
python3 -u fetch_data.py 2027 > data/fetch.log   # this season's results + lines (cached)
python3 props.py fetch > data/props_fetch.log   # new box scores + prop lines
python3 grade_picks.py grade                    # score yesterday's picks
python3 backtest.py > data/backtest.log         # retrain/re-test, pick best model version
python3 picks.py ${1:+--budget $1}              # tonight's winners, top 5 points props, stake split
python3 grade_picks.py log                      # remember tonight's picks for tomorrow's grading
python3 build_ui.py                             # writes picks.html
