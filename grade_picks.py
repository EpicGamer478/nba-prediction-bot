"""Log tonight's leans and grade past ones against final scores.

- `python3 grade_picks.py log`   appends picks from data/tonight.json to data/picks_log.csv
- `python3 grade_picks.py grade` fills in results for finished games and prints the running record
"""
import json
import os
import sys

import pandas as pd

from model import american_to_payout

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "data", "picks_log.csv")


def log():
    with open(os.path.join(HERE, "data", "tonight.json")) as f:
        t = json.load(f)
    rows = [{"date": t["date"], "id": g["id"], "game": f"{g['away']} @ {g['home']}", "market": p["market"],
             "side": p["side"], "edge": p["edge"], "spread": g["spread"], "total": g["total"],
             "home_ml": g["home_ml"], "away_ml": g["away_ml"], "model": t.get("model_used")}
            for g in t["games"] for p in g["picks"]]
    new = pd.DataFrame(rows)
    if os.path.exists(LOG):
        old = pd.read_csv(LOG, dtype={"id": str})
        new = pd.concat([old[old.date != t["date"]], new])  # rerunning a day replaces its picks
    new.to_csv(LOG, index=False)
    print(f"logged {len(rows)} picks for {t['date']}")


def grade():
    if not os.path.exists(LOG):
        print("no picks logged yet")
        return
    picks = pd.read_csv(LOG, dtype={"id": str})
    games = pd.read_csv(os.path.join(HERE, "data", "games.csv"), dtype={"id": str})
    fin = games[games.completed].set_index("id")
    profits = []
    for r in picks.itertuples():
        if r.id not in fin.index:
            profits.append(None)
            continue
        g = fin.loc[r.id]
        m, tot = g.home_pts - g.away_pts, g.home_pts + g.away_pts
        if r.market == "moneyline":
            won, ml = (m > 0, r.home_ml) if r.side == "home" else (m < 0, r.away_ml)
            profits.append(american_to_payout(ml) if won else -1.0)
        else:
            res = (m + r.spread) * (1 if r.side == "home" else -1) if r.market == "spread" else \
                  (tot - r.total) * (1 if r.side == "over" else -1)
            profits.append(0.0 if res == 0 else (100 / 110 if res > 0 else -1.0))
    picks["profit"] = profits
    picks.to_csv(LOG, index=False)
    done = picks.dropna(subset=["profit"])
    print(f"Graded {len(done)} of {len(picks)} picks")
    if len(done):
        print(done.groupby("market").profit.agg(bets="count", units="sum", roi="mean").round(3))


if __name__ == "__main__":
    {"log": log, "grade": grade}[sys.argv[1] if len(sys.argv) > 1 else "grade"]()
