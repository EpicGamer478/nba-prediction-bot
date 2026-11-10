"""Bundle data/picks.json and the backtest results into picks.html (the picks page)."""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def backtest_payload():
    with open(os.path.join(HERE, "data", "backtest.json")) as f:
        rep = json.load(f)
    best = rep["best"]
    r = rep[best]
    out = {"model": best, "accuracy": r["accuracy"], "acc_by_season": r.get("acc_by_season", {}),
           "by_threshold": r["by_threshold"], "variants": {
               k: rep[k]["accuracy"] for k in rep if k != "best"}}
    # Running profit of every bet the model would have made (4%+ edge, 1 unit each).
    b = pd.read_csv(os.path.join(HERE, "data", "backtest_bets.csv"))
    b = b[b.model == best]
    if len(b):
        daily = b.groupby("date").agg(profit=("profit", "sum"), bets=("profit", "size")).reset_index()
        daily["cum"] = daily.profit.cumsum().round(2)
        daily["n"] = daily.bets.cumsum()
        out["curve"] = daily[["date", "cum", "n"]].values.tolist()
    p = os.path.join(HERE, "data", "props_backtest.json")
    if os.path.exists(p):
        with open(p) as f:
            out["props"] = json.load(f)
    return out


with open(os.path.join(HERE, "data", "picks.json")) as f:
    data = json.load(f)
data["bt"] = backtest_payload()
with open(os.path.join(HERE, "ui_template.html")) as f:
    html = f.read()
with open(os.path.join(HERE, "picks.html"), "w") as f:
    f.write(html.replace("__DATA__", json.dumps(data, separators=(",", ":"), default=float).replace("</", "<\\/")))
print("wrote picks.html")
