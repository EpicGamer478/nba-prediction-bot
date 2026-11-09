"""Bundle tonight.json + backtest results into markets.html (the market page)."""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    with open(os.path.join(HERE, "data", "tonight.json")) as f:
        data = json.load(f)
    bets = pd.read_csv(os.path.join(HERE, "data", "backtest_bets.csv")).sort_values("date")
    curves = {}
    for model, bm in bets.groupby("model"):
        curves[model] = {}
        for market, b in bm.groupby("market"):
            daily = b.groupby("date").profit.sum().cumsum()
            # One point per week keeps the page small.
            daily.index = pd.to_datetime(daily.index)
            weekly = daily.resample("W").last().dropna()
            curves[model][market] = [[d.date().isoformat(), round(v, 2)] for d, v in weekly.items()]
    data["curves"] = curves
    with open(os.path.join(HERE, "template.html")) as f:
        html = f.read()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    with open(os.path.join(HERE, "markets.html"), "w") as f:
        f.write(html.replace("__DATA__", blob))
    print("wrote markets.html")


if __name__ == "__main__":
    main()
