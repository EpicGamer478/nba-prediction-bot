"""Predict tonight's games with the latest lines. Writes data/tonight.json.

Usage: python3 predict.py [YYYYMMDD]
"""
import json
import os
import sys
from datetime import date, datetime

import pandas as pd

import fetch_data as fd
from model import (Models, american_to_payout, american_to_prob, build_features,
                   prob_to_american)

HERE = os.path.dirname(os.path.abspath(__file__))
BREAKEVEN = 110 / 210


def main(day=None):
    day = day or date.today()
    ds = day.strftime("%Y%m%d")
    raw = fd.get_json(fd.SCOREBOARD.format(d=ds), f"sb_{ds}.json", use_cache=False)
    meta = {}
    for e in raw.get("events", []):
        c = e["competitions"][0]
        t = {x["homeAway"]: x["team"] for x in c["competitors"]}
        sc = {x["homeAway"]: int(x.get("score") or 0) for x in c["competitors"]}
        meta[e["id"]] = {"tip": e["date"], "status": e["status"]["type"]["shortDetail"],
                         "final": bool(e["status"]["type"].get("completed")),
                         "home_pts": sc["home"], "away_pts": sc["away"],
                         "home_name": t["home"]["displayName"], "away_name": t["away"]["displayName"],
                         "home_logo": t["home"].get("logo"), "away_logo": t["away"].get("logo"),
                         "home_color": t["home"].get("color"), "away_color": t["away"].get("color")}

    today = [{**g, **fd.parse_odds(fd.get_json(fd.ODDS.format(i=g["id"]), f"odds_live_{g['id']}.json",
                                               use_cache=False))}
             for g in fd.scoreboard_games(day)]
    hist = pd.read_csv(os.path.join(HERE, "data", "games.csv"))
    hist = hist[hist.completed & (pd.to_datetime(hist.date).dt.date < day)]
    tonight = pd.DataFrame(today)
    tonight["completed"] = False
    df = build_features(pd.concat([hist, tonight], ignore_index=True))
    with open(os.path.join(HERE, "data", "backtest.json")) as f:
        best = json.load(f).get("best", "market_blend")
    from backtest import VARIANTS
    model = Models(*VARIANTS[best]).fit(df[df.completed])
    pure = Models(use_market=False).fit(df[df.completed])

    t = df[df.id.astype(str).isin(tonight.id.astype(str))].copy()
    t = pd.concat([t, model.predict(t)], axis=1)
    t["pure_margin"] = pure.predict(t)["pred_margin"]

    with open(os.path.join(HERE, "data", "backtest.json")) as f:
        bt = json.load(f)

    games = []
    for r in t.itertuples():
        m = meta.get(str(r.id), {})
        g = {"id": str(r.id), "home": r.home, "away": r.away, **m,
             "elo_home": round(r.elo_home), "elo_away": round(r.elo_away),
             "spread": r.spread, "total": r.total, "home_ml": r.home_ml, "away_ml": r.away_ml,
             "pred_margin": round(r.pred_margin, 1), "pure_margin": round(r.pure_margin, 1),
             "pred_total": round(r.pred_total, 1),
             "p_home_win": round(r.p_home_win, 4), "p_home_cover": round(r.p_home_cover, 4),
             "p_over": round(r.p_over, 4), "home_b2b": int(r.home_b2b), "away_b2b": int(r.away_b2b),
             "picks": []}
        if pd.notna(r.home_ml) and pd.notna(r.away_ml):
            ih, ia = american_to_prob(r.home_ml), american_to_prob(r.away_ml)
            g["mkt_p_home_win"] = round(ih / (ih + ia), 4)
            for side, p, ml in (("home", r.p_home_win, r.home_ml), ("away", 1 - r.p_home_win, r.away_ml)):
                ev = p * american_to_payout(ml) - (1 - p)
                if ev > 0.02:
                    g["picks"].append({"market": "moneyline", "side": side, "edge": round(ev, 4),
                                       "odds": ml, "fair": prob_to_american(p)})
        if pd.notna(r.spread):
            for side, p in (("home", r.p_home_cover), ("away", 1 - r.p_home_cover)):
                if p - BREAKEVEN > 0.02:
                    g["picks"].append({"market": "spread", "side": side, "edge": round(p - BREAKEVEN, 4)})
        if pd.notna(r.total):
            for side, p in (("over", r.p_over), ("under", 1 - r.p_over)):
                if p - BREAKEVEN > 0.02:
                    g["picks"].append({"market": "total", "side": side, "edge": round(p - BREAKEVEN, 4)})
        games.append(g)

    out = {"date": day.isoformat(), "model_used": best, "generated": datetime.now().isoformat(timespec="minutes"),
           "margin_sd": round(model.margin_sd, 2), "total_sd": round(model.total_sd, 2),
           "games": sorted(games, key=lambda g: g.get("tip", "")), "backtest": bt}
    with open(os.path.join(HERE, "data", "tonight.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)

    for g in out["games"]:
        print(f"{g['away']:>4} @ {g['home']:<4} line {g['home']} {g['spread']:+} / o/u {g['total']} | "
              f"model margin {g['pred_margin']:+.1f} total {g['pred_total']:.1f} "
              f"P(home win) {g['p_home_win']:.0%} vs mkt {g.get('mkt_p_home_win', float('nan')):.0%}")
        for p in g["picks"]:
            print(f"      lean: {p['market']} {p['side']} (edge {p['edge']:+.1%})")


if __name__ == "__main__":
    main(datetime.strptime(sys.argv[1], "%Y%m%d").date() if len(sys.argv) > 1 else None)
