"""Walk-forward backtest against historical closing lines.

For each month, models are trained only on games before that month, then used to
bet that month's games. Flat 1-unit stakes. Spreads/totals assumed at -110.
Writes data/backtest.json (summary) and data/backtest_bets.csv (every bet).
"""
import json
import os

import numpy as np
import pandas as pd

from model import Models, american_to_payout, american_to_prob, build_features

HERE = os.path.dirname(os.path.abspath(__file__))
STD_PAYOUT = 100 / 110
BREAKEVEN = 110 / 210
THRESHOLDS = [0.0, 0.02, 0.04, 0.06, 0.08]


VARIANTS = {"stats_only": (False, "ridge"), "market_blend": (True, "ridge"), "market_gbm": (True, "gbm")}


def walk_forward(df, use_market, kind="ridge", first_season=2021):
    df = df.copy()
    df["month"] = df.date.dt.to_period("M")
    test_months = sorted(df[(df.season >= first_season) & df.completed].month.unique())
    preds = []
    for m in test_months:
        train = df[df.date < m.start_time]
        test = df[(df.month == m) & df.completed & (df.min_games >= 5)]
        test = test.dropna(subset=["spread", "total", "home_ml", "away_ml"])
        if len(test) == 0:
            continue
        preds.append(pd.concat([test, Models(use_market, kind).fit(train).predict(test)], axis=1))
    return pd.concat(preds)


def make_bets(p, thr):
    bets = []
    for r in p.itertuples():
        # Moneyline: bet the side with the most expected value.
        for side, prob, ml, won in (("home", r.p_home_win, r.home_ml, r.margin > 0),
                                    ("away", 1 - r.p_home_win, r.away_ml, r.margin < 0)):
            ev = prob * american_to_payout(ml) - (1 - prob)
            if ev > thr:
                bets.append(("moneyline", r, side, ev, american_to_payout(ml) if won else -1.0))
        # Spread
        cover = r.margin + r.spread
        for side, prob, res in (("home", r.p_home_cover, cover), ("away", 1 - r.p_home_cover, -cover)):
            if prob - BREAKEVEN > thr and res != 0:
                bets.append(("spread", r, side, prob - BREAKEVEN, STD_PAYOUT if res > 0 else -1.0))
        # Total
        diff = r.total_pts - r.total
        for side, prob, res in (("over", r.p_over, diff), ("under", 1 - r.p_over, -diff)):
            if prob - BREAKEVEN > thr and res != 0:
                bets.append(("total", r, side, prob - BREAKEVEN, STD_PAYOUT if res > 0 else -1.0))
    return pd.DataFrame([{"market": m, "date": r.date.date().isoformat(), "season": r.season,
                          "game": f"{r.away} @ {r.home}", "side": s, "edge": e, "profit": pl}
                         for m, r, s, e, pl in bets])


def summarize(b):
    if len(b) == 0:
        return {"bets": 0, "win_rate": None, "profit": 0, "roi": None}
    return {"bets": int(len(b)), "win_rate": round(float((b.profit > 0).mean()), 4),
            "profit": round(float(b.profit.sum()), 2), "roi": round(float(b.profit.mean()), 4)}


def accuracy(p):
    vig_free = p.home_ml.map(american_to_prob) / (
        p.home_ml.map(american_to_prob) + p.away_ml.map(american_to_prob))
    y = (p.margin > 0).astype(int)
    return {
        "games": int(len(p)),
        "brier_model": round(float(((p.p_home_win - y) ** 2).mean()), 4),
        "brier_market": round(float(((vig_free - y) ** 2).mean()), 4),
        "brier_elo": round(float(((p.elo_prob - y) ** 2).mean()), 4),
        "winner_acc_model": round(float(((p.p_home_win > 0.5) == y).mean()), 4),
        "winner_acc_market": round(float(((vig_free > 0.5) == y).mean()), 4),
        "margin_mae_model": round(float((p.pred_margin - p.margin).abs().mean()), 2),
        "margin_mae_market": round(float((-p.spread - p.margin).abs().mean()), 2),
        "total_mae_model": round(float((p.pred_total - p.total_pts).abs().mean()), 2),
        "total_mae_market": round(float((p.total - p.total_pts).abs().mean()), 2),
    }


def main():
    games = pd.read_csv(os.path.join(HERE, "data", "games.csv"))
    df = build_features(games)
    report = {}
    all_bets = []
    # Test on every season that has at least one earlier season of lines to train on.
    seasons_with_odds = sorted(df[df.spread.notna()].season.unique())
    first = int(seasons_with_odds[1]) if len(seasons_with_odds) > 1 else int(seasons_with_odds[0])
    for name, (use_market, kind) in VARIANTS.items():
        p = walk_forward(df, use_market, kind, first)
        rep = {"accuracy": accuracy(p), "by_threshold": {}, "by_season": {},
               "acc_by_season": {str(s): accuracy(ps) for s, ps in p.groupby("season")}}
        for thr in THRESHOLDS:
            b = make_bets(p, thr)
            rep["by_threshold"][str(thr)] = {m: summarize(b[b.market == m]) for m in
                                             ("moneyline", "spread", "total")}
        b = make_bets(p, 0.04)
        b["model"] = name
        all_bets.append(b)
        for s, bs in b.groupby("season"):
            rep["by_season"][str(s)] = {m: summarize(bs[bs.market == m]) for m in
                                        ("moneyline", "spread", "total")}
        report[name] = rep

        print(f"\n===== {name} =====")
        print(json.dumps(rep["accuracy"], indent=1))
        print("Edge threshold -> bets / win% / ROI")
        for thr, mk in rep["by_threshold"].items():
            print(f"  {float(thr):.0%}: " + " | ".join(
                f"{m}: {v['bets']} / {v['win_rate'] or 0:.1%} / {v['roi'] or 0:+.1%}" for m, v in mk.items()))
        print("By season at 4% edge (ROI):")
        for s, mk in rep["by_season"].items():
            print(f"  {s}: " + " | ".join(f"{m} {v['bets']} {v['roi'] or 0:+.1%}" for m, v in mk.items()))

    # Pick the variant whose win probabilities were most accurate (lowest Brier score).
    report["best"] = min(VARIANTS, key=lambda k: report[k]["accuracy"]["brier_model"])
    print("\nBest variant:", report["best"])
    with open(os.path.join(HERE, "data", "backtest.json"), "w") as f:
        json.dump(report, f, indent=1)
    pd.concat(all_bets).to_csv(os.path.join(HERE, "data", "backtest_bets.csv"), index=False)


if __name__ == "__main__":
    main()
