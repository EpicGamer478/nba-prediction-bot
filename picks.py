"""Tonight's NBA picks: who wins each game, the top 5 player points props,
and (optionally) how to split a betting budget across them.

Usage:
  python3 picks.py                 # winners + top 5 points props
  python3 picks.py --budget 100    # also split $100 across the bets
"""
import argparse
import contextlib
import io
import json
import os
from datetime import date

import props
import predict
from backtest import LOCK
from model import american_to_payout, american_to_prob

HERE = os.path.dirname(os.path.abspath(__file__))
KELLY_FRACTION = 0.25   # quarter Kelly: much less variance, most of the growth
MARKET_WEIGHT = 0.5     # shrink model probabilities halfway to the book's (backtest edge is unproven)
MAX_PER_BET = 0.15      # never put more than 15% of the budget on one bet


def kelly(p, american):
    """Fraction of bankroll the Kelly criterion says to bet (0 if no edge)."""
    b = american_to_payout(american)
    return max(0.0, (b * p - (1 - p)) / b)


def allocate(bets, budget):
    for bt in bets:
        bt["frac"] = min(KELLY_FRACTION * kelly(bt["p"], bt["odds"]), MAX_PER_BET)
    total = sum(bt["frac"] for bt in bets)
    scale = 1 / total if total > 1 else 1  # never exceed the budget
    for bt in bets:
        bt["stake"] = round(budget * bt["frac"] * scale, 2)
        bt["ev"] = bt["stake"] * (bt["p"] * american_to_payout(bt["odds"]) - (1 - bt["p"]))
    return bets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, help="total amount you're willing to bet tonight")
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        predict.main()
    with open(os.path.join(HERE, "data", "tonight.json")) as f:
        t = json.load(f)

    print(f"\nNBA PICKS — {date.today():%A %b %d}\n")
    print("WINNERS")
    bets = []
    for g in sorted(t["games"], key=lambda g: -max(g["p_home_win"], 1 - g["p_home_win"])):
        home = g["p_home_win"] >= 0.5
        win = g["home_name"] if home else g["away_name"]
        p_model = g["p_home_win"] if home else 1 - g["p_home_win"]
        tag = "  LOCK" if p_model >= LOCK else ""
        print(f"  {win:<24} {p_model:>4.0%}   ({g['away']} @ {g['home']}){tag}")
        if g.get("home_ml") is not None and g.get("away_ml") is not None:
            ml = g["home_ml"] if home else g["away_ml"]
            ph, pa = american_to_prob(g["home_ml"]), american_to_prob(g["away_ml"])
            p_mkt = (ph if home else pa) / (ph + pa)
            p = (1 - MARKET_WEIGHT) * p_model + MARKET_WEIGHT * p_mkt
            bets.append({"label": f"{win} to win", "odds": ml, "p": p})

    locks = [w for w in t["games"] if max(w["p_home_win"], 1 - w["p_home_win"]) >= LOCK]
    print(f"\nLOCKS ({LOCK:.0%}+ confidence): {len(locks)} tonight" if locks else f"\nNo locks ({LOCK:.0%}+) tonight.")

    print("\nTOP 5 POINTS PROPS")
    top = props.today()
    if top.empty:
        print("  No points props posted yet.")
    for r in top.itertuples():
        print(f"  {r.name:<24} {r.side.upper():<5} {r.line:<5} pts ({int(r.odds):+d})   "
              f"projects {r.proj:.1f}, {r.p_side:.0%} to hit")
        bets.append({"label": f"{r.name} {r.side} {r.line} pts", "odds": r.odds, "p": r.p_side})

    out = {"date": t["date"], "generated": t["generated"],
           "kelly_fraction": KELLY_FRACTION, "max_per_bet": MAX_PER_BET,
           "winners": [], "props": []}
    for g in t["games"]:
        home = g["p_home_win"] >= 0.5
        b = next((x for x in bets if x["label"] == f"{g['home_name'] if home else g['away_name']} to win"), None)
        out["winners"].append({
            "id": g["id"], "tip": g.get("tip"), "status": g.get("status"),
            "team": g["home_name"] if home else g["away_name"], "abbr": g["home"] if home else g["away"],
            "color": g.get("home_color") if home else g.get("away_color"),
            "opp": g["away_name"] if home else g["home_name"], "opp_abbr": g["away"] if home else g["home"],
            "at_home": home, "p_model": g["p_home_win"] if home else 1 - g["p_home_win"],
            "lock": max(g["p_home_win"], 1 - g["p_home_win"]) >= LOCK,
            "p": b["p"] if b else None, "odds": b["odds"] if b else None,
            "spread": g.get("spread"), "pred_margin": g.get("pred_margin")})
    for r in top.itertuples():
        out["props"].append({"name": r.name, "team": r.team, "side": r.side, "line": r.line,
                             "odds": float(r.odds), "proj": round(float(r.proj), 1), "p": round(float(r.p_side), 4)})
    bt = os.path.join(HERE, "data", "props_backtest.json")
    if os.path.exists(bt):
        with open(bt) as f:
            out["props_backtest"] = json.load(f)
    with open(os.path.join(HERE, "data", "backtest.json")) as f:
        gbt = json.load(f)
    out["game_backtest"] = gbt[gbt["best"]]["accuracy"]
    out["locks_backtest"] = gbt[gbt["best"]].get("locks")
    with open(os.path.join(HERE, "data", "picks.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)

    if args.budget:
        allocate(bets, args.budget)
        print(f"\nHOW TO SPLIT ${args.budget:,.2f}  (quarter Kelly)")
        for bt in sorted(bets, key=lambda b: -b["stake"]):
            to_win = bt["stake"] * american_to_payout(bt["odds"])
            line = f"  ${bt['stake']:>7.2f}  {bt['label']:<36} ({int(bt['odds']):+d})"
            print(line + (f"  wins ${to_win:.2f}" if bt["stake"] > 0 else "  no edge, skip"))
        used = sum(b["stake"] for b in bets)
        ev = sum(b["ev"] for b in bets)
        print(f"  Total bet ${used:.2f}, keep ${args.budget - used:.2f} unbet. "
              f"Expected profit ${ev:+.2f} if the model's edges are real.")
    print()


if __name__ == "__main__":
    main()
