"""Tonight's double-up parlay: the 2-4 leg moneyline parlay most likely to hit
that still pays at least STAKE in profit.

Model win chances come from data/tonight.json (written by predict.py). Live prices come
from Polymarket's public API; any game Polymarket doesn't list falls back to ESPN's
sportsbook moneyline.

Usage:
  python3 parlay.py            # writes data/parlay.json and prints the parlay
  python3 parlay.py --stake 10
"""
import argparse
import itertools
import json
import os
import ssl
import urllib.request
from datetime import datetime, timezone

import certifi

from model import american_to_payout

HERE = os.path.dirname(os.path.abspath(__file__))
STAKE = 10.0
MIN_PROFIT = 10.0          # profit wanted on STAKE, so the parlay must pay at least +100
LEGS = (2, 3, 4)
POOL = 8                   # only build parlays from the 8 most confident picks
POLY_EVENTS = "https://gamma-api.polymarket.com/events?series_id=10345&active=true&closed=false&limit=100"
SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def best_combo(legs, min_dec):
    """legs: list of dicts with p (model chance) and dec (decimal payout incl. stake).
    Returns the combo with the highest model chance whose payout is at least min_dec,
    breaking ties by expected value. None if no combo pays enough."""
    pool = sorted(legs, key=lambda l: -l["p"])[:POOL]
    best = None
    for k in LEGS:
        for c in itertools.combinations(pool, k):
            dec = 1.0
            p = 1.0
            for l in c:
                dec *= l["dec"]
                p *= l["p"]
            if dec < min_dec:
                continue
            key = (round(p, 3), p * dec)
            if best is None or key > best[0]:
                best = (key, c, p, dec)
    if best is None:
        return None
    _, c, p, dec = best
    return {"legs": list(c), "p": p, "dec": dec}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "nba-prediction-bot"})
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=20) as r:
        return json.load(r)


def polymarket_prices(day):
    """{team nickname: price to buy 'win'} for games on `day` (YYYY-MM-DD)."""
    prices = {}
    try:
        events = _get(POLY_EVENTS)
    except Exception as e:
        print(f"Polymarket unavailable ({e}); using sportsbook lines.")
        return prices
    for e in events:
        if not str(e.get("slug", "")).endswith(day):
            continue
        for m in e.get("markets", []):
            if m.get("sportsMarketType") != "moneyline" or not m.get("acceptingOrders", True):
                continue
            names = json.loads(m["outcomes"])
            mid = [float(x) for x in json.loads(m["outcomePrices"])]
            ask, bid = m.get("bestAsk"), m.get("bestBid")
            # Buying outcome 0 costs the ask; buying outcome 1 costs 1 - bid.
            buy = [float(ask) if ask else mid[0], 1 - float(bid) if bid else mid[1]]
            for n, b in zip(names, buy):
                if 0.01 < b < 0.99:
                    prices[n.lower()] = b
    return prices


def match_price(full_name, prices):
    """Match ESPN's 'Washington Wizards' to Polymarket's 'Wizards'."""
    full = full_name.lower()
    hits = [n for n in prices if full.endswith(n)]
    return prices[max(hits, key=len)] if hits else None


def tonight_legs(t, prices):
    now = datetime.now(timezone.utc)
    legs = []
    for g in t["games"]:
        if g.get("final") or (g.get("tip") and datetime.fromisoformat(g["tip"].replace("Z", "+00:00")) <= now):
            continue  # already started
        home = g["p_home_win"] >= 0.5
        team = g["home_name"] if home else g["away_name"]
        ml = g.get("home_ml") if home else g.get("away_ml")
        price = match_price(team, prices)
        if price:
            dec, src = 1 / price, "Polymarket"
        elif ml is not None:
            dec, src = 1 + american_to_payout(ml), "sportsbook"
        else:
            continue
        legs.append({"team": team, "abbr": g["home"] if home else g["away"],
                     "opp": g["away_name"] if home else g["home_name"], "at_home": home,
                     "color": g.get("home_color") if home else g.get("away_color"),
                     "tip": g.get("tip"), "p": max(g["p_home_win"], 1 - g["p_home_win"]),
                     "dec": dec, "price": 1 / dec, "source": src})
    return legs


def main(stake=STAKE, min_profit=MIN_PROFIT):
    with open(os.path.join(HERE, "data", "tonight.json")) as f:
        t = json.load(f)
    prices = polymarket_prices(t["date"])
    legs = tonight_legs(t, prices)
    combo = best_combo(legs, 1 + min_profit / stake)
    out = {"date": t["date"], "generated": datetime.now().strftime("%Y-%m-%dT%H:%M"),
           "stake": stake, "min_profit": min_profit, "polymarket_games": len(prices) // 2, "parlay": None}
    bt = os.path.join(HERE, "data", "backtest.json")
    if os.path.exists(bt):
        with open(bt) as f:
            out["backtest"] = json.load(f).get("parlay")
    if combo:
        out["parlay"] = {"legs": combo["legs"], "p": round(combo["p"], 4), "dec": round(combo["dec"], 4),
                         "profit": round(stake * (combo["dec"] - 1), 2),
                         "ev": round(stake * (combo["p"] * combo["dec"] - 1), 2)}
    with open(os.path.join(HERE, "data", "parlay.json"), "w") as f:
        json.dump(out, f, indent=1)

    print(f"\nDOUBLE-UP PARLAY  (${stake:.0f} to win at least ${min_profit:.0f})")
    if not combo:
        print("  No parlay tonight: nothing left to play pays enough.")
        return out
    for l in combo["legs"]:
        print(f"  {l['team']:<24} {l['p']:>4.0%}  price {l['price']:.2f} ({l['source']})")
    pr = out["parlay"]
    print(f"  {len(combo['legs'])} legs · {pr['p']:.0%} to hit · ${stake:.0f} wins ${pr['profit']:.2f}")
    if out.get("backtest"):
        b = out["backtest"]
        print(f"  Backtest: {b['nights']} nights, hit {b['hit_rate']:.0%}, return {b['roi']:+.1%} per parlay")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stake", type=float, default=STAKE)
    ap.add_argument("--min-profit", type=float, default=MIN_PROFIT)
    a = ap.parse_args()
    main(a.stake, a.min_profit)
