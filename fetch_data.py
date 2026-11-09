"""Download NBA game results + closing betting lines from ESPN's public API.

Writes data/games.csv with one row per finished game:
date, season, home, away, home_pts, away_pts, spread (home line), total, home_ml, away_ml
Raw JSON is cached in data/cache so reruns are fast.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from urllib.request import Request, urlopen

import ssl

import certifi
import pandas as pd

SSL_CTX = ssl.create_default_context(cafile=certifi.where())

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "cache")
os.makedirs(CACHE, exist_ok=True)

OFFLINE = os.environ.get("OFFLINE") == "1"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}"
ODDS = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{i}/competitions/{i}/odds"

# Bookmakers in order of preference for the "closing line" we test against.
PREFERRED = ["consensus", "ESPN BET", "DraftKings", "Caesars Sportsbook", "Caesars", "Westgate"]


def get_json(url, cache_name, use_cache=True):
    path = os.path.join(CACHE, cache_name)
    if (use_cache or OFFLINE) and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    if OFFLINE:
        return None
    for attempt in range(4):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=10, context=SSL_CTX) as r:
                data = json.load(r)
            with open(path, "w") as f:
                json.dump(data, f)
            return data
        except Exception:
            time.sleep(1 + attempt * 2)
    return None


def season_dates(start_year):
    """Oct 1 .. Jun 30 of the season starting in start_year (bubble season ran to Oct 2020)."""
    d = date(start_year, 10, 1)
    end = date(2020, 10, 15) if start_year == 2019 else date(start_year + 1, 6, 30)
    if start_year == 2020:
        d, end = date(2020, 12, 15), date(2021, 7, 25)
    today = date.today()
    while d <= min(end, today):
        yield d
        d += timedelta(days=1)


def scoreboard_games(d):
    ds = d.strftime("%Y%m%d")
    # Don't cache days that aren't finished yet.
    data = get_json(SCOREBOARD.format(d=ds), f"sb_{ds}.json", use_cache=d < date.today() - timedelta(days=1))
    out = []
    if not data:
        return out
    for e in data.get("events", []):
        st = e["status"]["type"]
        stype = e.get("season", {}).get("type")
        if stype not in (2, 3, 5):  # regular season, playoffs, play-in
            continue
        c = e["competitions"][0]
        teams = {t["homeAway"]: t for t in c["competitors"]}
        if "home" not in teams:
            continue
        out.append({
            "id": e["id"],
            "date": d.isoformat(),
            "season": e["season"]["year"],
            "season_type": stype,
            "completed": bool(st.get("completed")),
            "home": teams["home"]["team"]["abbreviation"],
            "away": teams["away"]["team"]["abbreviation"],
            "home_pts": int(teams["home"].get("score") or 0),
            "away_pts": int(teams["away"].get("score") or 0),
            "neutral": bool(c.get("neutralSite")),
        })
    return out


def parse_odds(data):
    if not data or not data.get("items"):
        return {}
    items = {i.get("provider", {}).get("name"): i for i in data["items"]}
    ordered = [items[p] for p in PREFERRED if p in items] + list(data["items"])
    res = {}
    for it in ordered:
        h, a = it.get("homeTeamOdds", {}), it.get("awayTeamOdds", {})
        if "spread" not in res and it.get("spread") is not None:
            res["spread"] = float(it["spread"])
            res["book"] = it.get("provider", {}).get("name")
        if "total" not in res and it.get("overUnder") is not None:
            res["total"] = float(it["overUnder"])
        if "home_ml" not in res and h.get("moneyLine") is not None and a.get("moneyLine") is not None:
            res["home_ml"] = float(h["moneyLine"])
            res["away_ml"] = float(a["moneyLine"])
    return res


def odds_for(game):
    data = get_json(ODDS.format(i=game["id"]), f"odds_{game['id']}.json", use_cache=game["completed"])
    return {**game, **parse_odds(data)}


def main():
    seasons = [int(s) for s in sys.argv[1:]] or list(range(2019, 2027))
    dates = [d for s in seasons for d in season_dates(s)]
    print(f"Fetching {len(dates)} scoreboards...", flush=True)
    with ThreadPoolExecutor(6) as ex:
        games = [g for day in ex.map(scoreboard_games, dates) for g in day]
    print(f"{len(games)} games, fetching odds...", flush=True)
    rows = []
    with ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(odds_for, games[::-1])):
            rows.append(r)
            if i % 1000 == 0:
                print(f"  odds {i}/{len(games)}", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("id").sort_values(["date", "id"])
    df.to_csv(os.path.join(HERE, "data", "games.csv"), index=False)
    fin = df[df.completed]
    print(f"Saved {len(df)} games ({len(fin)} finished). With spread: {fin.spread.notna().sum()}, "
          f"total: {fin.total.notna().sum()}, moneyline: {fin.home_ml.notna().sum()}")
    print(fin.groupby("season")[["spread", "home_ml"]].count())


if __name__ == "__main__":
    main()
