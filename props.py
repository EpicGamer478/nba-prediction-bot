"""Player prop projections, backtest and daily top picks.

Data: ESPN box scores (player game logs) + DraftKings prop lines via ESPN's odds API.
Usage:
  python3 props.py fetch      # download box scores + prop lines (cached)
  python3 props.py backtest   # walk-forward test on this season's past props
  python3 props.py today      # print today's top 5 props
"""
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

import fetch_data as fd
from model import american_to_payout, american_to_prob

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={i}"
PROPS = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{i}/competitions/{i}"
         "/odds/100/propBets?limit=1000&page={p}")
INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

# DraftKings market name -> which box-score stats it adds up.
MARKETS = {
    "Total Points": ["pts"], "Total Rebounds": ["reb"], "Total Assists": ["ast"],
    "Total 3-Point Field Goals": ["fg3m"], "Total Points and Rebounds": ["pts", "reb"],
    "Total Points and Assists": ["pts", "ast"], "Total Assists and Rebounds": ["ast", "reb"],
    "Total Points, Rebounds, and Assists": ["pts", "reb", "ast"],
}
SHORT = {"Total Points": "PTS", "Total Rebounds": "REB", "Total Assists": "AST",
         "Total 3-Point Field Goals": "3PM", "Total Points and Rebounds": "PTS+REB",
         "Total Points and Assists": "PTS+AST", "Total Assists and Rebounds": "REB+AST",
         "Total Points, Rebounds, and Assists": "PTS+REB+AST"}
STATS = ["pts", "reb", "ast", "fg3m"]
USE_MARKETS = ["Total Points"]  # only points props are picked
PROP_SEASON = 2027  # seasons before this one have no stored prop lines


def logit(p):
    p = np.clip(p, 0.02, 0.98)
    return np.log(p / (1 - p))


# ---------------- data ----------------
def games_table():
    g = pd.read_csv(os.path.join(DATA, "games.csv"), dtype={"id": str})
    return g


def parse_box(gid, d):
    rows = []
    if not d or "boxscore" not in d:
        return rows
    for team in d["boxscore"].get("players", []):
        abbr = team["team"]["abbreviation"]
        st = team["statistics"][0]
        keys = st["keys"]
        for a in st["athletes"]:
            if a.get("didNotPlay") or not a.get("stats"):
                continue
            s = dict(zip(keys, a["stats"]))
            try:
                mins = float(s["minutes"])
            except (ValueError, KeyError):
                continue
            if mins <= 0:
                continue
            rows.append({"game_id": gid, "team": abbr, "athlete_id": str(a["athlete"]["id"]),
                         "name": a["athlete"]["displayName"], "starter": bool(a.get("starter")),
                         "min": mins, "pts": float(s["points"]), "reb": float(s["rebounds"]),
                         "ast": float(s["assists"]),
                         "fg3m": float(s["threePointFieldGoalsMade-threePointFieldGoalsAttempted"].split("-")[0])})
    return rows


def _prop_key(it):
    try:
        return (it["type"]["name"], it["athlete"]["$ref"], float(it["current"]["target"]["value"]))
    except (KeyError, TypeError, ValueError):
        return None


def parse_props(gid, pages):
    items = [it for p in pages if p for it in p.get("items", [])]
    rows = []
    # Items come in over/under pairs for the same player and line (first = over).
    i = 0
    while i < len(items) - 1:
        a, b = items[i], items[i + 1]
        ka, kb = _prop_key(a), _prop_key(b)
        if ka and ka == kb and ka[0] in MARKETS:
            try:
                rows.append({"game_id": gid, "athlete_id": ka[1].split("/athletes/")[1].split("?")[0],
                             "market": ka[0], "line": ka[2],
                             "over_odds": float(a["odds"]["american"]["value"]),
                             "under_odds": float(b["odds"]["american"]["value"])})
            except (KeyError, ValueError, IndexError):
                pass
            i += 2
        else:
            i += 1
    return rows


def fetch_props(gid, live=False):
    tag = "live_" if live else ""
    first = fd.get_json(PROPS.format(i=gid, p=1), f"props_{tag}{gid}_1.json", use_cache=not live)
    pages = [first]
    for p in range(2, (first or {}).get("pageCount", 1) + 1):
        pages.append(fd.get_json(PROPS.format(i=gid, p=p), f"props_{tag}{gid}_{p}.json", use_cache=not live))
    return parse_props(gid, pages)


def fetch():
    g = games_table()
    done = g[g.completed & (g.season >= PROP_SEASON - 1)].iloc[::-1]  # newest first
    if os.environ.get("CACHED_ONLY_OLD") == "1":
        # Keep this season + whatever of last season is already downloaded.
        have = {f[4:-5] for f in os.listdir(fd.CACHE) if f.startswith("sum_")}
        done = done[(done.season >= PROP_SEASON) | done.id.isin(have)]
    print(f"Box scores for {len(done)} games...", flush=True)
    with ThreadPoolExecutor(6) as ex:
        boxes = list(ex.map(lambda i: parse_box(i, fd.get_json(SUMMARY.format(i=i), f"sum_{i}.json")), done.id))
    logs = pd.DataFrame([r for b in boxes for r in b]).merge(
        g[["id", "date", "season", "home", "away"]].rename(columns={"id": "game_id"}), on="game_id")
    logs["opp"] = np.where(logs.team == logs.home, logs.away, logs.home)
    logs.to_csv(os.path.join(DATA, "player_logs.csv"), index=False)
    print(f"  {len(logs)} player-games", flush=True)

    cur = done[done.season >= PROP_SEASON]
    print(f"Prop lines for {len(cur)} games this season...", flush=True)
    with ThreadPoolExecutor(4) as ex:
        props = [r for rows in ex.map(fetch_props, cur.id) for r in rows]
    pd.DataFrame(props).to_csv(os.path.join(DATA, "props_hist.csv"), index=False)
    print(f"  {len(props)} historical props", flush=True)


# ---------------- projections ----------------
def player_features(logs):
    """Pre-game projections for every player-game (only earlier games are used)."""
    logs = logs.sort_values(["athlete_id", "date", "game_id"]).copy()
    g = logs.groupby("athlete_id", group_keys=False)
    feats = pd.DataFrame(index=logs.index)
    feats["n_prev"] = g.cumcount()
    min_recent = g["min"].transform(lambda s: s.shift().ewm(span=6, min_periods=1).mean())
    min_long = g["min"].transform(lambda s: s.shift().rolling(25, min_periods=1).mean())
    feats["proj_min"] = 0.65 * min_recent + 0.35 * min_long
    for s in STATS:
        tot_r = g[s].transform(lambda x: x.shift().ewm(span=10, min_periods=1).mean())
        min_r = g["min"].transform(lambda x: x.shift().ewm(span=10, min_periods=1).mean())
        tot_l = g[s].transform(lambda x: x.shift().rolling(40, min_periods=1).sum())
        min_l = g["min"].transform(lambda x: x.shift().rolling(40, min_periods=1).sum())
        rate = 0.6 * (tot_r / min_r) + 0.4 * (tot_l / min_l)
        feats[f"proj_{s}"] = feats["proj_min"] * rate
        # Per-36 residual spread, scaled to projected minutes.
        feats[f"sd_{s}"] = g[s].transform(lambda x: x.shift().rolling(25, min_periods=3).std())
    return pd.concat([logs, feats], axis=1)


def latest_features(logs):
    """Projection for each player's *next* game, using all games so far."""
    logs = logs.copy()
    dummy = logs.sort_values("date").groupby("athlete_id").tail(1).copy()
    dummy["game_id"] = "next"
    dummy["date"] = "9999-12-31"
    for c in ["min"] + STATS:
        dummy[c] = np.nan
    f = player_features(pd.concat([logs, dummy]))
    return f[f.game_id == "next"].set_index("athlete_id")


def model_prob(row, stats):
    proj = sum(row[f"proj_{s}"] for s in stats)
    sd = math.sqrt(sum((row[f"sd_{s}"] if pd.notna(row[f"sd_{s}"]) else 0) ** 2 for s in stats))
    sd = max(sd, math.sqrt(max(proj, 0.5)) * 1.1)
    return proj, 1 - norm.cdf(row["line"], loc=proj, scale=sd)


def score_props(props, feats):
    out = []
    for r in props.itertuples(index=False):
        if r.athlete_id not in feats.index:
            continue
        f = feats.loc[r.athlete_id]
        if isinstance(f, pd.DataFrame):
            f = f.iloc[-1]
        row = {**r._asdict(), **f.to_dict()}
        proj, p = model_prob(row, MARKETS[r.market])
        po, pu = american_to_prob(r.over_odds), american_to_prob(r.under_odds)
        out.append({**r._asdict(), "name": f["name"], "team": f["team"], "proj": proj, "p_model": p,
                    "p_mkt": po / (po + pu), "n_prev": f["n_prev"], "proj_min": f["proj_min"]})
    return pd.DataFrame(out)


class Calibrator:
    """Blend model and market probabilities: logistic regression on their log-odds."""

    def fit(self, df, y):
        X = np.c_[logit(df.p_model), logit(df.p_mkt)]
        self.lr = LogisticRegression(C=0.5).fit(X, y)
        return self

    def predict(self, df):
        return self.lr.predict_proba(np.c_[logit(df.p_model), logit(df.p_mkt)])[:, 1]


def add_ev(df, p_over):
    df = df.copy()
    df["p_over"] = p_over
    ev_o = df.p_over * df.over_odds.map(american_to_payout) - (1 - df.p_over)
    ev_u = (1 - df.p_over) * df.under_odds.map(american_to_payout) - df.p_over
    df["side"] = np.where(ev_o >= ev_u, "over", "under")
    df["ev"] = np.maximum(ev_o, ev_u)
    df["p_side"] = np.where(df.side == "over", df.p_over, 1 - df.p_over)
    df["odds"] = np.where(df.side == "over", df.over_odds, df.under_odds)
    return df


def eligible(df):
    # Regular-rotation players with a real track record; skip lines more than ~10 pts from projection
    # (usually a sign of injury/role news the model doesn't know).
    return df[(df.n_prev >= 8) & (df.proj_min >= 22) & ((df.proj - df.line).abs() < 10)]


def top5(df):
    return eligible(df).sort_values("ev", ascending=False).drop_duplicates("athlete_id").head(5)


# ---------------- backtest ----------------
def backtest():
    logs = pd.read_csv(os.path.join(DATA, "player_logs.csv"), dtype={"athlete_id": str, "game_id": str})
    props = pd.read_csv(os.path.join(DATA, "props_hist.csv"), dtype={"athlete_id": str, "game_id": str})
    props = props[props.market.isin(USE_MARKETS)]
    feats = player_features(logs)
    key = feats.set_index(["game_id", "athlete_id"])
    rows = []
    for r in props.itertuples(index=False):
        if (r.game_id, r.athlete_id) not in key.index:
            continue  # player didn't play: prop voided
        f = key.loc[(r.game_id, r.athlete_id)]
        row = {**r._asdict(), **f.to_dict()}
        if pd.isna(row["proj_pts"]):
            continue
        proj, p = model_prob(row, MARKETS[r.market])
        actual = sum(row[s] for s in MARKETS[r.market])
        po, pu = american_to_prob(r.over_odds), american_to_prob(r.under_odds)
        rows.append({**r._asdict(), "date": row["date"], "name": row["name"], "proj": proj, "p_model": p,
                     "p_mkt": po / (po + pu), "actual": actual, "n_prev": row["n_prev"],
                     "proj_min": row["proj_min"]})
    bt = pd.DataFrame(rows)
    bt = bt[bt.actual != bt.line]
    bt["over_hit"] = (bt.actual > bt.line).astype(int)
    days = sorted(bt.date.unique())
    print(f"{len(bt)} props over {len(days)} days ({days[0]} to {days[-1]})")
    print(f"Avg miss: model {np.mean(abs(bt.proj - bt.actual)):.2f} vs line {np.mean(abs(bt.line - bt.actual)):.2f}")
    print(f"Brier: model {np.mean((bt.p_model - bt.over_hit) ** 2):.4f} | market {np.mean((bt.p_mkt - bt.over_hit) ** 2):.4f}")

    picks, all_cal = [], []
    for i, d in enumerate(days):
        train = bt[bt.date < d]
        test = bt[bt.date == d]
        if train.date.nunique() < 4:
            continue
        cal = Calibrator().fit(train, train.over_hit)
        t = add_ev(test, cal.predict(test))
        all_cal.append(t)
        top = top5(t)
        top = top.assign(won=np.where(top.side == "over", top.over_hit, 1 - top.over_hit))
        top["profit"] = np.where(top.won == 1, top.odds.map(american_to_payout), -1.0)
        picks.append(top)
    allc = pd.concat(all_cal)
    print(f"Brier blended (out of sample): {np.mean((allc.p_over - allc.over_hit) ** 2):.4f} "
          f"vs market {np.mean((allc.p_mkt - allc.over_hit) ** 2):.4f} on {len(allc)} props")
    p = pd.concat(picks)
    print(f"\nTop-5-per-day strategy: {len(p)} picks, {p.won.mean():.1%} hit, "
          f"{p.profit.sum():+.1f} units, ROI {p.profit.mean():+.1%}")
    print(p.groupby("market").agg(n=("won", "size"), hit=("won", "mean"), roi=("profit", "mean")).round(3))
    summary = {"props": int(len(bt)), "days": len(days), "picks": int(len(p)), "hit_rate": float(p.won.mean()),
               "roi": float(p.profit.mean()), "units": float(p.profit.sum()),
               "brier_model": float(np.mean((bt.p_model - bt.over_hit) ** 2)),
               "brier_market": float(np.mean((bt.p_mkt - bt.over_hit) ** 2))}
    with open(os.path.join(DATA, "props_backtest.json"), "w") as f:
        json.dump(summary, f, indent=1)
    p.to_csv(os.path.join(DATA, "props_backtest_picks.csv"), index=False)
    return bt


# ---------------- today ----------------
def injured_out():
    d = fd.get_json(INJURIES, "injuries_live.json", use_cache=False) or {}
    out = set()
    for team in d.get("injuries", []):
        for inj in team.get("injuries", []):
            if inj.get("status", "").lower() in ("out", "doubtful"):
                out.add(str(inj.get("athlete", {}).get("id", "")))
    return out


def today(day=None):
    day = day or date.today()
    logs = pd.read_csv(os.path.join(DATA, "player_logs.csv"), dtype={"athlete_id": str, "game_id": str})
    logs = logs[pd.to_datetime(logs.date).dt.date < day]
    games = [g for g in fd.scoreboard_games(day) if not g["completed"]]
    props = pd.DataFrame([r for g in games for r in fetch_props(g["id"], live=True)])
    if props.empty:
        return pd.DataFrame()
    props = props[props.market.isin(USE_MARKETS)]
    feats = latest_features(logs)
    scored = score_props(props, feats)
    out = injured_out()
    scored = scored[~scored.athlete_id.isin(out)]

    hist = backtest_frame()
    cal = Calibrator().fit(hist, hist.over_hit)
    return top5(add_ev(scored, cal.predict(scored)))


_BT = None


def backtest_frame():
    global _BT
    if _BT is None:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            _BT = backtest()
    return _BT


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "today"
    if cmd == "fetch":
        fetch()
    elif cmd == "backtest":
        backtest()
    else:
        print(today()[["name", "team", "market", "side", "line", "odds", "proj", "p_side", "ev"]])
