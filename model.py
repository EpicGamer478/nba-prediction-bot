"""Features + models for NBA moneyline / spread / total predictions.

Every feature for a game is computed only from games that finished before it,
so the backtest never sees the future.
"""
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

ELO_K = 20
ELO_HFA = 65
EWMA_ALPHA = 0.12  # weight on the newest game in rolling stats

MARGIN_FEATS = ["elo_diff", "form_margin_diff", "off_diff", "def_diff", "rest_diff",
                "home_b2b", "away_b2b", "neutral", "mkt_spread"]
TOTAL_FEATS = ["exp_pts_home", "exp_pts_away", "league_ppg", "home_b2b", "away_b2b", "mkt_total"]
# "Pure" variant: stats only, never looks at the betting line.
PURE_MARGIN = [f for f in MARGIN_FEATS if f != "mkt_spread"]
PURE_TOTAL = [f for f in TOTAL_FEATS if f != "mkt_total"]


# ---------- odds helpers ----------
def american_to_prob(ml):
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def american_to_payout(ml):
    """Profit on a 1-unit stake if the bet wins."""
    return ml / 100 if ml > 0 else 100 / -ml


def prob_to_american(p):
    p = min(max(p, 0.01), 0.99)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


# ---------- features ----------
def build_features(games: pd.DataFrame) -> pd.DataFrame:
    g = games.sort_values(["date", "id"]).reset_index(drop=True).copy()
    g["date"] = pd.to_datetime(g["date"])
    elo = defaultdict(lambda: 1500.0)
    form = defaultdict(lambda: {"margin": 0.0, "pf": 112.0, "pa": 112.0, "n": 0})
    last_played = {}
    team_season = {}
    league_ppg = 112.0
    rows = []

    for r in g.itertuples(index=False):
        h, a = r.home, r.away
        # Pull ratings 25% back to the mean at the start of each team's new season.
        for t in (h, a):
            if team_season.get(t) != r.season:
                if t in team_season:
                    elo[t] = 0.75 * elo[t] + 0.25 * 1505
                    f = form[t]
                    f["margin"] *= 0.5
                    f["n"] = min(f["n"], 5)
                team_season[t] = r.season

        hfa = 0 if r.neutral else ELO_HFA
        rest_h = min((r.date - last_played[h]).days, 4) if h in last_played else 3
        rest_a = min((r.date - last_played[a]).days, 4) if a in last_played else 3
        fh, fa = form[h], form[a]
        # Expected points: own offense vs opponent defense, relative to league average.
        exp_h = fh["pf"] + fa["pa"] - league_ppg
        exp_a = fa["pf"] + fh["pa"] - league_ppg

        rows.append({
            "elo_home": elo[h], "elo_away": elo[a],
            "elo_diff": elo[h] - elo[a] + hfa,
            "elo_prob": 1 / (1 + 10 ** (-(elo[h] - elo[a] + hfa) / 400)),
            "form_margin_diff": fh["margin"] - fa["margin"],
            "off_diff": fh["pf"] - fa["pf"], "def_diff": fa["pa"] - fh["pa"],
            "rest_diff": rest_h - rest_a,
            "home_b2b": int(rest_h == 1), "away_b2b": int(rest_a == 1),
            "exp_pts_home": exp_h, "exp_pts_away": exp_a, "league_ppg": league_ppg,
            "min_games": min(fh["n"], fa["n"]),
        })

        if not r.completed:
            continue
        # ---- update state with the result ----
        margin = r.home_pts - r.away_pts
        dr = elo[h] - elo[a] + hfa
        exp_win = 1 / (1 + 10 ** (-dr / 400))
        mov_mult = ((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * (dr if margin > 0 else -dr))
        delta = ELO_K * mov_mult * ((margin > 0) - exp_win)
        elo[h] += delta
        elo[a] -= delta
        for t, pf, pa in ((h, r.home_pts, r.away_pts), (a, r.away_pts, r.home_pts)):
            f = form[t]
            al = max(EWMA_ALPHA, 1 / (f["n"] + 1)) if f["n"] < 8 else EWMA_ALPHA
            f["margin"] += al * ((pf - pa) - f["margin"])
            f["pf"] += al * (pf - f["pf"])
            f["pa"] += al * (pa - f["pa"])
            f["n"] += 1
        league_ppg += 0.004 * ((r.home_pts + r.away_pts) / 2 - league_ppg)
        last_played[h] = last_played[a] = r.date

    feats = pd.DataFrame(rows)
    out = pd.concat([g, feats], axis=1)
    out["margin"] = out.home_pts - out.away_pts
    out["total_pts"] = out.home_pts + out.away_pts
    # Market spread is quoted as the home team's line (home -5 => +5 expected margin).
    out["mkt_spread"] = -out["spread"]
    out["mkt_total"] = out["total"]
    out["neutral"] = out["neutral"].astype(int)
    return out


# ---------- models ----------
class Models:
    """Margin model (spread + moneyline) and total-points model."""

    def __init__(self, use_market=True, kind="ridge"):
        self.kind = kind
        self.use_market = use_market
        self.mf = MARGIN_FEATS if use_market else PURE_MARGIN
        self.tf = TOTAL_FEATS if use_market else PURE_TOTAL

    def fit(self, df):
        d = df[df.completed & (df.min_games >= 5)].dropna(subset=["mkt_spread", "mkt_total"])
        # With the market on, learn only an adjustment to the book's number (residual model).
        self.base_m = "mkt_spread" if self.use_market else None
        self.base_t = "mkt_total" if self.use_market else None
        # Keep the line as a feature too, so the model can learn "stats vs. line" gaps.
        mf, tf = list(self.mf), list(self.tf)
        self.mf_x, self.tf_x = mf, tf
        ym = d["margin"] - (d[self.base_m] if self.base_m else 0)
        yt = d["total_pts"] - (d[self.base_t] if self.base_t else 0)
        self.margin = self._reg().fit(d[mf], ym)
        self.total = self._reg().fit(d[tf], yt)
        pm = self._pm(d)
        self.margin_sd = float((d["margin"] - pm).std())
        self.total_sd = float((d["total_pts"] - self._pt(d)).std())
        self.win = LogisticRegression().fit(pm.values.reshape(-1, 1), (d["margin"] > 0).astype(int))
        return self

    def _pm(self, df):
        return pd.Series(self.margin.predict(df[self.mf_x]), index=df.index) + (df[self.base_m] if self.base_m else 0)

    def _pt(self, df):
        return pd.Series(self.total.predict(df[self.tf_x]), index=df.index) + (df[self.base_t] if self.base_t else 0)

    def _reg(self):
        if self.kind == "gbm":
            # Shallow, heavily regularised boosting: NBA outcomes are noisy.
            return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.04, max_iter=250,
                                                 min_samples_leaf=80, l2_regularization=1.0)
        return Ridge(alpha=50.0)

    def predict(self, df):
        out = pd.DataFrame(index=df.index)
        out["pred_margin"] = self._pm(df)
        out["pred_total"] = self._pt(df)
        out["p_home_win"] = self.win.predict_proba(out[["pred_margin"]].values)[:, 1]
        # Probability the home side covers its spread, and the game goes over.
        out["p_home_cover"] = 1 - norm.cdf(-df["spread"], loc=out["pred_margin"], scale=self.margin_sd)
        out["p_over"] = 1 - norm.cdf(df["total"], loc=out["pred_total"], scale=self.total_sd)
        return out
