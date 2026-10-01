
from __future__ import annotations

import numpy as np
import pandas as pd
from config import MARKETS
from data import attach_historical_context, latest_injury_rows

BASE_CONTEXT = [
    "is_home", "team_rest", "spread_line", "total_line",
    "roof_closed", "temp", "wind",
    "injury_status_code", "practice_status_code",
]

def _shifted_roll(x, group_key, value_col, window, fn="mean"):
    g = x.groupby(group_key)[value_col]
    r = g.rolling(window, min_periods=1)
    if fn == "mean":
        out = r.mean()
    elif fn == "std":
        out = r.std()
    else:
        raise ValueError(fn)
    return out.reset_index(level=0, drop=True).groupby(x[group_key]).shift(1)

def build_training_frame(stats, schedules, injuries, market_name):
    cfg = MARKETS[market_name]
    target = cfg["target"]
    x = attach_historical_context(stats, schedules, injuries)
    x = x[x["position_group"].isin(cfg["positions"])].copy()
    x = x.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    for w in (3, 5, 10):
        x[f"target_l{w}"] = _shifted_roll(x, "player_id", target, w, "mean")
    x["target_sd5"] = _shifted_roll(x, "player_id", target, 5, "std")

    for col in cfg["opportunity"] + cfg.get("usage", []):
        x[f"{col}_l5"] = _shifted_roll(x, "player_id", col, 5, "mean")

    x["career_games_prior"] = x.groupby("player_id").cumcount().astype(float)
    x["season_week"] = x["week"].astype(float)

    # Opponent production allowed to the player's position group.
    allowed = (
        x.groupby(["season", "week", "opponent_team", "position_group"], as_index=False)[target]
        .sum()
        .rename(columns={target: "opp_allowed_game"})
        .sort_values(["opponent_team", "position_group", "season", "week"])
    )
    grp_keys = ["opponent_team", "position_group"]
    allowed["opp_allowed_l5"] = (
        allowed.groupby(grp_keys)["opp_allowed_game"]
        .rolling(5, min_periods=1).mean()
        .reset_index(level=[0, 1], drop=True)
        .groupby([allowed["opponent_team"], allowed["position_group"]]).shift(1)
    )
    x = x.merge(
        allowed[["season", "week", "opponent_team", "position_group", "opp_allowed_l5"]],
        on=["season", "week", "opponent_team", "position_group"],
        how="left",
    )

    features = [
        "target_l3", "target_l5", "target_l10", "target_sd5",
        "opp_allowed_l5", "career_games_prior", "season_week",
    ] + [f"{c}_l5" for c in cfg["opportunity"] + cfg.get("usage", [])]

    for c in BASE_CONTEXT:
        if c in x.columns:
            features.append(c)

    x = x.dropna(subset=["target_l3", "target_l5", "target_l10"]).copy()
    for c in features:
        x[c] = pd.to_numeric(x[c], errors="coerce")
        med = x[c].median()
        x[c] = x[c].fillna(0.0 if pd.isna(med) else med)

    x.attrs["feature_cols"] = features
    x.attrs["target_col"] = target
    x.attrs["market_name"] = market_name
    return x

def _current_injury(injuries, season, week, player_id):
    out = {
        "injury_status_code": 0.0,
        "practice_status_code": 0.0,
        "report_status": "",
        "practice_status": "",
        "report_primary_injury": "",
    }
    if injuries is None or injuries.empty:
        return out
    inj = latest_injury_rows(injuries)
    if inj.empty:
        return out
    q = inj[
        (pd.to_numeric(inj["season"], errors="coerce") == season) &
        (pd.to_numeric(inj["week"], errors="coerce") == week) &
        (inj["gsis_id"].astype(str) == str(player_id))
    ]
    if q.empty:
        return out
    r = q.iloc[-1]
    for k in out:
        if k in r.index and pd.notna(r[k]):
            out[k] = r[k]
    return out

def make_current_features(
    stats, training_frame, injuries, market_name, player_id,
    opponent, team, season, week, live_context=None
):
    cfg = MARKETS[market_name]
    target = cfg["target"]
    p = stats[stats["player_id"].astype(str) == str(player_id)].sort_values(["season", "week"]).copy()
    if p.empty:
        raise ValueError("No player history found")

    f = {}
    vals = p[target].astype(float)
    f["target_l3"] = vals.tail(3).mean()
    f["target_l5"] = vals.tail(5).mean()
    f["target_l10"] = vals.tail(10).mean()
    f["target_sd5"] = vals.tail(5).std() if len(vals.tail(5)) >= 2 else 0.0

    for col in cfg["opportunity"] + cfg.get("usage", []):
        f[f"{col}_l5"] = p[col].astype(float).tail(5).mean()

    f["career_games_prior"] = float(len(p))
    f["season_week"] = float(week)

    pos = p.iloc[-1]["position_group"]
    opp = training_frame[
        (training_frame["opponent_team"].astype(str) == str(opponent)) &
        (training_frame["position_group"].astype(str) == str(pos))
    ].sort_values(["season", "week"])
    if not opp.empty and opp["opp_allowed_l5"].notna().any():
        f["opp_allowed_l5"] = float(opp["opp_allowed_l5"].dropna().iloc[-1])
    else:
        f["opp_allowed_l5"] = float(training_frame["opp_allowed_l5"].median())

    inj = _current_injury(injuries, season, week, player_id)
    f["injury_status_code"] = float(inj["injury_status_code"])
    f["practice_status_code"] = float(inj["practice_status_code"])

    live_context = live_context or {}
    f["is_home"] = float(live_context.get("is_home", 0.5))
    f["team_rest"] = float(live_context.get("team_rest", 7.0) or 7.0)
    f["spread_line"] = float(live_context.get("spread_line", 0.0) or 0.0)
    f["total_line"] = float(live_context.get("total_line", training_frame["total_line"].median())
                            if "total_line" in training_frame.columns else 0.0)
    f["roof_closed"] = float(live_context.get("roof_closed", 0.0))
    f["temp"] = float(live_context.get("temp", np.nan))
    f["wind"] = float(live_context.get("wind", np.nan))

    current = pd.DataFrame([f])
    for c in training_frame.attrs["feature_cols"]:
        med = training_frame[c].median()
        fill = 0.0 if pd.isna(med) else float(med)
        if c not in current:
            current[c] = fill
        current[c] = pd.to_numeric(current[c], errors="coerce").fillna(fill)

    return current[training_frame.attrs["feature_cols"]], inj
