
from __future__ import annotations

import numpy as np
import pandas as pd
import nflreadpy as nfl

NUMERIC_PLAYER_COLS = [
    "passing_yards", "passing_tds", "attempts", "completions",
    "rushing_yards", "carries", "rushing_tds",
    "receiving_yards", "receptions", "targets", "receiving_tds",
    "passing_air_yards", "receiving_air_yards",
]

def _to_pandas(obj):
    return obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj)

def load_nfl_data(seasons: list[int]):
    stats = _to_pandas(nfl.load_player_stats(seasons, summary_level="week"))
    schedules = _to_pandas(nfl.load_schedules(seasons))
    try:
        injuries = _to_pandas(nfl.load_injuries(seasons))
    except Exception:
        injuries = pd.DataFrame()

    stats = standardize_stats(stats)
    schedules = standardize_schedules(schedules)
    injuries = standardize_injuries(injuries)
    return stats, schedules, injuries

def standardize_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).eq("REG")].copy()

    for col in ["season", "week"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in NUMERIC_PLAYER_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # nflverse has used both "team" and "recent_team" in different outputs.
    if "team" not in df.columns and "recent_team" in df.columns:
        df["team"] = df["recent_team"]
    if "player_display_name" not in df.columns:
        for candidate in ("player_name", "display_name"):
            if candidate in df.columns:
                df["player_display_name"] = df[candidate]
                break

    required = ["player_id", "player_display_name", "position_group", "team",
                "opponent_team", "season", "week"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Player stats are missing expected columns: {missing}")

    # Team-level opportunity shares.
    team_week = df.groupby(["season", "week", "team"], as_index=False).agg(
        team_targets=("targets", "sum"),
        team_carries=("carries", "sum"),
    )
    df = df.merge(team_week, on=["season", "week", "team"], how="left")
    df["target_share"] = np.where(df["team_targets"] > 0, df["targets"] / df["team_targets"], 0.0)
    df["carry_share"] = np.where(df["team_carries"] > 0, df["carries"] / df["team_carries"], 0.0)

    return df.sort_values(["season", "week", "player_id"]).reset_index(drop=True)

def standardize_schedules(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "game_type" in df.columns:
        df = df[df["game_type"].astype(str).eq("REG")].copy()
    for c in ["season", "week", "spread_line", "total_line", "away_rest", "home_rest", "temp", "wind"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def standardize_injuries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).eq("REG")].copy()
    elif "game_type" in df.columns:
        df = df[df["game_type"].astype(str).eq("REG")].copy()
    for c in ["season", "week"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "date_modified" in df.columns:
        df["date_modified"] = pd.to_datetime(df["date_modified"], errors="coerce", utc=True)
    return df

def injury_status_code(value) -> float:
    s = str(value or "").strip().lower()
    if not s or s in {"nan", "none"}:
        return 0.0
    if "out" in s:
        return 3.0
    if "doubt" in s:
        return 2.5
    if "question" in s:
        return 1.5
    if "probab" in s:
        return 0.5
    return 0.5

def practice_status_code(value) -> float:
    s = str(value or "").strip().lower()
    if not s or s in {"nan", "none"}:
        return 0.0
    if "did not" in s or "dnp" in s:
        return 3.0
    if "limited" in s:
        return 1.5
    if "full" in s:
        return 0.25
    return 0.5

def latest_injury_rows(injuries: pd.DataFrame) -> pd.DataFrame:
    if injuries.empty:
        return pd.DataFrame(columns=[
            "season", "week", "gsis_id", "report_status", "practice_status",
            "report_primary_injury", "injury_status_code", "practice_status_code"
        ])
    x = injuries.copy()
    sort_cols = [c for c in ["season", "week", "gsis_id", "date_modified"] if c in x.columns]
    if sort_cols:
        x = x.sort_values(sort_cols)
    keys = [c for c in ["season", "week", "gsis_id"] if c in x.columns]
    if len(keys) == 3:
        x = x.drop_duplicates(keys, keep="last")
    if "report_status" not in x.columns:
        x["report_status"] = ""
    if "practice_status" not in x.columns:
        x["practice_status"] = ""
    x["injury_status_code"] = x["report_status"].map(injury_status_code)
    x["practice_status_code"] = x["practice_status"].map(practice_status_code)
    return x

def attach_historical_context(stats: pd.DataFrame, schedules: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    x = stats.copy()

    # Injury context available before the game.
    inj = latest_injury_rows(injuries)
    if not inj.empty and "gsis_id" in inj.columns:
        keep = ["season", "week", "gsis_id", "injury_status_code", "practice_status_code"]
        for c in ("report_status", "practice_status", "report_primary_injury"):
            if c in inj.columns:
                keep.append(c)
        x = x.merge(
            inj[keep],
            left_on=["season", "week", "player_id"],
            right_on=["season", "week", "gsis_id"],
            how="left",
        )
    for c in ["injury_status_code", "practice_status_code"]:
        if c not in x.columns:
            x[c] = 0.0
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)

    # Team-specific schedule rows.
    if schedules is not None and not schedules.empty:
        common = [c for c in [
            "season", "week", "game_id", "home_team", "away_team",
            "spread_line", "total_line", "home_rest", "away_rest",
            "roof", "surface", "temp", "wind", "location", "gameday", "gametime"
        ] if c in schedules.columns]
        s = schedules[common].copy()

        home = s.copy()
        home["team"] = home.get("home_team")
        home["schedule_opponent"] = home.get("away_team")
        home["is_home"] = 1.0
        home["team_rest"] = home.get("home_rest", np.nan)

        away = s.copy()
        away["team"] = away.get("away_team")
        away["schedule_opponent"] = away.get("home_team")
        away["is_home"] = 0.0
        away["team_rest"] = away.get("away_rest", np.nan)

        team_sched = pd.concat([home, away], ignore_index=True)
        keep2 = [c for c in [
            "season", "week", "team", "is_home", "team_rest",
            "spread_line", "total_line", "roof", "surface", "temp", "wind"
        ] if c in team_sched.columns]
        x = x.merge(team_sched[keep2], on=["season", "week", "team"], how="left")

    if "is_home" not in x.columns:
        x["is_home"] = 0.5
    if "team_rest" not in x.columns:
        x["team_rest"] = 7.0
    if "spread_line" not in x.columns:
        x["spread_line"] = 0.0
    if "total_line" not in x.columns:
        x["total_line"] = np.nan
    if "temp" not in x.columns:
        x["temp"] = np.nan
    if "wind" not in x.columns:
        x["wind"] = np.nan
    if "roof" not in x.columns:
        x["roof"] = ""

    x["roof_closed"] = x["roof"].astype(str).str.lower().isin(["dome", "closed"]).astype(float)
    return x
