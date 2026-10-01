
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MARKETS
from features import make_current_features
from matcher import match_player
from model import evaluate_line, quality_score

API_TO_MARKET = {v["api_key"]: k for k, v in MARKETS.items()}

def _team_abbr_from_name(team_name, team_name_map):
    return team_name_map.get(str(team_name), str(team_name))

def scan_board(
    prop_board, game_context, player_index, stats, injuries,
    training_frames, bundles, team_name_map,
    season, week, thresholds, schedule_lookup=None, weather_by_event=None, n_sims=20000
):
    rows = []
    if prop_board.empty:
        return pd.DataFrame()

    ctx_by_event = {}
    if game_context is not None and not game_context.empty:
        ctx_by_event = game_context.set_index("event_id").to_dict("index")

    for _, prop in prop_board.iterrows():
        market_name = API_TO_MARKET.get(prop["market_key"])
        if market_name not in bundles:
            continue

        event_id = prop["event_id"]
        g = ctx_by_event.get(event_id, {})
        home = _team_abbr_from_name(prop["home_team"], team_name_map)
        away = _team_abbr_from_name(prop["away_team"], team_name_map)

        # Player team can be either side. Try both for robust name matching.
        player_row, match_score = match_player(prop["player"], player_index)
        if player_row is None:
            continue
        team = str(player_row["team"])
        if team not in {home, away}:
            # Keep only plausible event/team joins.
            continue
        opponent = away if team == home else home
        is_home = 1.0 if team == home else 0.0

        home_spread = g.get("home_spread", np.nan)
        team_spread = home_spread if is_home else (-home_spread if np.isfinite(home_spread) else np.nan)
        total = g.get("game_total", np.nan)

        sched = (schedule_lookup or {}).get((home, away), {})
        roof = str(sched.get("roof", ""))
        roof_closed = 1.0 if roof.lower() in {"dome", "closed"} else 0.0
        rest = sched.get("home_rest" if is_home else "away_rest", 7.0)

        wx = (weather_by_event or {}).get(event_id, {})
        live_temp = wx.get("temperature_f", sched.get("temp", np.nan))
        live_wind = wx.get("wind_mph", sched.get("wind", np.nan))

        live_context = {
            "is_home": is_home,
            "team_rest": rest,
            "spread_line": team_spread if np.isfinite(team_spread) else 0.0,
            "total_line": total,
            "roof_closed": roof_closed,
            "temp": live_temp,
            "wind": live_wind,
        }

        frame = training_frames[market_name]
        bundle = bundles[market_name]
        try:
            current, injury = make_current_features(
                stats, frame, injuries, market_name, player_row["player_id"],
                opponent, team, season, week, live_context
            )
        except Exception:
            continue

        for side in ("Over", "Under"):
            price = prop["best_over_price"] if side == "Over" else prop["best_under_price"]
            if not np.isfinite(price):
                continue
            res = evaluate_line(bundle, current, float(prop["line"]), side, float(price), n_sims=n_sims)
            market_p_side = float(prop["consensus_p_over"]) if side == "Over" else 1.0 - float(prop["consensus_p_over"])
            score = quality_score(res, bundle, market_p_side)

            uncertain_injury = str(injury.get("report_status", "")).lower() in {"questionable", "doubtful", "out"}
            qualifies = (
                res["probability_edge"] >= thresholds["min_edge"] and
                res["ev_per_dollar"] >= thresholds["min_ev"] and
                res["projection_gap"] >= thresholds["min_gap_mae"] * max(bundle["mae"], 0.25) and
                not uncertain_injury
            )

            rows.append({
                "score": round(score, 1),
                "signal": "QUALIFIES" if qualifies else "PASS",
                "player": player_row["player_display_name"],
                "team": team,
                "opponent": opponent,
                "market": market_name,
                "side": side,
                "line": float(prop["line"]),
                "best_price": int(price),
                "projection": round(res["projection"], 2),
                "model_probability": round(res["side_probability"], 4),
                "market_novig_probability": round(market_p_side, 4),
                "probability_edge": round(res["probability_edge"], 4),
                "model_vs_market": round(res["side_probability"] - market_p_side, 4),
                "ev_per_dollar": round(res["ev_per_dollar"], 4),
                "books": int(prop["books"]),
                "injury_status": injury.get("report_status", ""),
                "practice_status": injury.get("practice_status", ""),
                "injury": injury.get("report_primary_injury", ""),
                "match_score": round(match_score, 1),
                "holdout_mae": round(bundle["mae"], 2),
                "weather": wx.get("weather_note", ""),
                "temperature_f": wx.get("temperature_f", np.nan),
                "wind_mph": wx.get("wind_mph", np.nan),
                "precip_in": wx.get("precip_in", np.nan),
                "event_id": event_id,
                "commence_time": prop["commence_time"],
            })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["signal", "score", "probability_edge", "ev_per_dollar"],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)
