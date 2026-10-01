
from __future__ import annotations

import requests
import pandas as pd
import numpy as np
from config import ODDS_API_BASE, SPORT_KEY, MARKETS

def american_implied_probability(odds: float) -> float:
    odds = float(odds)
    return abs(odds) / (abs(odds) + 100.0) if odds < 0 else 100.0 / (odds + 100.0)

def profit_per_dollar(odds: float) -> float:
    odds = float(odds)
    return 100.0 / abs(odds) if odds < 0 else odds / 100.0

def _get(path, api_key, params=None, timeout=30):
    params = dict(params or {})
    params["apiKey"] = api_key
    url = f"{ODDS_API_BASE}{path}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json(), dict(r.headers)

def get_events(api_key: str):
    return _get(f"/sports/{SPORT_KEY}/events", api_key, {"dateFormat": "iso"})

def get_featured_odds(api_key: str, regions="us", bookmakers=None):
    params = {
        "regions": regions,
        "markets": "spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    return _get(f"/sports/{SPORT_KEY}/odds", api_key, params)

def get_event_props(api_key: str, event_id: str, market_names: list[str], regions="us", bookmakers=None):
    market_keys = [MARKETS[m]["api_key"] for m in market_names]
    params = {
        "regions": regions,
        "markets": ",".join(market_keys),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    return _get(f"/sports/{SPORT_KEY}/events/{event_id}/odds", api_key, params)

def flatten_props(event_json: dict) -> pd.DataFrame:
    rows = []
    event_id = event_json.get("id")
    for book in event_json.get("bookmakers", []):
        bkey = book.get("key")
        btitle = book.get("title", bkey)
        for market in book.get("markets", []):
            mkey = market.get("key")
            for out in market.get("outcomes", []):
                side = str(out.get("name", "")).title()
                if side not in {"Over", "Under"}:
                    continue
                rows.append({
                    "event_id": event_id,
                    "commence_time": event_json.get("commence_time"),
                    "home_team": event_json.get("home_team"),
                    "away_team": event_json.get("away_team"),
                    "bookmaker": btitle,
                    "bookmaker_key": bkey,
                    "market_key": mkey,
                    "player": out.get("description"),
                    "side": side,
                    "line": pd.to_numeric(out.get("point"), errors="coerce"),
                    "price": pd.to_numeric(out.get("price"), errors="coerce"),
                    "last_update": market.get("last_update") or book.get("last_update"),
                })
    return pd.DataFrame(rows)

def consensus_prop_board(flat: pd.DataFrame) -> pd.DataFrame:
    if flat.empty:
        return flat

    x = flat.dropna(subset=["player", "line", "price"]).copy()
    # Pair over/under at each bookmaker/line to remove vig.
    idx = ["event_id", "market_key", "bookmaker", "bookmaker_key", "player", "line",
           "home_team", "away_team", "commence_time"]
    piv = x.pivot_table(index=idx, columns="side", values="price", aggfunc="last").reset_index()
    if "Over" not in piv.columns or "Under" not in piv.columns:
        return pd.DataFrame()

    piv = piv.dropna(subset=["Over", "Under"]).copy()
    piv["p_over_raw"] = piv["Over"].map(american_implied_probability)
    piv["p_under_raw"] = piv["Under"].map(american_implied_probability)
    denom = piv["p_over_raw"] + piv["p_under_raw"]
    piv["p_over_novig"] = piv["p_over_raw"] / denom

    # Prefer the line posted by the most books for each player/market/event.
    line_pop = (
        piv.groupby(["event_id", "market_key", "player", "line"], as_index=False)
        .agg(book_count=("bookmaker", "nunique"))
    )
    line_pop["rank"] = line_pop.groupby(["event_id", "market_key", "player"])["book_count"] \
        .rank(method="first", ascending=False)
    modal = line_pop[line_pop["rank"] == 1][["event_id", "market_key", "player", "line"]]
    p = piv.merge(modal, on=["event_id", "market_key", "player", "line"], how="inner")

    def best_price(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return float(s.max()) if len(s) else np.nan

    board = (
        p.groupby(
            ["event_id", "market_key", "player", "line", "home_team", "away_team", "commence_time"],
            as_index=False
        )
        .agg(
            consensus_p_over=("p_over_novig", "mean"),
            books=("bookmaker", "nunique"),
            best_over_price=("Over", best_price),
            best_under_price=("Under", best_price),
        )
    )
    return board

def flatten_game_context(events_json: list[dict]) -> pd.DataFrame:
    rows = []
    for event in events_json:
        event_id = event.get("id")
        home = event.get("home_team")
        away = event.get("away_team")
        spreads, totals = [], []
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") == "totals":
                    pts = [o.get("point") for o in market.get("outcomes", []) if o.get("point") is not None]
                    if pts:
                        totals.append(float(pts[0]))
                elif market.get("key") == "spreads":
                    for o in market.get("outcomes", []):
                        if o.get("name") == home and o.get("point") is not None:
                            spreads.append(float(o["point"]))
                            break
        home_spread = float(np.median(spreads)) if spreads else np.nan
        total_line = float(np.median(totals)) if totals else np.nan
        rows.append({
            "event_id": event_id,
            "home_team": home,
            "away_team": away,
            "commence_time": event.get("commence_time"),
            "home_spread": home_spread,
            "game_total": total_line,
            "home_implied_total": (total_line / 2 - home_spread / 2)
                if np.isfinite(total_line) and np.isfinite(home_spread) else np.nan,
            "away_implied_total": (total_line / 2 + home_spread / 2)
                if np.isfinite(total_line) and np.isfinite(home_spread) else np.nan,
        })
    return pd.DataFrame(rows)
