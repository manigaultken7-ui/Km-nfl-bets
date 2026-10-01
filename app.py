
from __future__ import annotations

import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st

from config import MARKETS, DEFAULT_MARKETS
from data import load_nfl_data
from features import build_training_frame, make_current_features
from matcher import make_player_index, match_player
from model import fit_market_model, evaluate_line, quality_score
from odds import (
    get_featured_odds, get_events, get_event_props,
    flatten_props, consensus_prop_board, flatten_game_context
)
from scanner import scan_board
from weather import get_game_weather

st.set_page_config(page_title="NFL Prop Edge V2", page_icon="🏈", layout="wide")
st.title("🏈 NFL Prop Edge V2")
st.caption("Probability-first NFL player prop research: projection, no-vig market probability, EV, validation, and slate scanning.")

TEAM_NAME_MAP = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LA","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
    "Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS",
}

with st.sidebar:
    st.header("Model settings")
    seasons = st.multiselect("Training seasons", [2022, 2023, 2024, 2025, 2026], default=[2023, 2024, 2025, 2026])
    markets = st.multiselect("Markets to scan", list(MARKETS), default=DEFAULT_MARKETS)
    min_edge_pp = st.slider("Minimum edge vs price (pp)", 0.0, 15.0, 5.0, 0.5)
    min_ev_pct = st.slider("Minimum estimated EV", 0.0, 25.0, 4.0, 0.5)
    min_gap_mae = st.slider("Minimum projection gap (× holdout MAE)", 0.0, 1.5, 0.35, 0.05)
    n_sims = st.select_slider("Monte Carlo draws", options=[5000, 10000, 20000, 30000, 50000], value=20000)

    st.divider()
    st.header("Live odds")
    default_key = os.getenv("THE_ODDS_API_KEY", "")
    api_key = st.text_input("The Odds API key", value=default_key, type="password")
    bookmakers = st.text_input("Bookmakers (optional)", placeholder="draftkings,fanduel,betmgm,caesars")
    st.caption("Leaving bookmakers blank uses the API's regional defaults.")

@st.cache_data(ttl=3600, show_spinner=False)
def cached_nfl_data(seasons_tuple):
    return load_nfl_data(list(seasons_tuple))

@st.cache_resource(show_spinner=False)
def cached_model(stats_hash_key, market_name, _stats, _schedules, _injuries):
    frame = build_training_frame(_stats, _schedules, _injuries, market_name)
    bundle = fit_market_model(frame, market_name)
    return frame, bundle

def schedule_lookup_for_week(schedules, season, week):
    out = {}
    if schedules is None or schedules.empty:
        return out
    q = schedules[(schedules["season"] == season) & (schedules["week"] == week)]
    for _, r in q.iterrows():
        key = (str(r.get("home_team")), str(r.get("away_team")))
        out[key] = r.to_dict()
    return out

if not seasons:
    st.warning("Select at least one training season.")
    st.stop()

with st.spinner("Loading nflverse player, schedule, and injury data..."):
    stats, schedules, injuries = cached_nfl_data(tuple(seasons))

current_season = int(stats["season"].max())
played_weeks = stats.loc[stats["season"] == current_season, "week"]
current_week = int(played_weeks.max()) + 1 if len(played_weeks) else 1
player_index = make_player_index(stats)

tab1, tab2, tab3 = st.tabs(["🔥 Slate Scanner", "🎯 Single Prop", "📊 Model Diagnostics"])

with tab1:
    st.subheader("Live positive-EV slate scanner")
    st.write(
        "Pulls current player props, removes bookmaker vig from paired Over/Under prices, "
        "projects each player, and ranks only edges that pass your filters."
    )

    if st.button("Scan live NFL props", type="primary", use_container_width=True):
        if not api_key:
            st.error("Enter a The Odds API key in the sidebar.")
        elif not markets:
            st.error("Select at least one market.")
        else:
            book_param = bookmakers.strip() or None
            with st.spinner("Fetching current NFL events and game markets..."):
                events, event_headers = get_events(api_key)
                featured, featured_headers = get_featured_odds(api_key, bookmakers=book_param)
                game_ctx = flatten_game_context(featured)

            frames, bundles = {}, {}
            stats_key = f"{min(seasons)}-{max(seasons)}-{len(stats)}"
            progress = st.progress(0, text="Training market models...")
            for i, m in enumerate(markets, 1):
                frame, bundle = cached_model(stats_key, m, _stats=stats, _schedules=schedules, _injuries=injuries)
                frames[m], bundles[m] = frame, bundle
                progress.progress(i / len(markets), text=f"Model ready: {m}")
            progress.empty()

            all_props = []
            with st.spinner("Fetching player props event by event..."):
                for event in events:
                    try:
                        payload, _ = get_event_props(
                            api_key, event["id"], markets,
                            bookmakers=book_param
                        )
                        flat = flatten_props(payload)
                        if not flat.empty:
                            all_props.append(flat)
                    except Exception as e:
                        st.caption(f"Skipped one event prop feed: {e}")

            if not all_props:
                st.warning("No paired Over/Under props were returned for the selected markets.")
            else:
                flat = pd.concat(all_props, ignore_index=True)
                board = consensus_prop_board(flat)
                schedule_lookup = schedule_lookup_for_week(schedules, current_season, current_week)

                # Pull one forecast per game and reuse it for every prop in that event.
                weather_by_event = {}
                with st.spinner("Adding kickoff weather context..."):
                    for _, ev in board[["event_id","home_team","away_team","commence_time"]].drop_duplicates().iterrows():
                        home_abbr = TEAM_NAME_MAP.get(str(ev["home_team"]), str(ev["home_team"]))
                        away_abbr = TEAM_NAME_MAP.get(str(ev["away_team"]), str(ev["away_team"]))
                        sched = schedule_lookup.get((home_abbr, away_abbr), {})
                        roof = sched.get("roof", "")
                        location = str(sched.get("location", "")).lower()
                        neutral = bool(location and location not in {"home", "normal"})
                        weather_by_event[ev["event_id"]] = get_game_weather(
                            home_abbr, ev["commence_time"], roof=roof, neutral=neutral
                        )

                thresholds = {
                    "min_edge": min_edge_pp / 100.0,
                    "min_ev": min_ev_pct / 100.0,
                    "min_gap_mae": min_gap_mae,
                }
                result = scan_board(
                    board, game_ctx, player_index, stats, injuries,
                    frames, bundles, TEAM_NAME_MAP,
                    current_season, current_week, thresholds,
                    schedule_lookup=schedule_lookup,
                    weather_by_event=weather_by_event,
                    n_sims=n_sims,
                )

                if result.empty:
                    st.warning("No sportsbook players could be matched to the current nflverse player pool.")
                else:
                    qualifying = result[result["signal"] == "QUALIFIES"].copy()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Props analyzed", f"{len(result):,}")
                    c2.metric("Qualifying edges", f"{len(qualifying):,}")
                    c3.metric("Markets", f"{result['market'].nunique():,}")
                    c4.metric("Median books/line", f"{result['books'].median():.0f}")

                    show = qualifying if not qualifying.empty else result.head(25)
                    display_cols = [
                        "score","signal","player","team","opponent","market","side","line","best_price",
                        "projection","model_probability","market_novig_probability","probability_edge",
                        "ev_per_dollar","books","injury_status","wind_mph","temperature_f","holdout_mae"
                    ]
                    styled = show[display_cols].copy()
                    for c in ["model_probability","market_novig_probability","probability_edge","ev_per_dollar"]:
                        styled[c] = (styled[c] * 100).round(1).astype(str) + "%"
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                    st.download_button(
                        "Download scan CSV",
                        data=result.to_csv(index=False).encode(),
                        file_name="nfl_prop_edge_scan.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    st.caption(
                        "Score is a ranking heuristic, not a win probability. Props with Questionable/Doubtful/Out "
                        "status are automatically withheld from QUALIFIES even if the numeric edge is high."
                    )

with tab2:
    st.subheader("Single-prop analyzer")
    market = st.selectbox("Market", list(MARKETS), index=0, key="single_market")
    eligible = player_index[player_index["position_group"].isin(MARKETS[market]["positions"])]
    pname = st.selectbox("Player", eligible["player_display_name"].sort_values().tolist())
    prow = eligible[eligible["player_display_name"] == pname].iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    line = c1.number_input("Line", min_value=0.0, value=49.5, step=0.5)
    side = c2.selectbox("Side", ["Over", "Under"])
    price = c3.number_input("American odds", value=-110, step=5)
    opponent = c4.text_input("Opponent abbreviation", placeholder="DAL")

    if st.button("Analyze single prop", use_container_width=True):
        if not opponent:
            st.warning("Enter an opponent abbreviation.")
        else:
            stats_key = f"{min(seasons)}-{max(seasons)}-{len(stats)}"
            frame, bundle = cached_model(stats_key, market, _stats=stats, _schedules=schedules, _injuries=injuries)
            current, injury = make_current_features(
                stats, frame, injuries, market, prow["player_id"],
                opponent.upper(), prow["team"], current_season, current_week,
                live_context={}
            )
            res = evaluate_line(bundle, current, line, side, price, n_sims=n_sims)
            score = quality_score(res, bundle)

            a,b,c,d,e = st.columns(5)
            a.metric("Projection", f"{res['projection']:.1f}")
            b.metric(f"P({side})", f"{res['side_probability']*100:.1f}%")
            c.metric("Break-even", f"{res['break_even_probability']*100:.1f}%")
            d.metric("EV / $1", f"{res['ev_per_dollar']*100:+.1f}%")
            e.metric("Edge score", f"{score:.0f}/100")
            st.write(
                f"**Validation:** holdout MAE {bundle['mae']:.2f}, RMSE {bundle['rmse']:.2f}, "
                f"{bundle['n_test']:,} holdout observations."
            )
            if injury.get("report_status"):
                st.warning(
                    f"Injury report: {injury.get('report_status')} — "
                    f"{injury.get('report_primary_injury','')} | Practice: {injury.get('practice_status','')}"
                )

with tab3:
    st.subheader("Walk-forward-style holdout diagnostics")
    diag_market = st.selectbox("Diagnostic market", list(MARKETS), key="diag_market")
    if st.button("Build diagnostics", use_container_width=True):
        stats_key = f"{min(seasons)}-{max(seasons)}-{len(stats)}"
        frame, bundle = cached_model(stats_key, diag_market, _stats=stats, _schedules=schedules, _injuries=injuries)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Holdout MAE", f"{bundle['mae']:.2f}")
        c2.metric("Holdout RMSE", f"{bundle['rmse']:.2f}")
        c3.metric("Training rows", f"{bundle['n_train']:,}")
        c4.metric("Holdout rows", f"{bundle['n_test']:,}")

        diag = pd.DataFrame({
            "actual": bundle["holdout_actual"],
            "projection": bundle["holdout_pred"],
        })
        diag["error"] = diag["actual"] - diag["projection"]
        st.dataframe(diag.describe().T, use_container_width=True)
        st.caption(
            "The most important V3 upgrade is true closing-line backtesting: save every model snapshot and sportsbook "
            "line before kickoff, then track calibration, CLV and ROI by market."
        )

st.divider()
st.caption(
    "Research tool only. NFL player outcomes are noisy; no model can accurately guarantee individual props. "
    "Prefer large-sample validation and closing-line comparison over short winning streaks."
)
