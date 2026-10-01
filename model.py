
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import MARKETS
from odds import american_implied_probability, profit_per_dollar

def _new_model(count_market=False):
    return HistGradientBoostingRegressor(
        loss="poisson" if count_market else "squared_error",
        learning_rate=0.045,
        max_iter=350,
        max_leaf_nodes=15,
        min_samples_leaf=18,
        l2_regularization=1.5,
        random_state=42,
    )

def fit_market_model(frame, market_name):
    target = frame.attrs["target_col"]
    features = frame.attrs["feature_cols"]
    cfg = MARKETS[market_name]

    time_key = frame["season"].astype(int) * 100 + frame["week"].astype(int)
    times = np.sort(time_key.unique())
    if len(times) < 5:
        raise ValueError("Not enough historical weeks to validate the model.")
    cutoff = times[max(1, int(len(times) * 0.80)) - 1]
    train = frame[time_key <= cutoff].copy()
    test = frame[time_key > cutoff].copy()
    if len(test) < 25:
        split = max(1, int(len(frame) * 0.80))
        train, test = frame.iloc[:split].copy(), frame.iloc[split:].copy()

    model = _new_model(cfg.get("count_market", False))
    model.fit(train[features], train[target])
    pred = np.clip(model.predict(test[features]), 0, None)
    actual = test[target].to_numpy(dtype=float)
    residuals = actual - pred

    bundle = {
        "features": features,
        "target": target,
        "market_name": market_name,
        "residuals": residuals[np.isfinite(residuals)],
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(mean_squared_error(actual, pred) ** 0.5),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "holdout_actual": actual,
        "holdout_pred": pred,
    }

    live = _new_model(cfg.get("count_market", False))
    live.fit(frame[features], frame[target])
    bundle["model"] = live
    return bundle

def evaluate_line(bundle, current_features, line, side, american_odds, n_sims=25000):
    projection = float(max(0.0, bundle["model"].predict(current_features[bundle["features"]])[0]))
    residuals = np.asarray(bundle["residuals"], dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    rng = np.random.default_rng(20260930)

    if len(residuals) >= 25:
        draws = projection + rng.choice(residuals, size=n_sims, replace=True)
    else:
        draws = rng.normal(projection, max(bundle["rmse"], 1.0), size=n_sims)

    draws = np.clip(draws, 0, None)
    p_over = float((draws > line).mean())
    p_side = p_over if side == "Over" else 1.0 - p_over
    be = american_implied_probability(american_odds)
    ev = p_side * profit_per_dollar(american_odds) - (1.0 - p_side)

    return {
        "projection": projection,
        "p_over": p_over,
        "side_probability": p_side,
        "break_even_probability": be,
        "probability_edge": p_side - be,
        "ev_per_dollar": ev,
        "projection_gap": (projection - line) if side == "Over" else (line - projection),
    }

def quality_score(result, bundle, consensus_side_probability=None):
    # 0-100 ranking heuristic; it is not a calibrated win probability.
    edge = max(-0.20, min(0.20, result["probability_edge"]))
    ev = max(-0.30, min(0.40, result["ev_per_dollar"]))
    gap_units = result["projection_gap"] / max(bundle["mae"], 0.25)
    gap_units = max(-2.0, min(2.0, gap_units))

    score = 50 + 95 * edge + 35 * ev + 8 * gap_units
    if consensus_side_probability is not None and np.isfinite(consensus_side_probability):
        model_market_gap = result["side_probability"] - consensus_side_probability
        score += 18 * max(-0.20, min(0.20, model_market_gap))
    return float(max(0.0, min(100.0, score)))
