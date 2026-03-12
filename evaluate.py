import optuna
import numpy as np
import pandas as pd

def evaluate_params(df, symbol, friendly, regime, params):
    trades_df, equity_df, run_info = backtest_dataset(
        df=df,
        binance_symbol=symbol,
        friendly=friendly,
        regime=regime,
        starting_capital=STARTING_CAPITAL,
        fee_rate=params["FEE_RATE"],
        instant_entry=False,
        params=params,   # add this to your function signature
    )
    metrics = calc_metrics(trades_df, run_info)

    score = (
        0.40 * metrics["roi_pct"]
        + 0.30 * metrics["net_r"]
        + 0.20 * metrics["profit_factor"]
        + 0.10 * metrics["win_rate"]
        - 0.80 * abs(metrics["max_drawdown_pct"])
    )

    if metrics["total_trades"] < 30:
        score -= 1000
    if metrics["pass_status"] == "FAILED":
        score -= 5000

    return score, metrics

def objective(trial):
    params = {
        "MAX_RISK_PCT": trial.suggest_float("MAX_RISK_PCT", 0.001, 0.01),
        "MAX_DRAWDOWN_PCT": 0.10,
        "STRONG_BODY_PCT": trial.suggest_float("STRONG_BODY_PCT", 0.20, 0.60),
        "ADX_MIN": trial.suggest_int("ADX_MIN", 10, 40),
        "RSI_LONG_MAX": trial.suggest_int("RSI_LONG_MAX", 55, 75),
        "RSI_SHORT_MIN": trial.suggest_int("RSI_SHORT_MIN", 25, 45),
        "PULLBACK_LOOKBACK": trial.suggest_int("PULLBACK_LOOKBACK", 5, 30),
        "TIME_STOP_BARS": trial.suggest_int("TIME_STOP_BARS", 24, 144),
        "MAX_STOP_PCT": trial.suggest_float("MAX_STOP_PCT", 0.005, 0.05),
        "PARTIAL_R": trial.suggest_float("PARTIAL_R", 0.5, 2.0),
        "FINAL_R": trial.suggest_float("FINAL_R", 1.0, 4.0),
        "ATR_TRAIL_MULT": trial.suggest_float("ATR_TRAIL_MULT", 0.5, 4.0),
        "ATR_STOP_CAP": trial.suggest_float("ATR_STOP_CAP", 0.5, 4.0),
        "ATR_WICK_PAD": trial.suggest_float("ATR_WICK_PAD", 0.0, 0.5),
        "FEE_RATE": 0.0,
    }

    scores = []
    for ds in datasets_for_training:   # choose your training datasets here
        df_raw = load_data_from_path(ds["path"])
        df = add_indicators(df_raw)
        score, metrics = evaluate_params(df, ds["binance_symbol"], ds["friendly"], ds["regime"], params)
        scores.append(score)

    return float(np.median(scores))    # median is more robust than mean

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

print("Best params:", study.best_params)
print("Best score:", study.best_value)
