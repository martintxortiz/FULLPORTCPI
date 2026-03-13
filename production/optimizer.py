import os
import glob
import pandas as pd
import numpy as np
import optuna

from strategy import StrategyConfig, Bot7Strategy, PositionState

class FastBacktester:
    def __init__(self, data_dir: str, config: StrategyConfig, initial_capital: float = 10000.0):
        self.data_dir = data_dir
        self.config = config
        self.bot = Bot7Strategy(self.config)
        self.initial_capital = initial_capital
        
    def run_fast(self, symbol: str) -> tuple[float, float]:
        """
        Returns (Total Return %, Max Drawdown %)
        """
        base_symbol = symbol.replace("/", "").replace("USD", "USDT")
        symbol_path = os.path.join(self.data_dir, base_symbol)
        if not os.path.exists(symbol_path):
            return 0.0, 0.0
            
        files = sorted(glob.glob(os.path.join(symbol_path, "*.csv.gz")))
        
        equity = self.initial_capital
        cash = self.initial_capital
        pos = None
        
        peak_equity = self.initial_capital
        max_drawdown_pct = 0.0
        
        for file in files:
            df = pd.read_csv(file)
            rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
            df = df.rename(columns=rename_map)
            
            # Add indicators based on the strategy
            df = self.bot.add_indicators(df)
            total_bars = len(df)
            
            df_open = df["Open"].values
            df_high = df["High"].values
            df_low = df["Low"].values
            df_close = df["Close"].values
            df_atr = df["ATR"].values
            
            for i in range(self.config.min_bars_required, total_bars):
                window = df.iloc[:i+1]
                current_low = df_low[i]
                current_high = df_high[i]
                current_close = df_close[i]
                current_atr = df_atr[i]
                
                # Check Exits
                if pos is not None:
                    if current_low <= pos.stop:
                        proceeds = pos.qty * current_close
                        cash += proceeds
                        pos = None
                    elif current_high >= pos.target_2:
                        proceeds = pos.qty * current_close
                        cash += proceeds
                        pos = None
                    else:
                        if not pos.partial_done and current_high >= pos.target_1:
                            half = pos.qty * 0.5
                            proceeds = half * current_close
                            cash += proceeds
                            pos.qty -= half
                            pos.partial_done = True
                            pos.stop = pos.entry
                            
                        # Trailing Stop
                        if pos is not None:
                            if pos.partial_done and current_atr > 0:
                                new_stop = current_close - self.config.atr_trail_mult * current_atr
                                if new_stop > pos.stop:
                                    pos.stop = new_stop

                # Timeout fallback
                if pos is not None:
                    bars_held = i - pos.entry_bar_idx
                    if bars_held > self.config.time_stop_bars and current_close < pos.entry + 0.3 * pos.risk_price:
                        proceeds = pos.qty * current_close
                        cash += proceeds
                        pos = None
                        
                # Check Entries
                if pos is None:
                    equity_calc = cash
                    ok, _, pay = self.bot.diagnose_entry(window, equity_calc, cash)
                    if ok:
                        entry_price = pay["entry"]
                        notional = pay["notional"]
                        qty = notional / entry_price
                        cash -= notional
                        risk_price = pay["risk_price"]
                        
                        pos = PositionState(
                            entry=entry_price, stop=pay["stop"],
                            target_1=entry_price + self.config.partial_r * risk_price,
                            target_2=entry_price + self.config.final_r * risk_price,
                            risk_price=risk_price, qty=qty, partial_done=False, entry_bar_time=None
                        )
                        pos.entry_bar_idx = i
                        
                # Track Drawdown
                current_unrealized = pos.qty * current_close if pos is not None else 0.0
                current_total_equity = cash + current_unrealized
                
                if current_total_equity > peak_equity:
                    peak_equity = current_total_equity
                    
                drawdown = (peak_equity - current_total_equity) / peak_equity * 100
                if drawdown > max_drawdown_pct:
                    max_drawdown_pct = drawdown

        # Final Cash is the ending metric if we close everything out flat conceptually
        total_rtn_pct = ((cash - self.initial_capital) / self.initial_capital) * 100
        return total_rtn_pct, max_drawdown_pct

def objective(trial):
    cfg = StrategyConfig()
    
    # Suggest parameter values
    cfg.strong_body_pct = trial.suggest_float("strong_body_pct", 0.1, 0.4, step=0.05)
    cfg.adx_min = trial.suggest_float("adx_min", 10.0, 30.0, step=2.5)
    cfg.rsi_long_max = trial.suggest_float("rsi_long_max", 60.0, 85.0, step=5.0)
    cfg.pullback_lookback = trial.suggest_int("pullback_lookback", 5, 20)
    cfg.time_stop_bars = trial.suggest_int("time_stop_bars", 48, 192, step=12) 
    cfg.partial_r = trial.suggest_float("partial_r", 0.8, 1.5, step=0.1)
    cfg.final_r = trial.suggest_float("final_r", 1.5, 3.5, step=0.5)
    cfg.atr_trail_mult = trial.suggest_float("atr_trail_mult", 1.0, 3.0, step=0.5)
    cfg.atr_stop_cap = trial.suggest_float("atr_stop_cap", 1.5, 3.5, step=0.5)
    
    # Run fast Multi-File Backtester
    tester = FastBacktester("../data", config=cfg)
    total_rtn, max_dd = tester.run_fast("BTC/USD")
    
    return total_rtn, max_dd

if __name__ == "__main__":
    import logging
    import sys
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    # Multi-objective optimization: (Maximize Return, Minimize Drawdown)
    study = optuna.create_study(directions=["maximize", "minimize"])
    
    print("Starting parameter optimization across all BTC/USD periods...")
    print("Objectives: 1. Maximize Total Return (%) | 2. Minimize Max Drawdown (%)")
    
    study.optimize(objective, n_trials=30) # Adjust n_trials based on how long it takes
    
    print("\nOptimization finished.")
    print(f"Number of Pareto-optimal solutions: {len(study.best_trials)}")
    
    print("\nBest Pareto Front Trials:")
    for i, trial in enumerate(study.best_trials):
        rtn, dd = trial.values
        print(f"  --- Solution {i+1} ---")
        print(f"  Return: {rtn:.2f}% | Max Drawdown: {dd:.2f}%")
        for k, v in trial.params.items():
            print(f"    {k}: {v}")
