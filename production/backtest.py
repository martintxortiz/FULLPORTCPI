import os
import glob
import pandas as pd
import numpy as np
from strategy import StrategyConfig, Bot7Strategy, PositionState

class Backtester:
    def __init__(self, data_dir: str, initial_capital: float = 10000.0):
        self.data_dir = data_dir
        self.config = StrategyConfig()
        self.bot = Bot7Strategy(self.config)
        self.initial_capital = initial_capital
        
    def run_backtest(self, symbol: str):
        # symbol is "BTC/USD"
        base_symbol = symbol.replace("/", "").replace("USD", "USDT")
        symbol_path = os.path.join(self.data_dir, base_symbol)
        if not os.path.exists(symbol_path):
            print(f"Data not found for {symbol} at path {symbol_path}")
            return
            
        files = sorted(glob.glob(os.path.join(symbol_path, "*.csv.gz")))
        
        equity = self.initial_capital
        cash = self.initial_capital
        pos = None
        
        trades = []
        
        for file in files:
            print(f">>> Running backtest on {file} for {symbol}...")
            df = pd.read_csv(file)
            
            # Assuming typical Binance/Alpaca CSV. Adjust column names if needed.
            # Using typical lowercase columns or renaming
            rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
            df = df.rename(columns=rename_map)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            elif "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df.set_index("datetime", inplace=True)
            elif "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
                
            required = ["Open", "High", "Low", "Close"]
            if not all(c in df.columns for c in required):
                print(f"Skipping {file}, missing required columns")
                continue
                
            df = df.sort_index()
            # Calculate indicators for the whole file at once for speed
            df = self.bot.add_indicators(df)
            
            total_bars = len(df)
            
            # Event loop simulation
            for i in range(self.config.min_bars_required, total_bars):
                if i % 5000 == 0:
                    pct = (i / total_bars) * 100
                    print(f"Progress: {pct:.1f}% | Cash: {cash:.2f} | Open Pos: {pos is not None}")
                    
                window = df.iloc[:i+1] # Up to current bar
                current_bar = window.iloc[-1]
                
                # Check Exits
                if pos is not None:
                    # To avoid looking ahead, check exit using the current bar's limits
                    actions = self.bot.check_exit(window, pos)
                    for act, pct, rsn in actions:
                        if act == "SELL_PCT":
                            sell_qty = pos.qty * pct
                            sell_price = float(current_bar["Close"]) # Simulated fill
                            proceeds = sell_qty * sell_price
                            cash += proceeds
                            pos.qty -= sell_qty
                            
                            trade_pnl = proceeds - (sell_qty * pos.entry)
                            trades.append({
                                "time": window.index[-1], "type": "SELL", "price": sell_price, 
                                "qty": sell_qty, "reason": rsn, "pnl": trade_pnl
                            })
                            
                            if pos.qty <= (pos.qty * 0.01): # basically 0
                                pos = None
                
                # Check Entries
                if pos is None:
                    # Only simulate equity == cash for simple backtesting
                    equity = cash
                    ok, rsn, pay = self.bot.diagnose_entry(window, equity, cash)
                    if ok:
                        entry_price = pay["entry"]
                        notional = pay["notional"]
                        qty = notional / entry_price
                        cash -= notional
                        
                        risk_price = pay["risk_price"]
                        
                        pos = PositionState(
                            entry=entry_price,
                            stop=pay["stop"],
                            target_1=entry_price + self.config.partial_r * risk_price,
                            target_2=entry_price + self.config.final_r * risk_price,
                            risk_price=risk_price,
                            qty=qty,
                            partial_done=False,
                            entry_bar_time=window.index[-1]
                        )
                        
                        trades.append({"time": window.index[-1], "type": "BUY", "price": entry_price, "qty": qty, "reason": "ENTRY", "pnl": 0})
                        
        print(f"\n--- Backtest Complete for {symbol} ---")
        print(f"Final Cash/Equity: ${cash:.2f}")
        total_rtn = ((cash - self.initial_capital) / self.initial_capital) * 100
        print(f"Total Return: {total_rtn:.2f}%")
        
        sell_trades = [t for t in trades if t["type"] == "SELL"]
        if sell_trades:
            wins = len([t for t in sell_trades if t["pnl"] > 0])
            total_sells = len(sell_trades)
            win_rate = (wins / total_sells) * 100
            print(f"Total Sell Trades: {total_sells}")
            print(f"Win Rate: {win_rate:.2f}%")
        else:
            print("No completed trades.")
            
        return trades

if __name__ == "__main__":
    b = Backtester("../data")
    b.run_backtest("BTC/USD")
