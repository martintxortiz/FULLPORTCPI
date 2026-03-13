import unittest
import pandas as pd
import numpy as np
from strategy import Bot7Strategy, StrategyConfig, PositionState

class TestBot7Strategy(unittest.TestCase):
    def setUp(self):
        self.config = StrategyConfig()
        self.bot = Bot7Strategy(self.config)

    def test_strong_bull_candle(self):
        # Create a series representing a candle
        # Body = |Close - Open| = |105 - 100| = 5
        # Range = High - Low = 110 - 90 = 20
        # Ratio = 5 / 20 = 0.25 >= 0.2 (strong_body_pct)
        # Close > Open -> True
        row = pd.Series({"Open": 100, "High": 110, "Low": 90, "Close": 105})
        self.assertTrue(self.bot.is_strong_bull_candle(row))
        
        # Weak bull candle
        row_weak = pd.Series({"Open": 100, "High": 110, "Low": 90, "Close": 102})
        self.assertFalse(self.bot.is_strong_bull_candle(row_weak))
        
        # Bear candle
        row_bear = pd.Series({"Open": 105, "High": 110, "Low": 90, "Close": 100})
        self.assertFalse(self.bot.is_strong_bull_candle(row_bear))

    def test_calc_notional(self):
        entry = 100.0
        stop = 98.0 # 2% stop loss
        equity = 10000.0
        cash = 10000.0
        
        # Risk is 1% of 10k = $100. 2% stop loss means we need $5000 notional.
        # Max cash per trade is 25% of 10k = $2500. So Notional is capped at $2500.
        notional = self.bot.calc_notional(entry, stop, equity, cash)
        self.assertEqual(notional, 2500.0)
        
        notional_wide_stop = self.bot.calc_notional(100.0, 90.0, 10000.0, 10000.0)
        self.assertEqual(notional_wide_stop, 0.0)
        
        # Valid stop: 1% (stop at 99.0)
        # Risk = $100. Stop = 1%. Notional = 100 / 0.01 = 10000.0
        # Wait, max cash is $2500 limit.
        notional_valid_stop = self.bot.calc_notional(100.0, 99.0, 10000.0, 10000.0)
        self.assertEqual(notional_valid_stop, 2500.0)

    def test_indicators(self):
        # Create a dummy dataframe with enough rows
        dates = pd.date_range("2024-01-01", periods=300, freq="5min")
        df = pd.DataFrame({
            "Open": np.random.uniform(100, 105, 300),
            "High": np.random.uniform(105, 110, 300),
            "Low": np.random.uniform(90, 95, 300),
            "Close": np.random.uniform(95, 105, 300),
            "Volume": np.random.uniform(1000, 5000, 300)
        }, index=dates)
        
        res = self.bot.add_indicators(df)
        self.assertIn("MA50", res.columns)
        self.assertIn("MA200", res.columns)
        self.assertIn("ATR", res.columns)
        self.assertIn("RSI", res.columns)
        self.assertIn("ADX", res.columns)
        
        # Ensure latest indicators are not nan
        self.assertFalse(np.isnan(res.iloc[-1]["MA200"]))

    def test_diagnose_entry_not_enough_bars(self):
        # Only 100 bars, min required is 250
        dates = pd.date_range("2024-01-01", periods=100, freq="5min")
        df = pd.DataFrame({"Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": 1000}, index=dates)
        df = self.bot.add_indicators(df)
        
        ok, rsn, pay = self.bot.diagnose_entry(df, 10000.0, 10000.0)
        self.assertFalse(ok)
        self.assertEqual(rsn, "Not enough bars")

if __name__ == "__main__":
    unittest.main()
