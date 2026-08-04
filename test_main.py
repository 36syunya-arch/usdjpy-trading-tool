# -*- coding: utf-8 -*-
"""USDJPY 支援ツールのネットワーク不要ユニットテスト。"""

import tempfile
import unittest
from unittest import mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import main


def make_ohlcv(index, closes):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.02,
            "low": closes - 0.02,
            "close": closes,
            "volume": np.zeros(len(closes)),
        },
        index=index,
    )


class DataQualityTests(unittest.TestCase):
    def test_yfinance_empty_response_is_retried(self):
        index = pd.date_range("2026-08-04", periods=2, freq="5min", tz="UTC")
        valid = make_ohlcv(index, [150.0, 150.1])

        class FakeYFinance:
            def __init__(self):
                self.calls = 0

            def download(self, *args, **kwargs):
                self.calls += 1
                return pd.DataFrame() if self.calls == 1 else valid

        fake = FakeYFinance()
        with mock.patch.object(main, "yf", fake), mock.patch.object(main.time, "sleep"):
            result = main._download_yfinance(
                "JPY=X", "USDJPY", interval="5m", period="5d"
            )

        self.assertEqual(fake.calls, 2)
        self.assertEqual(len(result), 2)

    def test_drop_unconfirmed_bar_accepts_numpy_mask(self):
        index = pd.date_range("2026-08-04 00:00", periods=3, freq="5min", tz="UTC")
        frame = make_ohlcv(index, [150.0, 150.1, 150.2])
        now = datetime(2026, 8, 4, 0, 12, tzinfo=timezone.utc)

        confirmed, info = main.drop_unconfirmed_bar(
            frame, interval_minutes=5, grace_minutes=1, now=now
        )

        self.assertEqual(len(confirmed), 2)
        self.assertEqual(info["dropped"], 1)
        self.assertEqual(float(confirmed["close"].iloc[-1]), 150.1)

    def test_freshness_uses_bar_end_not_start(self):
        index = pd.DatetimeIndex([pd.Timestamp("2026-08-04 00:00", tz="UTC")])
        frame = make_ohlcv(index, [150.0])
        now = datetime(2026, 8, 4, 0, 6, tzinfo=timezone.utc)

        result = main.check_data_freshness(frame, 5, now=now)

        self.assertTrue(result["fresh"])
        self.assertEqual(result["age_minutes"], 1.0)

    def test_vwap_fallback_is_applied_per_day(self):
        index = pd.to_datetime(
            [
                "2026-08-03 00:00:00+00:00",
                "2026-08-03 00:05:00+00:00",
                "2026-08-04 00:00:00+00:00",
                "2026-08-04 00:05:00+00:00",
            ]
        )
        frame = make_ohlcv(index, [100.0, 102.0, 200.0, 204.0])
        frame.loc[index[2:], "volume"] = [1.0, 3.0]

        result = main.calc_vwap_daily(frame)

        # 1日目は volume=0 なので単純平均代理。
        expected_day1 = np.mean(
            [
                (100.02 + 99.98 + 100.0) / 3,
                (102.02 + 101.98 + 102.0) / 3,
            ]
        )
        self.assertAlmostEqual(float(result.iloc[1]), expected_day1, places=8)
        # 2日目は volume 1:3 の加重平均。
        self.assertAlmostEqual(float(result.iloc[3]), 203.0, places=8)


class CotTests(unittest.TestCase):
    def test_missing_cot_columns_are_not_treated_as_zero(self):
        result = main.summarize_cot(
            [{"report_date_as_yyyy_mm_dd": "2026-07-28T00:00:00.000"}]
        )
        self.assertFalse(result["available"])

    def test_valid_cot_net_and_weekly_change(self):
        records = [
            {
                "market_and_exchange_names": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
                "report_date_as_yyyy_mm_dd": "2026-07-28T00:00:00.000",
                "lev_money_positions_long": "76752",
                "lev_money_positions_short": "178742",
            },
            {
                "lev_money_positions_long": "81033",
                "lev_money_positions_short": "177218",
            },
        ]

        result = main.summarize_cot(records)

        self.assertTrue(result["available"])
        self.assertEqual(result["net_position"], -101990.0)
        self.assertEqual(result["net_change"], -5805.0)


class DirectionAndRiskTests(unittest.TestCase):
    @staticmethod
    def evaluation(direction, score):
        return {"direction": direction, "score": {"合計": score}}

    def test_equal_scores_do_not_default_to_long(self):
        result = main.select_direction(
            self.evaluation("long", 82), self.evaluation("short", 82)
        )
        self.assertIsNone(result["direction"])
        self.assertTrue(result["ambiguous"])

    def test_small_direction_edge_is_ambiguous(self):
        result = main.select_direction(
            self.evaluation("long", 82),
            self.evaluation("short", 79),
            min_edge=5,
        )
        self.assertEqual(result["direction"], "long")
        self.assertTrue(result["ambiguous"])

    def test_lot_size_is_floored_before_display(self):
        result = main.calc_lot_size(
            account_balance=100_000,
            risk_percent=1.0,
            risk_pips=7.7,
            execution_buffer_pips=0.0,
        )
        self.assertEqual(result["recommended_lot"], 0.12)
        self.assertEqual(result["input_quantity_x1000"], 12)
        self.assertLessEqual(result["estimated_loss_jpy"], result["risk_amount_jpy"])

    def test_high_derived_level_is_not_long_stop_support(self):
        levels = [
            {
                "price": 149.9,
                "touch_count": 3,
                "from_highs": 3,
                "from_lows": 0,
                "timeframe": "1h",
            }
        ]

        score = main.calc_key_level_score(150.0, levels, atr=0.05, direction="long")
        plan = main.suggest_sl_tp(150.0, "long", atr=0.05, key_levels=levels)

        self.assertEqual(score["score"], 2)
        self.assertFalse(plan["valid"])


class BreakoutAndNotificationTests(unittest.TestCase):
    def test_tiny_range_excess_is_not_confirmed_breakout(self):
        index = pd.date_range("2026-08-04 00:00", periods=30, freq="5min", tz="UTC")
        closes = np.linspace(150.00, 150.10, len(index))
        frame = make_ohlcv(index, closes)
        previous_high = float(frame["high"].iloc[-25:-1].max())
        # 直前高値を0.05pipだけ上回るが、最低バッファ0.1pipには届かない。
        frame.loc[index[-1], ["open", "close"]] = previous_high + 0.0005
        frame.loc[index[-1], "high"] = previous_high + 0.0005

        result = main.judge_session_breakout(
            frame, direction="long", lookback=24, atr=0.05
        )

        self.assertNotEqual(result["state"], "レンジを明確にブレイク")
        self.assertLess(result["score"], 15)

    def test_notification_cooldown_and_direction_change(self):
        now = datetime(2026, 8, 4, 0, 0, tzinfo=timezone.utc)
        state = {
            "last_direction": "long",
            "last_notified_at": (now - timedelta(minutes=20)).isoformat(),
        }

        same = main.notification_allowed(state, "long", 180, now=now)
        opposite = main.notification_allowed(state, "short", 180, now=now)

        self.assertFalse(same["allowed"])
        self.assertTrue(opposite["allowed"])

    def test_notification_state_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            expected = {
                "last_direction": "short",
                "last_notified_at": "2026-08-04T00:00:00+00:00",
            }
            main.save_notification_state(path, expected)
            self.assertEqual(main.load_notification_state(path), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
