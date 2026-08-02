# -*- coding: utf-8 -*-
"""
USDJPY デイトレ支援ツール - Phase1 プロトタイプ

必要ライブラリ:
    pip install yfinance pandas numpy requests

実行方法:
    python main.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta

# ============================================================
# 1. データ取得
# ============================================================

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """新しいyfinanceが返すMultiIndex列を単純な列名に変換する。"""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    return df


def fetch_usdjpy(interval: str, period: str) -> pd.DataFrame:
    """USDJPYのOHLCVデータを取得する。
    interval: '5m','1h','1d' など yfinance準拠
    period:   '7d','60d','1y' など yfinance準拠(5mは60日が上限)
    """
    df = yf.download("JPY=X", interval=interval, period=period,
                      progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"USDJPYデータ取得失敗: interval={interval}")
    df = _flatten_columns(df)
    return df


def fetch_dxy(period: str = "60d", interval: str = "1d") -> pd.DataFrame:
    """DXY(ドルインデックス)を取得する。"""
    df = yf.download("DX-Y.NYB", interval=interval, period=period,
                      progress=False, auto_adjust=True)
    df = _flatten_columns(df)
    return df


def fetch_us2y_yield(days: int = 90) -> pd.DataFrame:
    """FRED(米セントルイス連銀)から米2年債利回り(DGS2)を取得する。
    APIキー不要のCSVエンドポイントを使用。
    """
    end = datetime.now().date()
    start = end - timedelta(days=days)
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id=DGS2&cosd={start}&coed={end}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text), parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", "DGS2": "yield"})
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    df = df.dropna().set_index("date")
    return df


# ============================================================
# 2. テクニカル指標
# ============================================================

def calc_ema(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    return df[col].ewm(span=period, adjust=False).mean()


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_vwap_daily(df_5m: pd.DataFrame) -> pd.Series:
    """5分足データに対し、日毎にリセットされるVWAPを計算する。"""
    df = df_5m.copy()
    df["date"] = df.index.date
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"] = typical * df["volume"]
    df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
    df["cum_vol"] = df.groupby("date")["volume"].cumsum()
    vwap = df["cum_tp_vol"] / df["cum_vol"].replace(0, np.nan)
    return vwap


# ============================================================
# 3. スイングハイ/ロー・水平線抽出
# ============================================================

def find_swing_points(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """フラクタル形式でスイングハイ/ローを検出する。
    window本前後より高い/低いかで判定。
    """
    highs, lows = df["high"], df["low"]
    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)

    for i in range(window, len(df) - window):
        h_slice = highs.iloc[i - window: i + window + 1]
        l_slice = lows.iloc[i - window: i + window + 1]
        if highs.iloc[i] == h_slice.max() and (h_slice == h_slice.max()).sum() == 1:
            swing_high.iloc[i] = True
        if lows.iloc[i] == l_slice.min() and (l_slice == l_slice.min()).sum() == 1:
            swing_low.ilo
