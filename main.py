# -*- coding: utf-8 -*-
"""USDJPY デイトレ支援ツール - Phase 1 安全修正版。

必要ライブラリ:
    pip install -r requirements.txt

このツールは売買を自動執行せず、条件が揃った場合に Discord へ通知する。
FRED の米2年債利回りと CFTC COT は日次・週次の補助情報であり、
リアルタイムの注文判断データではない。
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # 単体テスト時に純粋関数だけ読み込めるようにする
    requests = None

try:
    import yfinance as yf
except ImportError:  # 同上
    yf = None


# ============================================================
# 0. 設定
# ============================================================

ACCOUNT_BALANCE_JPY = float(os.environ.get("ACCOUNT_BALANCE_JPY", "272778"))
RISK_PERCENT = float(os.environ.get("RISK_PERCENT", "3.5"))
EXECUTION_BUFFER_PIPS = float(os.environ.get("EXECUTION_BUFFER_PIPS", "1.0"))
ENTRY_SCORE_THRESHOLD = int(os.environ.get("ENTRY_SCORE_THRESHOLD", "80"))
MIN_DIRECTION_SCORE_EDGE = int(os.environ.get("MIN_DIRECTION_SCORE_EDGE", "5"))
MAX_RISK_PIPS_FOR_DAYTRADE = float(
    os.environ.get("MAX_RISK_PIPS_FOR_DAYTRADE", "60")
)
NOTIFICATION_COOLDOWN_MINUTES = int(
    os.environ.get("NOTIFICATION_COOLDOWN_MINUTES", "180")
)
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
# ログ用チャンネル（見送りを含む全実行結果を記録。通知はミュート推奨）
DISCORD_LOG_WEBHOOK_URL = os.environ.get("DISCORD_LOG_WEBHOOK_URL", "")
SIGNAL_STATE_FILE = Path(os.environ.get("SIGNAL_STATE_FILE", ".signal_state.json"))

HTTP_HEADERS = {
    "User-Agent": "USDJPY-Daytrade-Support/1.0 (personal research tool)",
}


def _require_dependency(module: Any, package_name: str) -> None:
    if module is None:
        raise RuntimeError(
            f"必要ライブラリ '{package_name}' がありません。"
            "先に pip install -r requirements.txt を実行してください。"
        )


# ============================================================
# 1. データ取得
# ============================================================

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance の MultiIndex 列を OHLCV の単純な列名へ変換する。"""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda value: str(value).lower().strip())
    return df


def _normalize_ohlcv(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """OHLCV の列・型・並びを検証し、後段が安全に扱える形へ揃える。"""
    if df.empty:
        raise RuntimeError(f"{label}データ取得失敗: データが空です")

    out = _flatten_columns(df).sort_index()
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise RuntimeError(f"{label}データの必須列が不足しています: {missing}")
    if "volume" not in out.columns:
        out["volume"] = 0.0

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[~out.index.duplicated(keep="last")]
    if out.empty:
        raise RuntimeError(f"{label}データ取得失敗: 有効なOHLCがありません")
    return out


def _download_yfinance(ticker: str, label: str, **kwargs: Any) -> pd.DataFrame:
    """Yahoo の一時的な429・空レスポンスを短い間隔で再試行する。"""
    _require_dependency(yf, "yfinance")
    last_error: Optional[Exception] = None
    retry_delays = (2, 5)
    for attempt in range(len(retry_delays) + 1):
        try:
            frame = yf.download(
                ticker,
                progress=False,
                auto_adjust=True,
                threads=False,
                **kwargs,
            )
            if not frame.empty:
                return frame
            last_error = RuntimeError("Yahoo Finance が空データを返しました")
        except Exception as error:
            last_error = error
        if attempt < len(retry_delays):
            time.sleep(retry_delays[attempt])
    raise RuntimeError(
        f"{label}データ取得失敗（Yahoo Financeの一時制限を含む）: {last_error}"
    )


def fetch_usdjpy(interval: str, period: str) -> pd.DataFrame:
    """Yahoo Finance から USDJPY の OHLCV を取得する。"""
    df = _download_yfinance(
        "JPY=X",
        f"USDJPY(interval={interval})",
        interval=interval,
        period=period,
    )
    return _normalize_ohlcv(df, f"USDJPY(interval={interval})")


def fetch_dxy(period: str = "60d", interval: str = "1d") -> pd.DataFrame:
    """Yahoo Finance から DXY（ドルインデックス）を取得する。"""
    df = _download_yfinance(
        "DX-Y.NYB",
        f"DXY(interval={interval})",
        interval=interval,
        period=period,
    )
    return _normalize_ohlcv(df, f"DXY(interval={interval})")


def fetch_us2y_yield(days: int = 90, timeout: int = 30,
                      retry_delays: tuple = (2, 5)) -> pd.DataFrame:
    """FRED の日次系列 DGS2（米2年債利回り）を取得する。

    FRED は混雑時に応答が遅れることがあるため、タイムアウトを長めに取り、
    一時的な失敗に対してリトライする。それでも取得できない場合は例外を送出し、
    呼び出し側で「データなし」として扱う（加点せず、ハードゲートで通知停止）。
    """
    _require_dependency(requests, "requests")
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id=DGS2&cosd={start}&coed={end}"
    )

    last_error = None
    for attempt in range(len(retry_delays) + 1):
        try:
            response = requests.get(url, timeout=timeout, headers=HTTP_HEADERS)
            response.raise_for_status()
            df = pd.read_csv(
                StringIO(response.text), parse_dates=["observation_date"]
            )
            df = df.rename(
                columns={"observation_date": "date", "DGS2": "yield"}
            )
            if "yield" not in df.columns:
                raise RuntimeError("FRED DGS2 の yield 列を取得できませんでした")
            df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
            return df.dropna(subset=["yield"]).set_index("date").sort_index()
        except Exception as error:
            last_error = error
            if attempt < len(retry_delays):
                print(
                    f"[FRED取得リトライ {attempt + 1}/{len(retry_delays)}] {error}"
                )
                time.sleep(retry_delays[attempt])

    raise RuntimeError(f"FRED DGS2 の取得に失敗しました: {last_error}")


# ============================================================
# 2. テクニカル指標
# ============================================================

def calc_ema(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    return df[col].ewm(span=period, adjust=False).mean()


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def calc_vwap_daily(df_5m: pd.DataFrame) -> pd.Series:
    """日ごとにリセットする VWAP（出来高ゼロの日は時間加重平均代理）。

    スポットFXには中央集権的な出来高がない。Yahoo の volume が日単位で
    すべてゼロなら、その日だけ typical price の累積単純平均へ切り替える。
    全期間合計で判定すると、一部の日だけ volume が欠けた場合に NaN となる
    ため、必ず日単位で判定する。
    """
    if df_5m.empty:
        return pd.Series(dtype=float, index=df_5m.index, name="vwap")

    df = df_5m.copy()
    index = df.index
    if index.tz is not None:
        session_dates = index.tz_convert("UTC").date
    else:
        session_dates = index.date
    session = pd.Series(session_dates, index=index, name="session")

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0.0)
    volume = volume.clip(lower=0.0)

    cumulative_count = session.groupby(session).cumcount() + 1
    proxy = typical.groupby(session).cumsum() / cumulative_count

    cumulative_volume = volume.groupby(session).cumsum()
    cumulative_tp_volume = (typical * volume).groupby(session).cumsum()
    day_has_volume = volume.groupby(session).transform("sum") > 0

    result = proxy.copy()
    weighted_mask = day_has_volume & (cumulative_volume > 0)
    result.loc[weighted_mask] = (
        cumulative_tp_volume.loc[weighted_mask]
        / cumulative_volume.loc[weighted_mask]
    )
    result.name = "vwap"
    return result


# ============================================================
# 3. スイングハイ / ロー・水平線
# ============================================================

def find_swing_points(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """前後 window 本より一意に高い / 低い点を確定スイングとする。"""
    highs, lows = df["high"], df["low"]
    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)

    for i in range(window, len(df) - window):
        high_slice = highs.iloc[i - window : i + window + 1]
        low_slice = lows.iloc[i - window : i + window + 1]
        if (
            highs.iloc[i] == high_slice.max()
            and int((high_slice == high_slice.max()).sum()) == 1
        ):
            swing_high.iloc[i] = True
        if (
            lows.iloc[i] == low_slice.min()
            and int((low_slice == low_slice.min()).sum()) == 1
        ):
            swing_low.iloc[i] = True

    result = df.copy()
    result["swing_high"] = swing_high
    result["swing_low"] = swing_low
    return result


def extract_key_levels(
    df_with_swings: pd.DataFrame,
    lookback: int = 100,
    top_n: int = 5,
    timeframe: str = "unknown",
) -> list[dict]:
    """直近スイングを ATR 幅でクラスタリングし、由来を保持する。"""
    recent = df_with_swings.tail(lookback)
    if recent.empty:
        return []

    atr = calc_atr(recent, 14).iloc[-1]
    if pd.isna(atr) or atr <= 0:
        atr = float(recent["close"].iloc[-1]) * 0.0005
    cluster_width = max(float(atr) * 0.5, 0.001)

    points: list[dict] = []
    for price in recent.loc[recent["swing_high"], "high"].tolist():
        points.append({"price": float(price), "origin": "swing_high"})
    for price in recent.loc[recent["swing_low"], "low"].tolist():
        points.append({"price": float(price), "origin": "swing_low"})
    if not points:
        return []

    points.sort(key=lambda item: item["price"])
    clusters: list[list[dict]] = []
    current = [points[0]]
    cluster_start = points[0]["price"]
    for point in points[1:]:
        if point["price"] - cluster_start <= cluster_width:
            current.append(point)
        else:
            clusters.append(current)
            current = [point]
            cluster_start = point["price"]
    clusters.append(current)

    levels = []
    for cluster in clusters:
        prices = [item["price"] for item in cluster]
        from_highs = sum(item["origin"] == "swing_high" for item in cluster)
        from_lows = sum(item["origin"] == "swing_low" for item in cluster)
        levels.append(
            {
                "price": round(float(np.mean(prices)), 3),
                "touch_count": len(cluster),
                "from_highs": int(from_highs),
                "from_lows": int(from_lows),
                "timeframe": timeframe,
            }
        )

    levels.sort(key=lambda item: item["touch_count"], reverse=True)
    return levels[:top_n]


def assign_level_roles(levels: list[dict], current_price: float) -> list[dict]:
    """現在値より下を support、上（同値を含む）を resistance とする。"""
    output = []
    for level in levels:
        item = dict(level)
        item["role"] = "support" if level["price"] < current_price else "resistance"
        item["distance"] = abs(current_price - level["price"])
        output.append(item)
    return output


def combine_key_levels(
    long_term_levels: list[dict], short_term_levels: list[dict], top_n: int = 8
) -> list[dict]:
    """1時間足と5分足の候補を統合する（由来・時間軸は維持）。"""
    combined = [dict(level) for level in long_term_levels + short_term_levels]
    if not combined:
        return []
    # 同じタッチ数なら上位足を優先する。
    combined.sort(
        key=lambda item: (
            item.get("touch_count", 0),
            1 if item.get("timeframe") == "1h" else 0,
        ),
        reverse=True,
    )
    return combined[:top_n]


# ============================================================
# 4. データ品質・トレンド・短期判定
# ============================================================

def _as_utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")


def drop_unconfirmed_bar(
    df: pd.DataFrame,
    interval_minutes: int,
    grace_minutes: int = 1,
    now: Optional[datetime] = None,
) -> tuple[pd.DataFrame, dict]:
    """形成途中の足を除外する。

    pandas の比較結果は環境により ndarray となるため `.values` を使わず、
    明示的に NumPy の bool 配列へ変換する。
    """
    if df.empty:
        return df.copy(), {"dropped": 0, "remaining": 0, "reason": "データが空"}

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)

    index_utc = _as_utc_index(pd.DatetimeIndex(df.index))
    bar_end = index_utc + pd.Timedelta(minutes=interval_minutes)
    cutoff = current - pd.Timedelta(minutes=grace_minutes)
    confirmed_mask = np.asarray(bar_end <= cutoff, dtype=bool)

    output = df.loc[confirmed_mask].copy()
    return output, {
        "dropped": int((~confirmed_mask).sum()),
        "remaining": len(output),
    }


def check_data_freshness(
    df: pd.DataFrame,
    interval_minutes: int,
    max_age_multiplier: float = 3.0,
    grace_minutes: int = 1,
    now: Optional[datetime] = None,
) -> dict:
    """最新確定足の終了時刻からの経過時間でデータ鮮度を判定する。"""
    if df.empty:
        return {"fresh": False, "age_minutes": None, "reason": "データが空"}

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)

    last_start = pd.Timestamp(df.index[-1])
    if last_start.tzinfo is None:
        last_start = last_start.tz_localize("UTC")
    else:
        last_start = last_start.tz_convert("UTC")
    last_end = last_start + pd.Timedelta(minutes=interval_minutes)
    age = (pd.Timestamp(current) - last_end).total_seconds() / 60.0
    max_age = interval_minutes * max_age_multiplier
    fresh = -grace_minutes <= age <= max_age
    return {
        "fresh": bool(fresh),
        "age_minutes": round(float(age), 1),
        "max_allowed_minutes": round(float(max_age), 1),
    }


def resample_ohlcv_complete(
    df: pd.DataFrame, rule: str, bars_required: int
) -> pd.DataFrame:
    """上位足へ変換し、構成本数が不足するバーを除外する。"""
    aggregation = df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    counts = df["close"].resample(rule).count()
    return aggregation.loc[counts >= bars_required].dropna()


def judge_trend(
    df_with_swings: pd.DataFrame,
    lookback_swings: int = 4,
    epsilon: Optional[float] = None,
) -> dict:
    """確定スイング高値・安値の明確な切り上げ / 切り下げを判定する。"""
    swing_highs = df_with_swings.loc[
        df_with_swings["swing_high"], "high"
    ].tail(lookback_swings)
    swing_lows = df_with_swings.loc[
        df_with_swings["swing_low"], "low"
    ].tail(lookback_swings)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"trend": "不明", "reason": "スイング点不足"}

    if epsilon is None:
        atr = calc_atr(df_with_swings, 14).iloc[-1]
        if pd.isna(atr) or atr <= 0:
            atr = float(df_with_swings["close"].iloc[-1]) * 0.0005
        epsilon = float(atr) * 0.05

    high_diff = swing_highs.diff().dropna()
    low_diff = swing_lows.diff().dropna()
    higher_high = bool((high_diff > epsilon).all())
    higher_low = bool((low_diff > epsilon).all())
    lower_high = bool((high_diff < -epsilon).all())
    lower_low = bool((low_diff < -epsilon).all())

    if higher_high and higher_low:
        return {"trend": "上昇トレンド", "reason": "高値・安値ともに明確に切り上げ"}
    if lower_high and lower_low:
        return {"trend": "下降トレンド", "reason": "高値・安値ともに明確に切り下げ"}
    return {
        "trend": "レンジ",
        "reason": "高値・安値の方向不一致または変化が微小",
    }


def detect_breakout(
    df: pd.DataFrame,
    lookback: int = 10,
    buffer_atr_ratio: float = 0.05,
) -> Optional[str]:
    """直近レンジを終値が ATR バッファ込みで抜けたか判定する。"""
    if len(df) < lookback + 2:
        return None
    window = df.iloc[-lookback - 1 : -1]
    recent_low = float(window["low"].min())
    recent_high = float(window["high"].max())
    last_close = float(df["close"].iloc[-1])

    atr = calc_atr(df, 14).iloc[-1]
    if pd.isna(atr) or atr <= 0:
        atr = last_close * 0.0005
    buffer = max(float(atr) * buffer_atr_ratio, 0.001)

    if last_close < recent_low - buffer:
        return "down"
    if last_close > recent_high + buffer:
        return "up"
    return None


def judge_trend_with_breakout(
    df_with_swings: pd.DataFrame,
    df_raw: pd.DataFrame,
    lookback_swings: int = 4,
    breakout_lookback: int = 10,
) -> dict:
    """確定スイング判定を、ATRバッファ付き直近ブレイクで補正する。"""
    swing_result = judge_trend(df_with_swings, lookback_swings)
    breakout = detect_breakout(df_raw, breakout_lookback)
    breakout_trend = {"down": "下降トレンド", "up": "上昇トレンド"}.get(
        breakout
    )
    if breakout_trend and breakout_trend != swing_result["trend"]:
        return {
            "trend": breakout_trend,
            "reason": (
                f"直近{breakout_lookback}本のレンジをATRバッファ込みでブレイク"
                "（スイング確定待ちを補正）"
            ),
        }
    return swing_result


def judge_environment(trend_1h: dict, trend_4h: dict, trend_1d: dict) -> dict:
    """1h / 4h / 日足の一致度を、逆行と不明を区別して評価する。"""
    trends = [trend_1h["trend"], trend_4h["trend"], trend_1d["trend"]]
    up = trends.count("上昇トレンド")
    down = trends.count("下降トレンド")
    unknown = trends.count("不明")

    if up == 3:
        return {"score": 20, "direction": "long", "detail": "1h/4h/日足すべて上昇一致"}
    if down == 3:
        return {"score": 20, "direction": "short", "detail": "1h/4h/日足すべて下降一致"}
    if up >= 1 and down >= 1:
        return {"score": 0, "direction": None, "detail": "上昇と下降が混在（明確な矛盾）"}
    if up == 2:
        score = 12 if unknown else 15
        return {"score": score, "direction": "long", "detail": "2つが上昇一致・逆行なし"}
    if down == 2:
        score = 12 if unknown else 15
        return {"score": score, "direction": "short", "detail": "2つが下降一致・逆行なし"}
    if up == 1:
        score = 6 if unknown else 8
        return {"score": score, "direction": "long", "detail": "1つのみ上昇・逆行なし"}
    if down == 1:
        score = 6 if unknown else 8
        return {"score": score, "direction": "short", "detail": "1つのみ下降・逆行なし"}
    if unknown:
        return {"score": 0, "direction": None, "detail": "時間足データが不足"}
    return {"score": 3, "direction": None, "detail": "全時間足レンジ（方向感なし）"}


def calc_vwap_alignment(df_5m: pd.DataFrame, direction: str, atr: float) -> dict:
    last_close = float(df_5m["close"].iloc[-1])
    last_vwap = df_5m["vwap"].iloc[-1]
    if pd.isna(last_vwap) or pd.isna(atr) or atr <= 0:
        return {"score": 5, "diff": None, "ratio": None, "note": "データ不足のため中立"}

    diff = last_close - float(last_vwap)
    directional_diff = diff if direction == "long" else -diff
    ratio = directional_diff / atr
    if ratio >= 0.5:
        score = 10
    elif ratio >= 0.15:
        score = 7
    elif ratio >= -0.15:
        score = 5
    elif ratio >= -0.5:
        score = 2
    else:
        score = 0
    return {"score": score, "diff": round(diff, 3), "ratio": round(float(ratio), 2)}


def _three_state(value: float, epsilon: float) -> str:
    if value > epsilon:
        return "up"
    if value < -epsilon:
        return "down"
    return "flat"


def calc_trend_score(
    trend_result: dict, df_1h: pd.DataFrame, direction: str
) -> dict:
    """1時間足のスイング、EMA位置、EMA傾きを段階評価する。"""
    ema20 = calc_ema(df_1h, 20)
    atr_1h = calc_atr(df_1h, 14).iloc[-1]
    if pd.isna(atr_1h) or atr_1h <= 0:
        atr_1h = max(float(df_1h["close"].iloc[-1]) * 0.0005, 1e-6)
    epsilon = float(atr_1h) * 0.05

    last_close = float(df_1h["close"].iloc[-1])
    last_ema = float(ema20.iloc[-1])
    previous_ema = float(ema20.iloc[-6]) if len(ema20) >= 6 else float(ema20.iloc[0])
    expected = "up" if direction == "long" else "down"
    price_state = _three_state(last_close - last_ema, epsilon)
    slope_state = _three_state(last_ema - previous_ema, epsilon)

    price_ok = price_state == expected
    slope_ok = slope_state == expected
    price_against = price_state not in (expected, "flat")
    slope_against = slope_state not in (expected, "flat")

    swing_trend = trend_result["trend"]
    swing_matches = (
        swing_trend == "上昇トレンド" and direction == "long"
    ) or (swing_trend == "下降トレンド" and direction == "short")
    swing_opposes = (
        swing_trend == "上昇トレンド" and direction == "short"
    ) or (swing_trend == "下降トレンド" and direction == "long")

    if swing_opposes or (price_against and slope_against):
        score, detail = 0, f"方向と逆行(price={price_state}, slope={slope_state})"
    elif swing_matches and price_ok and slope_ok:
        score, detail = 15, "スイング・EMA位置・EMA傾きすべて方向一致"
    elif swing_matches and not price_against and not slope_against:
        score, detail = 12, "スイング判定が方向一致（EMAは逆行なし）"
    elif price_ok and slope_ok:
        score, detail = 10, "EMA20の位置・傾きが方向一致"
    elif price_ok or slope_ok:
        score, detail = 6, f"EMAの片方のみ一致(price={price_state}, slope={slope_state})"
    elif price_state == "flat" and slope_state == "flat":
        score, detail = 3, "完全な横ばい（方向感なし）"
    else:
        score, detail = 3, f"方向感が薄い(price={price_state}, slope={slope_state})"
    return {"score": score, "detail": detail, "swing_trend": swing_trend}


def calc_key_level_score(
    latest_price: float,
    key_levels: list[dict],
    atr: float,
    direction: str,
) -> dict:
    """SLに使える由来の水平線だけを「背後の支え」として加点する。"""
    if not key_levels or pd.isna(atr) or atr <= 0:
        return {
            "score": 5,
            "nearest_distance": None,
            "ratio": None,
            "detail": "水平線またはATRが取得不可",
            "behind": None,
            "ahead": None,
        }

    leveled = assign_level_roles(key_levels, latest_price)
    if direction == "long":
        behind = [
            level
            for level in leveled
            if level["role"] == "support" and level.get("from_lows", 0) > 0
        ]
        ahead = [level for level in leveled if level["role"] == "resistance"]
    else:
        behind = [
            level
            for level in leveled
            if level["role"] == "resistance" and level.get("from_highs", 0) > 0
        ]
        ahead = [level for level in leveled if level["role"] == "support"]

    behind_near = min(behind, key=lambda level: level["distance"]) if behind else None
    ahead_near = min(ahead, key=lambda level: level["distance"]) if ahead else None

    if behind_near is None:
        base, detail = 2, "SL根拠に使える背後の確定スイング水平線がない"
    else:
        behind_ratio = behind_near["distance"] / atr
        if behind_ratio <= 1.0:
            base, detail = 15, "背後の支えが至近（SL根拠が明確）"
        elif behind_ratio <= 2.0:
            base, detail = 11, "背後の支えがやや近い"
        elif behind_ratio <= 4.0:
            base, detail = 6, "背後の支えがやや遠い"
        else:
            base, detail = 3, "背後の支えが遠い"

    penalty = 0
    if ahead_near is not None:
        ahead_ratio = ahead_near["distance"] / atr
        if ahead_ratio <= 0.5:
            penalty = 8
            detail += " / 進行方向すぐ先に障害あり（大幅減点）"
        elif ahead_ratio <= 1.5:
            penalty = 4
            detail += " / 進行方向に障害が近い（減点）"

    return {
        "score": max(0, base - penalty),
        "nearest_distance": (
            round(float(behind_near["distance"]), 3) if behind_near else None
        ),
        "ratio": (
            round(float(behind_near["distance"] / atr), 2) if behind_near else None
        ),
        "ahead_distance": (
            round(float(ahead_near["distance"]), 3) if ahead_near else None
        ),
        "detail": detail,
        "behind": behind_near,
        "ahead": ahead_near,
    }


def judge_session_breakout(
    df_5m: pd.DataFrame,
    direction: str,
    lookback: int = 24,
    atr: Optional[float] = None,
) -> dict:
    """直近2時間レンジ内の位置と、ATRバッファ付きブレイクを評価する。"""
    if len(df_5m) < lookback + 2:
        return {"score": 5, "state": "データ不足", "position_pct": None}

    window = df_5m.iloc[-lookback - 1 : -1]
    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    last_close = float(df_5m["close"].iloc[-1])
    range_width = range_high - range_low
    if range_width <= 0:
        return {"score": 5, "state": "レンジ幅ゼロ", "position_pct": None}

    if atr is None or pd.isna(atr) or atr <= 0:
        atr = calc_atr(df_5m, 14).iloc[-1]
    if pd.isna(atr) or atr <= 0:
        atr = last_close * 0.0005
    buffer = max(float(atr) * 0.05, 0.001)

    raw_position = (last_close - range_low) / range_width * 100.0
    directional_position = raw_position if direction == "long" else 100.0 - raw_position
    confirmed_break = (
        last_close > range_high + buffer
        if direction == "long"
        else last_close < range_low - buffer
    )

    if confirmed_break:
        score, state = 15, "レンジを明確にブレイク"
    elif directional_position >= 80:
        score, state = 13, "レンジ端（ブレイク未確認）"
    elif directional_position >= 60:
        score, state = 10, "有利方向寄り"
    elif directional_position >= 40:
        score, state = 6, "レンジ中央付近"
    elif directional_position >= 20:
        score, state = 3, "不利方向寄り"
    else:
        score, state = 0, "不利方向の端"

    return {
        "score": score,
        "state": state,
        "position_pct": round(float(directional_position), 1),
        "range_low": round(range_low, 3),
        "range_high": round(range_high, 3),
    }


def judge_momentum(
    df_5m: pd.DataFrame,
    direction: str,
    lookback: int = 12,
    atr: Optional[float] = None,
) -> dict:
    if len(df_5m) < lookback + 1:
        return {"score": 3, "move_pips": 0.0, "ratio": None, "state": "データ不足"}

    move = float(df_5m["close"].iloc[-1] - df_5m["close"].iloc[-lookback - 1])
    move_pips = round(move * 100.0, 1)
    if atr is None or pd.isna(atr) or atr <= 0:
        atr = float(df_5m["close"].iloc[-1]) * 0.0005
    directional_move = move if direction == "long" else -move
    ratio = directional_move / (float(atr) * math.sqrt(lookback))

    if ratio >= 1.0:
        score, state = 10, "強い順行の勢い"
    elif ratio >= 0.3:
        score, state = 7, "順行の勢いあり"
    elif ratio >= -0.3:
        score, state = 3, "勢いが乏しい（中立）"
    elif ratio >= -1.0:
        score, state = 1, "逆行の勢い"
    else:
        score, state = 0, "強い逆行"
    return {
        "score": score,
        "move_pips": move_pips,
        "ratio": round(float(ratio), 2),
        "state": state,
    }


# ============================================================
# 5. DXY / 米2年債 / 経済指標 / COT
# ============================================================

def judge_macro_alignment(
    dxy_df: pd.DataFrame,
    us2y_df: pd.DataFrame,
    direction: str,
    lookback: int = 5,
    dxy_flat_threshold: float = 0.15,
    y2_flat_threshold: float = 0.03,
) -> dict:
    """日次 DXY・DGS2 の方向を補助点として評価する。"""

    def recent_slope(series: pd.Series, number: int) -> Optional[float]:
        values = series.dropna().tail(number)
        if len(values) < min(number, 3):
            return None
        return float(values.iloc[-1] - values.iloc[0])

    def state_of(slope: Optional[float], flat_threshold: float) -> str:
        if slope is None:
            return "unavailable"
        if abs(slope) < flat_threshold:
            return "flat"
        return "up" if slope > 0 else "down"

    dxy_slope = recent_slope(dxy_df["close"], lookback) if "close" in dxy_df else None
    y2_slope = recent_slope(us2y_df["yield"], lookback) if "yield" in us2y_df else None
    dxy_state = state_of(dxy_slope, dxy_flat_threshold)
    y2_state = state_of(y2_slope, y2_flat_threshold)
    expected = "up" if direction == "long" else "down"

    def score_state(state: str, full: int) -> int:
        if state == "unavailable":
            return 0
        if state == expected:
            return full
        if state == "flat":
            return full // 2
        return 0

    return {
        "dxy_slope": round(dxy_slope, 4) if dxy_slope is not None else None,
        "dxy_state": dxy_state,
        "dxy_score": score_state(dxy_state, 10),
        "us2y_slope": round(y2_slope, 4) if y2_slope is not None else None,
        "us2y_state": y2_state,
        "us2y_score": score_state(y2_state, 5),
        "data_available": dxy_state != "unavailable" and y2_state != "unavailable",
    }


def fetch_economic_calendar() -> list[dict]:
    """Fair Economy Media の週間カレンダーを取得する。"""
    _require_dependency(requests, "requests")
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    response = requests.get(url, timeout=15, headers=HTTP_HEADERS)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError("経済指標カレンダーの形式が不正です")
    return data


def is_near_high_impact_event(
    events: list[dict],
    currencies: tuple[str, ...] = ("USD", "JPY"),
    window_minutes: int = 30,
    now: Optional[datetime] = None,
) -> dict:
    """USD/JPY の High 指標の前後 window_minutes 内か判定する。"""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)

    nearest = None
    nearest_diff = None
    for event in events:
        if str(event.get("country", "")).upper() not in currencies:
            continue
        if str(event.get("impact", "")).lower() != "high":
            continue
        try:
            raw_date = str(event["date"]).replace("Z", "+00:00")
            event_time = datetime.fromisoformat(raw_date)
        except (KeyError, TypeError, ValueError):
            continue
        if event_time.tzinfo is None:
            continue
        diff = abs((event_time.astimezone(timezone.utc) - current).total_seconds()) / 60.0
        if nearest_diff is None or diff < nearest_diff:
            nearest_diff = diff
            nearest = event

    return {
        "is_near": nearest_diff is not None and nearest_diff <= window_minutes,
        "fetched": True,
        "nearest_event": nearest.get("title") if nearest else None,
        "nearest_diff_minutes": round(float(nearest_diff), 1) if nearest_diff is not None else None,
    }


def fetch_cot_report(weeks: int = 2) -> list[dict]:
    """CFTC TFF（先物のみ）の円先物レコードを新しい順で取得する。"""
    _require_dependency(requests, "requests")
    url = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
    params = {
        "$where": (
            "market_and_exchange_names = "
            "'JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE'"
        ),
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": int(weeks),
    }
    response = requests.get(
        url, params=params, timeout=15, headers=HTTP_HEADERS
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError("CFTC COT の形式が不正です")
    return data


def summarize_cot(records: list[dict]) -> dict:
    """レバレッジドファンドの円先物ネットを厳格に検証して要約する。"""
    if not records:
        return {"available": False, "reason": "レコードなし"}

    def first_present(record: dict, names: tuple[str, ...]) -> Any:
        for name in names:
            if name in record and record[name] not in (None, ""):
                return record[name]
        return None

    def get_net(record: dict) -> Optional[float]:
        long_value = first_present(
            record, ("lev_money_positions_long", "lev_money_positions_long_all")
        )
        short_value = first_present(
            record, ("lev_money_positions_short", "lev_money_positions_short_all")
        )
        if long_value is None or short_value is None:
            return None
        try:
            return float(long_value) - float(short_value)
        except (TypeError, ValueError):
            return None

    latest = records[0]
    net_latest = get_net(latest)
    if net_latest is None:
        return {"available": False, "reason": "必要なCFTC列が欠損または不正"}

    result = {
        "available": True,
        "contract_name": latest.get("market_and_exchange_names", "不明"),
        "report_date": latest.get("report_date_as_yyyy_mm_dd", "不明"),
        "net_position": net_latest,
        "net_change": None,
    }
    if len(records) >= 2:
        previous_net = get_net(records[1])
        if previous_net is not None:
            result["net_change"] = net_latest - previous_net
    return result


# ============================================================
# 6. スコアリング
# ============================================================

def calc_score(
    env_score: int,
    trend_score_result: dict,
    macro_result: dict,
    key_level_result: dict,
    atr_ratio_ok: bool,
    near_indicator_time: bool,
    vwap_result: dict,
    breakout_result: dict,
    momentum_result: dict,
) -> dict:
    scores = {
        "VWAP位置": int(vwap_result["score"]),
        "セッション位置": int(breakout_result["score"]),
        "モメンタム": int(momentum_result["score"]),
        "重要ライン": int(key_level_result["score"]),
        "トレンド(1h)": int(trend_score_result["score"]),
        "環境認識(上位足)": min(10, round(env_score / 2)),
        "DXY": int(macro_result["dxy_score"]),
        "米2年債": int(macro_result["us2y_score"]),
        "ボラティリティ": 5 if atr_ratio_ok else 2,
        "経済指標": 0 if near_indicator_time else 5,
    }
    total = sum(scores.values())
    scores["合計"] = total
    scores["推奨"] = (
        "条件付きエントリー候補" if total >= ENTRY_SCORE_THRESHOLD else "見送り推奨"
    )
    return scores


def select_direction(
    eval_long: dict,
    eval_short: dict,
    min_edge: int = MIN_DIRECTION_SCORE_EDGE,
) -> dict:
    """両方向の差が小さい時は、自動的にロングへ倒さず方向不明とする。"""
    long_score = int(eval_long["score"]["合計"])
    short_score = int(eval_short["score"]["合計"])
    edge = abs(long_score - short_score)
    if long_score == short_score:
        return {
            "best": None,
            "direction": None,
            "edge": 0,
            "ambiguous": True,
            "reason": "ロング・ショートが同点",
        }
    best = eval_long if long_score > short_score else eval_short
    return {
        "best": best,
        "direction": best["direction"],
        "edge": edge,
        "ambiguous": edge < min_edge,
        "reason": (
            f"方向差が{edge}点で基準{min_edge}点未満"
            if edge < min_edge
            else f"方向差{edge}点"
        ),
    }


# ============================================================
# 7. 損切り・利確・ロット
# ============================================================

def suggest_sl_tp(
    entry_price: float,
    direction: str,
    atr: float,
    key_levels: list[dict],
    rr_min: float = 2.0,
    min_buffer_pips: float = 1.0,
) -> dict:
    """確定スイング由来の水平線に基づき SL / TP を算出する。"""
    if pd.isna(atr) or atr <= 0:
        return {"valid": False, "reason": "ATRが不正"}

    leveled = assign_level_roles(key_levels, entry_price)
    if direction == "long":
        sl_candidates = [
            level
            for level in leveled
            if level["role"] == "support" and level.get("from_lows", 0) > 0
        ]
        tp_obstacles = [level for level in leveled if level["role"] == "resistance"]
    else:
        sl_candidates = [
            level
            for level in leveled
            if level["role"] == "resistance" and level.get("from_highs", 0) > 0
        ]
        tp_obstacles = [level for level in leveled if level["role"] == "support"]

    if not sl_candidates:
        return {"valid": False, "reason": "SL根拠となる確定スイング水平線が存在しない"}

    anchor = min(sl_candidates, key=lambda level: level["distance"])
    buffer = max(float(atr) * 0.3, min_buffer_pips / 100.0)
    if direction == "long":
        stop_loss = anchor["price"] - buffer
        risk = entry_price - stop_loss
    else:
        stop_loss = anchor["price"] + buffer
        risk = stop_loss - entry_price
    if risk <= 0:
        return {"valid": False, "reason": "SL位置がエントリー価格と逆転"}

    take_profit = (
        entry_price + risk * rr_min
        if direction == "long"
        else entry_price - risk * rr_min
    )
    obstacle = None
    for level in tp_obstacles:
        in_path = (
            entry_price < level["price"] < take_profit
            if direction == "long"
            else take_profit < level["price"] < entry_price
        )
        if not in_path:
            continue
        if obstacle is None:
            obstacle = level
        elif direction == "long" and level["price"] < obstacle["price"]:
            obstacle = level
        elif direction == "short" and level["price"] > obstacle["price"]:
            obstacle = level

    return {
        "valid": True,
        "entry": round(float(entry_price), 3),
        "stop_loss": round(float(stop_loss), 3),
        "take_profit": round(float(take_profit), 3),
        "risk_pips": round(float(risk) * 100.0, 1),
        "rr_ratio": rr_min,
        "sl_anchor": round(float(anchor["price"]), 3),
        "sl_anchor_timeframe": anchor.get("timeframe"),
        "obstacle_before_tp": (
            round(float(obstacle["price"]), 3) if obstacle else None
        ),
    }


def calc_lot_size(
    account_balance: float,
    risk_percent: float,
    risk_pips: float,
    pip_value_per_lot: float = 1000.0,
    lot_step: float = 0.01,
    execution_buffer_pips: float = 0.0,
) -> dict:
    """1,000通貨刻みで必ず切り捨て、許容損失を超えない数量にする。"""
    risk_amount = account_balance * (risk_percent / 100.0)
    effective_risk_pips = risk_pips + max(0.0, execution_buffer_pips)
    if effective_risk_pips <= 0 or pip_value_per_lot <= 0 or lot_step <= 0:
        return {
            "risk_amount_jpy": round(risk_amount, 0),
            "recommended_lot": 0.0,
            "input_quantity_x1000": 0,
            "reason": "リスク幅またはロット条件が不正",
        }

    raw_lot = risk_amount / (effective_risk_pips * pip_value_per_lot)
    steps = math.floor((raw_lot / lot_step) + 1e-12)
    recommended_lot = max(0.0, steps * lot_step)
    input_quantity = math.floor(recommended_lot * 100.0 + 1e-9)
    estimated_loss = recommended_lot * effective_risk_pips * pip_value_per_lot
    return {
        "risk_amount_jpy": round(risk_amount, 0),
        "effective_risk_pips": round(effective_risk_pips, 1),
        "recommended_lot": round(recommended_lot, 2),
        "input_quantity_x1000": int(input_quantity),
        "estimated_loss_jpy": round(estimated_loss, 0),
    }


# ============================================================
# 8. Discord 通知・重複防止
# ============================================================

def load_notification_state(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def save_notification_state(path: Path, state: dict) -> None:
    """一時ファイルから置換し、途中終了で壊れた JSON を残さない。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def notification_allowed(
    state: dict,
    direction: str,
    cooldown_minutes: int,
    now: Optional[datetime] = None,
) -> dict:
    """同方向の通知は cooldown 内に再送しない。方向転換は即時許可する。"""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)

    if state.get("last_direction") != direction:
        return {"allowed": True, "remaining_minutes": 0.0}
    raw_time = state.get("last_notified_at")
    if not raw_time:
        return {"allowed": True, "remaining_minutes": 0.0}
    try:
        last_time = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        elapsed = (current - last_time.astimezone(timezone.utc)).total_seconds() / 60.0
    except (TypeError, ValueError):
        return {"allowed": True, "remaining_minutes": 0.0}

    remaining = cooldown_minutes - elapsed
    return {
        "allowed": remaining <= 0,
        "remaining_minutes": round(max(0.0, remaining), 1),
    }


def send_discord_log(
    score: dict,
    direction: str,
    sltp: dict,
    lot: dict,
    gate_reasons: list,
    extra_info: dict,
    webhook_url: str,
) -> None:
    """毎回の実行結果をログ用チャンネルへ送る（見送りも含む）。

    後からバックテスト・検証に使えるよう判断根拠を残すのが目的。
    ログ送信の失敗は本処理を止めない（エントリー通知とは別扱い）。
    """
    if not webhook_url:
        return

    now_jst = datetime.now(timezone.utc) + timedelta(hours=9)
    direction_jp = "ロング" if direction == "long" else "ショート"

    if gate_reasons:
        verdict = "⛔ 見送り（ハードゲート）"
    elif score.get("合計", 0) >= ENTRY_SCORE_THRESHOLD:
        verdict = "🚨 エントリー推奨"
    else:
        verdict = "⏸ 見送り（点数不足）"

    lines = [
        f"`{now_jst.strftime('%m/%d %H:%M')}` **{verdict}**  "
        f"合計 **{score.get('合計', '-')}点**  方向: {direction_jp}",
        f"ロング{extra_info.get('long_total', '-')}点 / "
        f"ショート{extra_info.get('short_total', '-')}点",
    ]

    breakdown = " / ".join(
        f"{key}{value}"
        for key, value in score.items()
        if key not in ("合計", "推奨")
    )
    if breakdown:
        lines.append("内訳: " + breakdown)

    if sltp.get("valid"):
        lines.append(
            f"E:{sltp['entry']} / SL:{sltp['stop_loss']} / "
            f"TP:{sltp['take_profit']} / {sltp['risk_pips']}pips / "
            f"数量{lot.get('input_quantity_x1000', '-')}"
        )
    else:
        lines.append(f"SL/TP算出不可: {sltp.get('reason', '-')}")

    if extra_info.get("note"):
        lines.append(f"備考: {extra_info['note']}")

    if gate_reasons:
        lines.append("ゲート: " + " ; ".join(dict.fromkeys(gate_reasons)))

    try:
        response = requests.post(
            webhook_url, json={"content": "\n".join(lines)}, timeout=10
        )
        response.raise_for_status()
    except Exception as error:
        print(f"[ログ通知エラー（本処理には影響なし）] {error}")


def send_discord_notification(
    score: dict,
    direction: str,
    sltp: dict,
    lot: dict,
    webhook_url: str,
    analysis_time: Optional[datetime] = None,
) -> bool:
    """Discord 通知を送信し、成功時だけ True を返す。"""
    if not webhook_url:
        print("[通知スキップ] DISCORD_WEBHOOK_URL が未設定です")
        return False
    _require_dependency(requests, "requests")

    current = analysis_time or datetime.now(timezone.utc)
    current_jst = current.astimezone(timezone(timedelta(hours=9)))
    direction_jp = "買い（ロング）" if direction == "long" else "売り（ショート）"
    lines = [
        f"🚨 **USDJPY エントリー候補**（{direction_jp}）",
        f"判定時刻: {current_jst:%Y-%m-%d %H:%M} JST",
        f"合計スコア: **{score['合計']}点**",
        "",
        "**スコア内訳**",
    ]
    for key in [
        "VWAP位置",
        "セッション位置",
        "モメンタム",
        "重要ライン",
        "トレンド(1h)",
        "環境認識(上位足)",
        "DXY",
        "米2年債",
        "ボラティリティ",
        "経済指標",
    ]:
        lines.append(f"・{key}: {score.get(key, '-')}点")
    lines += [
        "",
        "**エントリープラン**",
        f"エントリー参考値: {sltp['entry']}",
        f"損切り: {sltp['stop_loss']}",
        f"利確: {sltp['take_profit']}",
        f"RR比: 1:{sltp['rr_ratio']}",
        f"推奨ロット: {lot.get('recommended_lot', '-')}",
        f"SBI入力数値（×1,000単位）: {lot.get('input_quantity_x1000', '-')}",
        "",
        "※自動売買ではありません。発注前に価格・スプレッド・指標時刻を再確認してください。",
    ]

    response = requests.post(
        webhook_url,
        json={"content": "\n".join(lines)},
        timeout=15,
        headers=HTTP_HEADERS,
    )
    response.raise_for_status()
    print(f"[Discord通知] 送信成功（HTTP {response.status_code}）")
    return True


# ============================================================
# 9. 方向別評価とメイン処理
# ============================================================

def evaluate_direction(
    direction: str,
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    dxy_1d: pd.DataFrame,
    us2y: pd.DataFrame,
    trend_1h: dict,
    env: dict,
    key_levels: list[dict],
    latest_price: float,
    latest_atr: float,
    atr_ratio_ok: bool,
    near_indicator_time: bool,
) -> dict:
    macro = judge_macro_alignment(dxy_1d, us2y, direction=direction, lookback=5)
    vwap = calc_vwap_alignment(df_5m, direction, atr=latest_atr)
    breakout = judge_session_breakout(df_5m, direction, lookback=24, atr=latest_atr)
    momentum = judge_momentum(df_5m, direction, lookback=12, atr=latest_atr)
    trend = calc_trend_score(trend_1h, df_1h, direction)
    key_level = calc_key_level_score(latest_price, key_levels, latest_atr, direction)
    score = calc_score(
        env_score=env["score"] if env["direction"] in (direction, None) else 0,
        trend_score_result=trend,
        macro_result=macro,
        key_level_result=key_level,
        atr_ratio_ok=atr_ratio_ok,
        near_indicator_time=near_indicator_time,
        vwap_result=vwap,
        breakout_result=breakout,
        momentum_result=momentum,
    )
    return {
        "direction": direction,
        "score": score,
        "macro": macro,
        "vwap": vwap,
        "breakout": breakout,
        "momentum": momentum,
        "trend": trend,
        "key_level": key_level,
    }


def main() -> None:
    print("=== USDJPY デイトレ支援ツール Phase 1（安全修正版） ===\n")

    df_5m = fetch_usdjpy(interval="5m", period="5d")
    df_1h = fetch_usdjpy(interval="1h", period="60d")
    df_1d = fetch_usdjpy(interval="1d", period="1y")
    # マクロデータは補助情報。取得に失敗しても処理は継続し、
    # 空データとして扱う（加点されず、ハードゲートで通知は停止される）。
    try:
        dxy_1d = fetch_dxy(period="1mo", interval="1d")
    except Exception as error:
        print(f"[DXY取得失敗] {error} → データなしとして継続します")
        dxy_1d = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    try:
        us2y = fetch_us2y_yield(days=30)
    except Exception as error:
        print(f"[米2年債取得失敗] {error} → データなしとして継続します")
        us2y = pd.DataFrame(columns=["yield"])

    df_5m, drop5 = drop_unconfirmed_bar(df_5m, 5)
    df_1h, drop1h = drop_unconfirmed_bar(df_1h, 60)
    df_1d, drop1d = drop_unconfirmed_bar(df_1d, 1440)
    dxy_1d, drop_dxy = drop_unconfirmed_bar(dxy_1d, 1440)

    print("【データ品質チェック】")
    print(
        "  未確定足を除外: "
        f"5分足{drop5['dropped']}本 / 1時間足{drop1h['dropped']}本 / "
        f"日足{drop1d['dropped']}本 / DXY日足{drop_dxy['dropped']}本"
    )
    if df_5m.empty or df_1h.empty or df_1d.empty or dxy_1d.empty:
        print("\n[中断] 確定足が不足しているため判定できません")
        return

    freshness = check_data_freshness(df_5m, interval_minutes=5)
    print(
        f"  5分足の鮮度: 最新確定足終了から{freshness['age_minutes']}分 "
        f"（許容{freshness.get('max_allowed_minutes')}分）"
        f" → {'OK' if freshness['fresh'] else '古い'}"
    )

    df_5m["ema20"] = calc_ema(df_5m, 20)
    df_5m["atr"] = calc_atr(df_5m, 14)
    df_5m["vwap"] = calc_vwap_daily(df_5m)

    df_1h_sw = find_swing_points(df_1h, window=3)
    key_levels_1h = extract_key_levels(
        df_1h_sw, lookback=100, top_n=5, timeframe="1h"
    )
    df_5m_sw = find_swing_points(df_5m, window=3)
    key_levels_5m = extract_key_levels(
        df_5m_sw, lookback=300, top_n=5, timeframe="5m"
    )
    key_levels = combine_key_levels(key_levels_1h, key_levels_5m, top_n=8)
    trend_1h = judge_trend_with_breakout(
        df_1h_sw, df_1h, lookback_swings=4, breakout_lookback=10
    )

    print("\n【重要水平線（1h + 5分足）】")
    for level in key_levels:
        if level.get("from_highs", 0) and level.get("from_lows", 0):
            origin = "高安両方由来"
        elif level.get("from_highs", 0):
            origin = "高値由来"
        else:
            origin = "安値由来"
        print(
            f"  {level['price']}円（タッチ{level['touch_count']}回 / "
            f"{origin} / {level.get('timeframe', '-')}）"
        )
    print(f"\n【1h足トレンド】 {trend_1h['trend']}（{trend_1h['reason']}）")

    df_4h = resample_ohlcv_complete(df_1h, "4h", bars_required=4)
    df_4h_sw = find_swing_points(df_4h, window=2)
    trend_4h = judge_trend_with_breakout(
        df_4h_sw, df_4h, lookback_swings=3, breakout_lookback=10
    )
    df_1d_sw = find_swing_points(df_1d, window=2)
    trend_1d = judge_trend_with_breakout(
        df_1d_sw, df_1d, lookback_swings=3, breakout_lookback=10
    )
    env = judge_environment(trend_1h, trend_4h, trend_1d)
    print("\n【環境認識】")
    print(f"  1h: {trend_1h['trend']} / 4h: {trend_4h['trend']} / 日足: {trend_1d['trend']}")
    print(f"  → {env['detail']}（内部スコア{env['score']} / 20）")

    latest_price = float(df_5m["close"].iloc[-1])
    latest_atr = float(df_5m["atr"].iloc[-1])
    atr_ratio_ok = not pd.isna(latest_atr) and 0.03 < latest_atr < 0.20

    try:
        economic_events = fetch_economic_calendar()
        economic_check = is_near_high_impact_event(economic_events, window_minutes=30)
    except Exception as error:
        print(f"\n[経済指標カレンダー取得失敗] {error}")
        economic_check = {
            "is_near": True,
            "fetched": False,
            "nearest_event": None,
            "nearest_diff_minutes": None,
        }
    print("\n【経済指標カレンダー】")
    if economic_check["nearest_event"]:
        print(
            f"  最寄りの重要指標: {economic_check['nearest_event']} "
            f"（時間差{economic_check['nearest_diff_minutes']}分）"
        )
    print(f"  発表前後30分以内: {economic_check['is_near']}")

    try:
        cot_summary = summarize_cot(fetch_cot_report(weeks=2))
    except Exception as error:
        print(f"\n[COTレポート取得失敗] {error}")
        cot_summary = {"available": False, "reason": str(error)}
    print("\n【COT（週次・参考情報）】")
    if cot_summary.get("available"):
        net = cot_summary["net_position"]
        position_jp = "円買い越し" if net > 0 else "円売り越し"
        report_date = str(cot_summary["report_date"]).split("T")[0]
        print(f"  レポート日: {report_date}")
        print(f"  レバレッジドファンド: {net:+.0f}枚（{position_jp}）")
        if cot_summary.get("net_change") is not None:
            print(f"  前週比: {cot_summary['net_change']:+.0f}枚")
    else:
        print(f"  データ利用不可: {cot_summary.get('reason', '不明')}")

    evaluation_args = {
        "df_5m": df_5m,
        "df_1h": df_1h,
        "dxy_1d": dxy_1d,
        "us2y": us2y,
        "trend_1h": trend_1h,
        "env": env,
        "key_levels": key_levels,
        "latest_price": latest_price,
        "latest_atr": latest_atr,
        "atr_ratio_ok": atr_ratio_ok,
        "near_indicator_time": economic_check["is_near"],
    }
    eval_long = evaluate_direction("long", **evaluation_args)
    eval_short = evaluate_direction("short", **evaluation_args)
    selection = select_direction(eval_long, eval_short)

    print("\n【両方向の採点比較】")
    print(f"  ロング: {eval_long['score']['合計']}点")
    print(f"  ショート: {eval_short['score']['合計']}点")
    if selection["best"] is None:
        print(f"  → 採用方向なし（{selection['reason']}）")
        best = None
    else:
        best = selection["best"]
        direction_jp = "ロング（買い）" if best["direction"] == "long" else "ショート（売り）"
        print(f"  → 暫定方向: {direction_jp} / {selection['reason']}")

    gate_reasons: list[str] = []
    if selection["ambiguous"]:
        gate_reasons.append(f"方向の優位差不足（{selection['reason']}）")
    if not freshness["fresh"]:
        gate_reasons.append(f"5分足データが古い（足終了から{freshness['age_minutes']}分）")
    if economic_check["is_near"]:
        gate_reasons.append("重要経済指標の発表前後30分以内")
    if not economic_check.get("fetched", True):
        gate_reasons.append("経済指標カレンダー取得失敗（安全側に停止）")

    if best is None:
        score = max(
            (eval_long["score"], eval_short["score"]),
            key=lambda item: item["合計"],
        )
        sltp = {"valid": False, "reason": "方向を選択できない"}
        lot = {"recommended_lot": 0.0, "input_quantity_x1000": 0}
    else:
        direction = best["direction"]
        score = best["score"]
        macro = best["macro"]
        print(f"\n【採用候補の根拠（{direction}）】")
        print(
            f"  DXY: {macro['dxy_slope']} ({macro['dxy_state']}) / "
            f"米2年債: {macro['us2y_slope']} ({macro['us2y_state']})"
        )
        print(
            f"  VWAP: diff={best['vwap']['diff']} / ATR比={best['vwap']['ratio']} / "
            f"{best['vwap']['score']}点"
        )
        print(
            f"  直近2時間レンジ: {best['breakout']['state']} "
            f"({best['breakout']['position_pct']}%) / {best['breakout']['score']}点"
        )
        print(
            f"  モメンタム: {best['momentum']['move_pips']}pips / "
            f"{best['momentum']['state']} / {best['momentum']['score']}点"
        )
        print(
            f"  重要ライン: {best['key_level']['detail']} / "
            f"{best['key_level']['score']}点"
        )
        print(f"  1hトレンド: {best['trend']['detail']} / {best['trend']['score']}点")

        if not macro.get("data_available", False):
            gate_reasons.append("DXYまたは米2年債データが不足")
        sltp = suggest_sl_tp(latest_price, direction, latest_atr, key_levels)
        if not sltp.get("valid"):
            gate_reasons.append(f"SL/TP算出不可（{sltp.get('reason')}）")
            lot = {"recommended_lot": 0.0, "input_quantity_x1000": 0}
        else:
            if sltp["risk_pips"] > MAX_RISK_PIPS_FOR_DAYTRADE:
                gate_reasons.append(
                    f"SL距離{sltp['risk_pips']}pipsが上限"
                    f"{MAX_RISK_PIPS_FOR_DAYTRADE:g}pipsを超過"
                )
            if sltp.get("obstacle_before_tp") is not None:
                gate_reasons.append(
                    f"TP前に水平線障害あり（{sltp['obstacle_before_tp']}円）"
                )
            lot = calc_lot_size(
                account_balance=ACCOUNT_BALANCE_JPY,
                risk_percent=RISK_PERCENT,
                risk_pips=sltp["risk_pips"],
                execution_buffer_pips=EXECUTION_BUFFER_PIPS,
            )
            if lot["input_quantity_x1000"] < 1:
                gate_reasons.append("計算ロットが最小取引単位未満")

    print("\n【スコア内訳】")
    for key, value in score.items():
        print(f"  {key}: {value}")
    print("\n【SL/TP・数量】")
    if sltp.get("valid"):
        print(
            f"  Entry {sltp['entry']} / SL {sltp['stop_loss']} / "
            f"TP {sltp['take_profit']} / risk {sltp['risk_pips']}pips"
        )
        print(
            f"  {RISK_PERCENT:g}%リスク・執行余裕{EXECUTION_BUFFER_PIPS:g}pips込み: "
            f"{lot['recommended_lot']} lot / SBI入力 {lot['input_quantity_x1000']}"
        )
        print(
            f"  許容損失 {lot['risk_amount_jpy']:.0f}円 / "
            f"概算損失 {lot['estimated_loss_jpy']:.0f}円"
        )
    else:
        print(f"  算出不可: {sltp.get('reason')}")

    log_extra = {
        "long_total": eval_long["score"]["合計"],
        "short_total": eval_short["score"]["合計"],
    }

    if gate_reasons:
        print(f"\n[通知なし・ハードゲート] 暫定スコア{score['合計']}点")
        for reason in dict.fromkeys(gate_reasons):
            print(f"  - {reason}")
        send_discord_log(
            score, best["direction"], sltp, lot, gate_reasons,
            log_extra, DISCORD_LOG_WEBHOOK_URL,
        )
        return
    if score["合計"] < ENTRY_SCORE_THRESHOLD:
        print(
            f"\n[通知なし] {score['合計']}点（基準{ENTRY_SCORE_THRESHOLD}点未満）"
        )
        send_discord_log(
            score, best["direction"], sltp, lot, [],
            log_extra, DISCORD_LOG_WEBHOOK_URL,
        )
        return

    state = load_notification_state(SIGNAL_STATE_FILE)
    cooldown = notification_allowed(
        state,
        direction=best["direction"],
        cooldown_minutes=NOTIFICATION_COOLDOWN_MINUTES,
    )
    if not cooldown["allowed"]:
        print(
            "\n[通知なし・重複防止] 同方向の前回通知から間隔が短いため抑制 "
            f"（あと約{cooldown['remaining_minutes']}分）"
        )
        send_discord_log(
            score, best["direction"], sltp, lot,
            [f"重複防止（あと約{cooldown['remaining_minutes']}分）"],
            log_extra, DISCORD_LOG_WEBHOOK_URL,
        )
        return

    try:
        sent = send_discord_notification(
            score,
            best["direction"],
            sltp,
            lot,
            DISCORD_WEBHOOK_URL,
        )
    except Exception as error:
        print(f"[Discord通知エラー] {error}")
        raise
    send_discord_log(
        score, best["direction"], sltp, lot, [],
        {**log_extra, "note": "エントリー通知を送信しました"},
        DISCORD_LOG_WEBHOOK_URL,
    )
    if sent:
        save_notification_state(
            SIGNAL_STATE_FILE,
            {
                "last_direction": best["direction"],
                "last_notified_at": datetime.now(timezone.utc).isoformat(),
                "last_score": score["合計"],
                "last_entry": sltp["entry"],
            },
        )


if __name__ == "__main__":
    main()
