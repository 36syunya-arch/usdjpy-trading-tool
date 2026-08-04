# -*- coding: utf-8 -*-
"""
USDJPY デイトレ支援ツール - Phase1 プロトタイプ

必要ライブラリ:
    pip install yfinance pandas numpy requests

実行方法:
    python main.py
"""

import math
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta, timezone

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
    """5分足データに対し、日毎にリセットされるVWAPを計算する。
    FXのyfinanceデータは出来高(volume)が0のことが多く、その場合は
    本来のVWAP計算(出来高加重)ができないため、単純平均(等加重)に
    フォールバックする。
    """
    df = df_5m.copy()
    df["date"] = df.index.date
    typical = (df["high"] + df["low"] + df["close"]) / 3

    total_volume = df["volume"].sum()
    if total_volume == 0:
        # 出来高データがない(FXでは一般的) → 単純平均でフォールバック
        df["cum_typ"] = typical.groupby(df["date"]).cumsum()
        df["count"] = df.groupby("date").cumcount() + 1
        return df["cum_typ"] / df["count"]

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
            swing_low.iloc[i] = True

    result = df.copy()
    result["swing_high"] = swing_high
    result["swing_low"] = swing_low
    return result


def extract_key_levels(df_with_swings: pd.DataFrame, lookback: int = 100, top_n: int = 5,
                        timeframe: str = "unknown"):
    """直近lookback本のスイング高値/安値から重要水平線を抽出する。

    重要: スイング高値由来か安値由来かを区別して保持する。
    以前は両者を混ぜていたため、ロングの損切り根拠に過去のスイング高値を
    使ってしまう等の問題があった。
    """
    recent = df_with_swings.tail(lookback)
    atr = calc_atr(recent, 14).iloc[-1]
    if pd.isna(atr) or atr <= 0:
        atr = float(recent["close"].iloc[-1]) * 0.0005
    cluster_width = float(atr) * 0.5  # ATR基準のクラスタ幅(固定0.1円だと連鎖結合しやすい)

    points = []
    for p in recent.loc[recent["swing_high"], "high"].tolist():
        points.append({"price": p, "origin": "swing_high"})
    for p in recent.loc[recent["swing_low"], "low"].tolist():
        points.append({"price": p, "origin": "swing_low"})

    if not points:
        return []

    points.sort(key=lambda x: x["price"])
    clusters = []
    current = [points[0]]
    cluster_start = points[0]["price"]
    for p in points[1:]:
        # クラスタ先頭からの距離で判定(隣接判定だと連鎖結合してしまう)
        if p["price"] - cluster_start <= cluster_width:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
            cluster_start = p["price"]
    clusters.append(current)

    levels = []
    for c in clusters:
        prices = [x["price"] for x in c]
        highs_n = sum(1 for x in c if x["origin"] == "swing_high")
        lows_n = sum(1 for x in c if x["origin"] == "swing_low")
        levels.append({
            "price": round(float(np.mean(prices)), 3),
            "touch_count": len(c),
            "from_highs": highs_n,
            "from_lows": lows_n,
            "timeframe": timeframe,
        })

    levels.sort(key=lambda x: x["touch_count"], reverse=True)
    return levels[:top_n]


def assign_level_roles(levels: list, current_price: float) -> list:
    """現在価格を基準に、各水平線の役割(サポート/レジスタンス)を決定する。
    現在価格より下 = サポート候補、上 = レジスタンス候補。
    """
    out = []
    for lv in levels:
        role = "support" if lv["price"] < current_price else "resistance"
        item = dict(lv)
        item["role"] = role
        item["distance"] = abs(current_price - lv["price"])
        out.append(item)
    return out


# ============================================================
# 4. トレンド / レンジ判定(ダウ理論ベース)
# ============================================================

def drop_unconfirmed_bar(df: pd.DataFrame, interval_minutes: int,
                          grace_minutes: int = 1) -> tuple:
    """形成途中(未確定)の最新足を除外する。

    yfinanceの最新行は形成途中の足であることが多く、そのまま使うと
    「一瞬高値を超えただけ」でブレイク判定され、足が戻った後も
    通知済みになってしまう。バー開始時刻+足の長さが現在時刻を
    過ぎている行だけを確定足として扱う。
    """
    if df.empty:
        return df, {"dropped": 0, "reason": "データが空"}

    now = datetime.now(timezone.utc)
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    bar_end = idx + pd.Timedelta(minutes=interval_minutes)
    cutoff = now - pd.Timedelta(minutes=grace_minutes)
    confirmed_mask = np.asarray(bar_end <= cutoff, dtype=bool)
    dropped = int((~confirmed_mask).sum())
    out = df.loc[confirmed_mask].copy()
    return out, {"dropped": dropped, "remaining": len(out)}


def check_data_freshness(df: pd.DataFrame, interval_minutes: int,
                          max_age_multiplier: float = 3.0) -> dict:
    """最新の確定足が古すぎないかを確認する。
    GitHub Actionsのcronは遅延することがあるため、実行時刻ではなく
    データそのものの鮮度で判断する必要がある。
    """
    if df.empty:
        return {"fresh": False, "age_minutes": None, "reason": "データが空"}

    now = datetime.now(timezone.utc)
    last_idx = df.index[-1]
    if last_idx.tzinfo is None:
        last_idx = last_idx.tz_localize("UTC")
    age = (now - last_idx).total_seconds() / 60
    max_age = interval_minutes * max_age_multiplier
    return {
        "fresh": age <= max_age,
        "age_minutes": round(age, 1),
        "max_allowed_minutes": round(max_age, 1),
    }


def resample_ohlcv_complete(df: pd.DataFrame, rule: str, bars_required: int) -> pd.DataFrame:
    """上位足へ変換し、構成本数が足りない未完成バーを除外する。

    以前はdropna()のみだったため、1時間足1本だけで作られた
    未完成の4時間足がそのまま残っていた。
    """
    agg = df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    })
    counts = df.resample(rule).size()
    agg = agg[counts >= bars_required]
    return agg.dropna()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """1時間足など既存データを、より上位の時間足(例: 4時間足)へ変換する。"""
    out = pd.DataFrame({
        "open": df["open"].resample(rule).first(),
        "high": df["high"].resample(rule).max(),
        "low": df["low"].resample(rule).min(),
        "close": df["close"].resample(rule).last(),
        "volume": df["volume"].resample(rule).sum(),
    })
    return out.dropna()


def judge_environment(trend_1h: dict, trend_4h: dict, trend_1d: dict) -> dict:
    """1h/4h/日足のトレンド方向一致度から環境認識スコアを機械的に算出する。
    以前は「3つ完全一致(20点)/2つ一致(10点)/それ以外0点」の3段階だったが、
    これだと「1つだけトレンド、残り2つはレンジ(矛盾ではない)」のケースまで
    0点になり、極端に厳しすぎたため、矛盾の有無で段階を追加する。
    """
    trends = [trend_1h["trend"], trend_4h["trend"], trend_1d["trend"]]
    up = trends.count("上昇トレンド")
    down = trends.count("下降トレンド")

    if up == 3:
        return {"score": 20, "direction": "long", "detail": "1h/4h/日足すべて上昇一致"}
    if down == 3:
        return {"score": 20, "direction": "short", "detail": "1h/4h/日足すべて下降一致"}
    # 反対方向が1つでもある場合は「矛盾あり」として高得点を与えない
    if up == 2 and down == 0:
        return {"score": 15, "direction": "long", "detail": "2つが上昇一致・逆行なし"}
    if down == 2 and up == 0:
        return {"score": 15, "direction": "short", "detail": "2つが下降一致・逆行なし"}
    if up >= 1 and down >= 1:
        return {"score": 0, "direction": None, "detail": "上昇と下降が混在(明確な矛盾)"}
    if up == 1 and down == 0:
        return {"score": 8, "direction": "long", "detail": "1つのみ上昇、他はレンジ(矛盾なし)"}
    if down == 1 and up == 0:
        return {"score": 8, "direction": "short", "detail": "1つのみ下降、他はレンジ(矛盾なし)"}
    return {"score": 3, "direction": None, "detail": "全時間足レンジ(方向感なし)"}


def detect_breakout(df: pd.DataFrame, lookback: int = 10) -> str:
    """直近lookback本(直近1本を除く)の高値/安値を、最新の終値が
    明確に超えているかを判定する。スイング確定待ちによる遅れを
    補うための即応性の高い補助シグナル。
    """
    if len(df) < lookback + 2:
        return None
    recent_low = df["low"].iloc[-lookback - 1:-1].min()
    recent_high = df["high"].iloc[-lookback - 1:-1].max()
    last_close = df["close"].iloc[-1]
    if last_close < recent_low:
        return "down"
    if last_close > recent_high:
        return "up"
    return None


def judge_trend_with_breakout(df_with_swings: pd.DataFrame, df_raw: pd.DataFrame,
                                lookback_swings: int = 4, breakout_lookback: int = 10) -> dict:
    """スイング確定ベースのトレンド判定に、ブレイクアウト補正を加える。
    スイング判定は前後window本のデータが揃わないと確定しないため、
    急な反転直後は「古いトレンドのまま」と誤判定しやすい。
    直近の明確なブレイクアウトが逆方向を示している場合はそちらを優先する。
    """
    swing_result = judge_trend(df_with_swings, lookback_swings)
    brk = detect_breakout(df_raw, breakout_lookback)
    brk_trend = {"down": "下降トレンド", "up": "上昇トレンド"}.get(brk)

    if brk_trend and brk_trend != swing_result["trend"]:
        return {
            "trend": brk_trend,
            "reason": f"直近{breakout_lookback}本のレンジを明確にブレイク(スイング確定待ちの遅れを補正)",
        }
    return swing_result


def combine_key_levels(long_term_levels: list, short_term_levels: list, top_n: int = 8) -> list:
    """1h足ベース(長期)と5分足ベース(短期・直近セッション)の水平線を統合する。
    長期水平線しかないと現在価格から遠すぎることが多いため、
    直近セッションで形成された新しいレンジの高安値も候補に加える。
    """
    combined = long_term_levels + short_term_levels
    if not combined:
        return []
    combined.sort(key=lambda x: x["touch_count"], reverse=True)
    return combined[:top_n]


def calc_vwap_alignment(df_5m: pd.DataFrame, direction: str, atr: float) -> dict:
    """現在値がVWAPに対してエントリー方向と整合的な位置にあるかを、
    ATR(値動きの大きさ)に対する比率で段階評価する。
    僅かな価格変動で0点/10点が反転する脆さを防ぐため、
    「一致/不一致」の2値ではなく、乖離の大きさに応じて5段階で評価する。
    """
    last_close = df_5m["close"].iloc[-1]
    last_vwap = df_5m["vwap"].iloc[-1]
    if pd.isna(last_vwap) or not atr or pd.isna(atr) or atr <= 0:
        return {"score": 5, "diff": None, "ratio": None, "note": "データ不足のため中立点"}

    diff = last_close - last_vwap
    # directional_diffが正 = エントリー方向にVWAPから明確に離れている(優位)
    directional_diff = diff if direction == "long" else -diff
    ratio = directional_diff / atr

    if ratio >= 0.5:
        score = 10
    elif ratio >= 0.15:
        score = 7
    elif ratio >= -0.15:
        score = 5  # VWAP付近(僅差)は中立点。ここで0/10が反転する脆さを解消
    elif ratio >= -0.5:
        score = 2
    else:
        score = 0

    return {"score": score, "diff": round(float(diff), 3), "ratio": round(float(ratio), 2)}


def _three_state(value: float, epsilon: float) -> str:
    """値を up / down / flat の3状態に分類する。
    epsilon以内は「flat(中立)」とし、ロング・ショートどちらにも
    一致させないことで、横ばい相場での方向偏重を防ぐ。
    """
    if value > epsilon:
        return "up"
    if value < -epsilon:
        return "down"
    return "flat"


def calc_trend_score(trend_result: dict, df_1h: pd.DataFrame, direction: str) -> dict:
    """1h足のトレンドを段階評価する。
    価格とEMA20の関係、EMA20の傾きをいずれも3状態(up/down/flat)で判定し、
    横ばい時にショート側だけ加点される偏重バグを回避する。
    """
    ema20 = calc_ema(df_1h, 20)
    atr_1h = calc_atr(df_1h, 14).iloc[-1]
    if pd.isna(atr_1h) or atr_1h <= 0:
        atr_1h = max(float(df_1h["close"].iloc[-1]) * 0.0005, 1e-6)
    epsilon = float(atr_1h) * 0.05

    last_close = float(df_1h["close"].iloc[-1])
    last_ema = float(ema20.iloc[-1])
    prev_ema = float(ema20.iloc[-6]) if len(ema20) >= 6 else float(ema20.iloc[0])

    expected = "up" if direction == "long" else "down"
    price_state = _three_state(last_close - last_ema, epsilon)
    slope_state = _three_state(last_ema - prev_ema, epsilon)

    price_ok = price_state == expected
    slope_ok = slope_state == expected
    price_against = price_state != expected and price_state != "flat"
    slope_against = slope_state != expected and slope_state != "flat"

    swing_trend = trend_result["trend"]
    swing_matches = (
        (swing_trend == "上昇トレンド" and direction == "long") or
        (swing_trend == "下降トレンド" and direction == "short")
    )
    swing_opposes = (
        (swing_trend == "上昇トレンド" and direction == "short") or
        (swing_trend == "下降トレンド" and direction == "long")
    )

    if swing_opposes or (price_against and slope_against):
        score, detail = 0, f"方向と逆行(price={price_state}, slope={slope_state})"
    elif swing_matches and price_ok and slope_ok:
        score, detail = 15, "スイング・EMA位置・EMA傾きすべて方向一致"
    elif swing_matches:
        score, detail = 12, "スイング判定が方向一致"
    elif price_ok and slope_ok:
        score, detail = 10, "EMA20の位置・傾きが方向一致(スイングはレンジ)"
    elif price_ok or slope_ok:
        score, detail = 6, f"EMAの片方のみ一致(price={price_state}, slope={slope_state})"
    elif price_state == "flat" and slope_state == "flat":
        score, detail = 3, "完全な横ばい(方向感なし・中立)"
    else:
        score, detail = 3, f"方向感が薄い(price={price_state}, slope={slope_state})"

    return {"score": score, "detail": detail, "swing_trend": swing_trend}


def calc_key_level_score(latest_price: float, key_levels: list, atr: float,
                          direction: str) -> dict:
    """水平線の評価を売買方向別に行う。

    重要: 以前は「最も近い水平線までの距離」だけで加点していたため、
    ロングのすぐ上にある抵抗線(=上値を塞ぐ悪材料)まで好材料として
    15点が入っていた。方向を考慮して以下のように評価する:
      - 背後の支え(ロングなら下のサポート)が近い → 加点(SLの根拠になる)
      - 進行方向の障害(ロングなら上のレジスタンス)が近い → 減点
    """
    if not key_levels or not atr or pd.isna(atr) or atr <= 0:
        return {"score": 5, "nearest_distance": None, "ratio": None,
                "detail": "水平線またはATRが取得不可", "behind": None, "ahead": None}

    leveled = assign_level_roles(key_levels, latest_price)
    if direction == "long":
        behind = [lv for lv in leveled if lv["role"] == "support"]
        ahead = [lv for lv in leveled if lv["role"] == "resistance"]
    else:
        behind = [lv for lv in leveled if lv["role"] == "resistance"]
        ahead = [lv for lv in leveled if lv["role"] == "support"]

    behind_near = min(behind, key=lambda x: x["distance"]) if behind else None
    ahead_near = min(ahead, key=lambda x: x["distance"]) if ahead else None

    # 背後の支え(SLの根拠)が近いほど加点
    if behind_near is None:
        base, detail = 2, "背後に支えとなる水平線がない(SL根拠なし)"
    else:
        b_ratio = behind_near["distance"] / atr
        if b_ratio <= 1.0:
            base, detail = 15, "背後の支えが至近(SL根拠として明確)"
        elif b_ratio <= 2.0:
            base, detail = 11, "背後の支えがやや近い"
        elif b_ratio <= 4.0:
            base, detail = 6, "背後の支えがやや遠い"
        else:
            base, detail = 3, "背後の支えが遠い"

    # 進行方向の障害が近すぎる場合は減点(TPまで届かない可能性が高い)
    penalty = 0
    if ahead_near is not None:
        a_ratio = ahead_near["distance"] / atr
        if a_ratio <= 0.5:
            penalty, detail = 8, detail + " / 進行方向すぐ先に障害あり(大幅減点)"
        elif a_ratio <= 1.5:
            penalty, detail = 4, detail + " / 進行方向に障害が近い(減点)"

    score = max(0, base - penalty)
    return {
        "score": score,
        "nearest_distance": round(float(behind_near["distance"]), 3) if behind_near else None,
        "ratio": round(float(behind_near["distance"] / atr), 2) if behind_near else None,
        "ahead_distance": round(float(ahead_near["distance"]), 3) if ahead_near else None,
        "detail": detail,
        "behind": behind_near,
        "ahead": ahead_near,
    }


def judge_session_breakout(df_5m: pd.DataFrame, direction: str, lookback: int = 24,
                            atr: float = None) -> dict:
    """直近lookback本(5分足)のレンジに対する現在価格の位置を段階評価する。

    重要な変更:
      - 終値がレンジ高値と「同値」でも100%となり満点15点だったため、
        ATRバッファ(atr*0.1)を超えた場合のみ「ブレイク確定」とする。
      - ブレイク前のレンジ端は監視レベルに留め、高得点を与えない
        (以前は13点。抵抗線直下での高値追いを高評価してしまっていた)。
    """
    if len(df_5m) < lookback + 2:
        return {"score": 5, "state": "データ不足", "position_pct": None}

    window = df_5m.iloc[-lookback - 1:-1]
    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    last_close = float(df_5m["close"].iloc[-1])
    range_width = range_high - range_low

    if range_width <= 0:
        return {"score": 5, "state": "レンジ幅ゼロ", "position_pct": None}

    if atr is None or pd.isna(atr) or atr <= 0:
        atr = last_close * 0.0005
    breakout_buffer = float(atr) * 0.1

    position_pct = (last_close - range_low) / range_width * 100
    directional_pct = position_pct if direction == "long" else (100 - position_pct)

    if direction == "long":
        breakout_distance = last_close - range_high
    else:
        breakout_distance = range_low - last_close

    if breakout_distance >= breakout_buffer:
        score, state = 15, "ATRバッファ込みでブレイク確定"
    elif breakout_distance > 0:
        score, state = 7, "高安値を超過したがバッファ未達(騙し警戒)"
    elif directional_pct >= 80:
        score, state = 6, "レンジ端・ブレイク前(監視レベル)"
    elif directional_pct >= 60:
        score, state = 8, "有利方向寄り(押し目/戻り待ちに適す)"
    elif directional_pct >= 40:
        score, state = 6, "レンジ中央付近"
    elif directional_pct >= 20:
        score, state = 3, "不利方向寄り"
    else:
        score, state = 0, "不利方向の端"

    return {
        "score": score,
        "state": state,
        "position_pct": round(float(directional_pct), 1),
        "range_low": round(range_low, 3),
        "range_high": round(range_high, 3),
        "breakout_distance_pips": round(float(breakout_distance) * 100, 1),
    }


def judge_momentum(df_5m: pd.DataFrame, direction: str, lookback: int = 12,
                    atr: float = None) -> dict:
    """直近lookback本の値幅から勢いを段階評価する。
    以前は符号のみで判定していたため、0.1pipの動きでも満点10点だった。
    ATRに対する比率で段階評価し、微小な動きは中立点にする。
    """
    if len(df_5m) < lookback + 1:
        return {"score": 3, "move_pips": 0.0, "ratio": None, "state": "データ不足"}

    move = float(df_5m["close"].iloc[-1] - df_5m["close"].iloc[-lookback - 1])
    move_pips = round(move * 100, 1)

    if atr is None or pd.isna(atr) or atr <= 0:
        atr = float(df_5m["close"].iloc[-1]) * 0.0005
    directional_move = move if direction == "long" else -move
    ratio = directional_move / (atr * lookback ** 0.5)

    if ratio >= 1.0:
        score, state = 10, "強い順行の勢い"
    elif ratio >= 0.3:
        score, state = 7, "順行の勢いあり"
    elif ratio >= -0.3:
        score, state = 3, "勢いが乏しい(中立)"
    elif ratio >= -1.0:
        score, state = 1, "逆行の勢い"
    else:
        score, state = 0, "強い逆行"

    return {"score": score, "move_pips": move_pips,
            "ratio": round(float(ratio), 2), "state": state}


def judge_trend(df_with_swings: pd.DataFrame, lookback_swings: int = 4,
                 epsilon: float = None) -> dict:
    """直近のスイング高値・安値の切り上げ/切り下げでトレンドを機械的に判定する。

    注意: pandasのis_monotonic_increasingは同値を含むため、
    [150.0, 150.0]のような同値列が上昇・下降の両方でTrueになり、
    横ばいが「上昇トレンド」と誤判定されていた。
    そのため、epsilonを超える厳密な変化があるかで判定する。
    """
    swing_highs = df_with_swings.loc[df_with_swings["swing_high"], "high"].tail(lookback_swings)
    swing_lows = df_with_swings.loc[df_with_swings["swing_low"], "low"].tail(lookback_swings)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"trend": "不明", "reason": "スイング点不足"}

    if epsilon is None:
        atr = calc_atr(df_with_swings, 14).iloc[-1]
        if pd.isna(atr) or atr <= 0:
            atr = float(df_with_swings["close"].iloc[-1]) * 0.0005
        epsilon = float(atr) * 0.05

    high_diff = swing_highs.diff().dropna()
    low_diff = swing_lows.diff().dropna()

    hh = bool((high_diff > epsilon).all())
    hl = bool((low_diff > epsilon).all())
    lh = bool((high_diff < -epsilon).all())
    ll = bool((low_diff < -epsilon).all())

    if hh and hl:
        return {"trend": "上昇トレンド", "reason": "高値・安値ともに明確に切り上げ"}
    if lh and ll:
        return {"trend": "下降トレンド", "reason": "高値・安値ともに明確に切り下げ"}
    return {"trend": "レンジ", "reason": "高値・安値の方向不一致または変化が微小"}


# ============================================================
# 5. DXY / 米2年債との連動判定
# ============================================================

def judge_macro_alignment(dxy_df: pd.DataFrame, us2y_df: pd.DataFrame,
                            direction: str, lookback: int = 5,
                            dxy_flat_threshold: float = 0.15,
                            y2_flat_threshold: float = 0.03) -> dict:
    """DXY・米2年債利回りの方向が、エントリー方向と一致しているかを判定する。
    USDJPYはDXYと正相関、米2年債利回りと正相関が理論値。

    重要: 「横ばい(スロープがほぼゼロ)」を一致扱いしないよう、
    上昇・下降・中立の3状態で判定する。以前は符号比較のみだったため、
    スロープ0の時にshort側が自動的に「一致」になるバグがあった。

    注意: DXY・米2年債はいずれも日次データであり、5分足の判断材料としては
    時間軸が粗い。あくまで大まかな地合いの参考値として扱うこと。
    """
    def recent_slope(series: pd.Series, n: int):
        """データ不足の場合はNoneを返す(0.0を返すとflat扱いされ加点されてしまう)"""
        s = series.dropna().tail(n)
        if len(s) < 2:
            return None
        return float(s.iloc[-1] - s.iloc[0])

    def judge(slope, flat_threshold: float) -> str:
        if slope is None:
            return "unavailable"
        if abs(slope) < flat_threshold:
            return "flat"
        return "up" if slope > 0 else "down"

    dxy_slope = recent_slope(dxy_df["close"], lookback) if "close" in dxy_df else None
    y2_slope = recent_slope(us2y_df["yield"], lookback) if "yield" in us2y_df else None

    dxy_state = judge(dxy_slope, dxy_flat_threshold)
    y2_state = judge(y2_slope, y2_flat_threshold)

    expected = "up" if direction == "long" else "down"

    def to_score(state: str, full: int) -> int:
        if state == "unavailable":
            return 0  # データ欠損は加点しない(以前はflat扱いで加点されていた)
        if state == expected:
            return full
        if state == "flat":
            return full // 2
        return 0

    return {
        "dxy_slope": round(dxy_slope, 4) if dxy_slope is not None else None,
        "dxy_state": dxy_state,
        "dxy_score": to_score(dxy_state, 10),
        "us2y_slope": round(y2_slope, 4) if y2_slope is not None else None,
        "us2y_state": y2_state,
        "us2y_score": to_score(y2_state, 5),
        "data_available": dxy_state != "unavailable" and y2_state != "unavailable",
    }


def fetch_economic_calendar() -> list:
    """Forex Factoryが提供する無料の週間経済指標カレンダー(JSON)を取得する。
    APIキー不要。リクエスト頻度は5分に2回程度までに抑えることが推奨されている。
    """
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def is_near_high_impact_event(events: list, currencies=("USD", "JPY"),
                                window_minutes: int = 30) -> dict:
    """USD/JPYに関わる重要度Highの指標発表の前後window_minutes分以内かを判定する。

    重要: HTTP取得に成功しても中身が空・壊れている場合があるため、
    データの妥当性を検証し、不正な場合はfetched=Falseとして
    ハードゲートを作動させる(フェイルオープンの防止)。
    """
    fail = {
        "is_near": True, "fetched": False,
        "nearest_event": None, "nearest_diff_minutes": None,
    }

    if not isinstance(events, list) or not events:
        fail["reason"] = "経済指標データが空または形式不正"
        return fail

    now = datetime.now(timezone.utc)
    nearest = None
    nearest_diff = None
    parsed_count = 0

    for e in events:
        if not isinstance(e, dict):
            continue
        if not all(k in e for k in ("date", "country", "impact", "title")):
            continue
        try:
            event_time = datetime.fromisoformat(e["date"])
        except (TypeError, ValueError):
            continue
        if event_time.tzinfo is None:
            continue
        parsed_count += 1

        if e.get("country") not in currencies:
            continue
        if e.get("impact") != "High":
            continue

        diff_minutes = abs((event_time - now).total_seconds()) / 60
        if nearest_diff is None or diff_minutes < nearest_diff:
            nearest_diff = diff_minutes
            nearest = e

    if parsed_count == 0:
        fail["reason"] = "有効な日付を持つイベントが1件も解析できなかった"
        return fail

    is_near = nearest_diff is not None and nearest_diff <= window_minutes
    return {
        "is_near": is_near,
        "fetched": True,
        "parsed_count": parsed_count,
        "nearest_event": nearest.get("title") if nearest else None,
        "nearest_diff_minutes": round(nearest_diff, 1) if nearest_diff is not None else None,
    }


def fetch_cot_report(weeks: int = 2) -> list:
    """CFTC(米商品先物取引委員会)のTFFレポート(Traders in Financial Futures)から
    円先物(CME上場の標準契約)のポジションデータを取得する。APIキー不要、無料。
    直近weeks週分を新しい順に取得する。
    完全一致で契約名を指定し、類似の別契約(ミニ先物等)との混同を防ぐ。
    """
    url = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
    params = {
        "$where": "market_and_exchange_names = 'JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": weeks,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def summarize_cot(records: list) -> dict:
    """レバレッジドファンド(投機筋)の円先物ネットポジションと、
    前週からの変化を要約する。フィールド名はCFTC側の仕様に依存するため、
    取得できない場合は安全側に倒してNoneを返す。
    """
    if not records or len(records) < 1:
        return {"available": False}

    def get_net(rec: dict):
        try:
            long_ = float(rec.get("lev_money_positions_long", 0) or 0)
            short_ = float(rec.get("lev_money_positions_short", 0) or 0)
            return long_ - short_
        except (TypeError, ValueError):
            return None

    latest = records[0]
    net_latest = get_net(latest)
    if net_latest is None:
        return {"available": False}

    result = {
        "available": True,
        "contract_name": latest.get("market_and_exchange_names", "不明"),
        "report_date": latest.get("report_date_as_yyyy_mm_dd", "不明"),
        "net_position": net_latest,
        "net_change": None,
    }

    if len(records) >= 2:
        net_prev = get_net(records[1])
        if net_prev is not None:
            result["net_change"] = net_latest - net_prev

    return result


# ============================================================
# 6. スコアリングエンジン(100点満点)
# ============================================================

def calc_score(env_score: int, trend_score_result: dict, macro_result: dict,
               key_level_result: dict, atr_ratio_ok: bool, near_indicator_time: bool,
               vwap_result: dict, breakout_result: dict, momentum_result: dict) -> dict:
    """
    配点(短期軸=5分足の根拠を主軸に、上位足は補助情報として軽く扱う設計):
      VWAP位置                10点(ATR基準の段階評価)
      セッションレンジ内位置    15点(段階評価)
      モメンタム(直近の勢い)   10点
      重要ライン近接           15点(ATR基準の段階評価)
      トレンド判定(1h)        15点(スイング+EMA20の段階評価)
      環境認識(上位足整合)     10点
      DXY方向一致              10点
      米2年債方向一致           5点
      ボラティリティ適正        5点
      経済指標                  5点(発表前後は0点)
    """
    scores = {}

    # --- 短期軸(5分足)の根拠を主軸に(すべて段階評価で微小変動による反転を防止) ---
    scores["VWAP位置"] = vwap_result["score"]
    scores["セッション位置"] = breakout_result["score"]

    scores["モメンタム"] = momentum_result["score"]

    scores["重要ライン"] = key_level_result["score"]
    scores["トレンド(1h)"] = trend_score_result["score"]

    # --- 上位足・マクロは補助情報として軽めに ---
    scores["環境認識(上位足)"] = min(10, round(env_score / 2))
    scores["DXY"] = macro_result["dxy_score"]
    scores["米2年債"] = macro_result["us2y_score"]
    scores["ボラティリティ"] = 5 if atr_ratio_ok else 2
    scores["経済指標"] = 0 if near_indicator_time else 5

    total = sum(scores.values())
    scores["合計"] = total
    scores["推奨"] = "エントリー推奨" if total >= 80 else "見送り推奨"
    return scores


# ============================================================
# 7. 損切り・利確・ロット計算
# ============================================================

def suggest_sl_tp(entry_price: float, direction: str, atr: float,
                   key_levels: list, rr_min: float = 2.0) -> dict:
    """方向別の水平線に基づいてSL/TPを算出する。

    重要な変更:
      - 以前は水平線がない場合に「現在価格±0.5円」という架空のラインを
        使っていたため、構造的な根拠がないのに51.5pipsのSLが作られ、
        60pips上限のゲートを通過してしまっていた。これを廃止する。
      - ロングのSLはスイング安値由来のサポートのみ、
        ショートのSLはスイング高値由来のレジスタンスのみを使う。
      - TPは進行方向の最初の障害を超えられない場合、実現性が低いため
        警告フラグを立てる。
    """
    leveled = assign_level_roles(key_levels, entry_price)

    if direction == "long":
        # ロングのSL根拠は「スイング安値由来のサポート」のみ
        sl_candidates = [lv for lv in leveled
                         if lv["role"] == "support" and lv["from_lows"] > 0]
        tp_obstacles = [lv for lv in leveled if lv["role"] == "resistance"]
    else:
        sl_candidates = [lv for lv in leveled
                         if lv["role"] == "resistance" and lv["from_highs"] > 0]
        tp_obstacles = [lv for lv in leveled if lv["role"] == "support"]

    if not sl_candidates:
        return {"valid": False, "reason": "SLの根拠となる確定スイング水平線が存在しない"}

    anchor = min(sl_candidates, key=lambda x: x["distance"])
    buffer = atr * 0.3

    if direction == "long":
        sl = anchor["price"] - buffer
        risk = entry_price - sl
    else:
        sl = anchor["price"] + buffer
        risk = sl - entry_price

    if risk <= 0:
        return {"valid": False, "reason": "SL位置がエントリー価格と逆転している"}

    tp = entry_price + risk * rr_min if direction == "long" else entry_price - risk * rr_min

    # TPまでの間に障害(反対側の水平線)がないか確認
    obstacle_in_path = None
    for lv in tp_obstacles:
        if direction == "long" and entry_price < lv["price"] < tp:
            if obstacle_in_path is None or lv["price"] < obstacle_in_path["price"]:
                obstacle_in_path = lv
        elif direction == "short" and tp < lv["price"] < entry_price:
            if obstacle_in_path is None or lv["price"] > obstacle_in_path["price"]:
                obstacle_in_path = lv

    return {
        "valid": True,
        "entry": round(entry_price, 3),
        "stop_loss": round(sl, 3),
        "take_profit": round(tp, 3),
        "risk_pips": round(risk * 100, 1),
        "rr_ratio": rr_min,
        "sl_anchor": round(anchor["price"], 3),
        "obstacle_before_tp": round(obstacle_in_path["price"], 3) if obstacle_in_path else None,
    }


def calc_lot_size(account_balance: float, risk_percent: float, risk_pips: float,
                   spread_pips: float = 0.3, slippage_pips: float = 0.2) -> dict:
    """許容リスク額から取引数量(1,000通貨単位)を算出する。

    重要な変更:
      - 以前は round(lot, 2) で四捨五入してから100倍していたため、
        数量が切り上がり許容リスクを超えることがあった。
        (例: 正確値0.1591 lot → 0.16 lot = 16,000通貨、想定損失9,600円 > 許容9,547円)
      - 1,000通貨単位で直接計算し、必ず切り捨てる。
      - スプレッドとスリッページを実効リスクpipsに加算する。
        USDJPYは1,000通貨あたり1pip = 約10円。
    """
    risk_amount = account_balance * risk_percent / 100
    effective_risk_pips = risk_pips + spread_pips + slippage_pips

    if not math.isfinite(effective_risk_pips) or effective_risk_pips <= 0:
        return {"recommended_lot": 0, "input_quantity_x1000": 0,
                "reason": "リスク幅が不正です"}

    quantity_x1000 = math.floor(risk_amount / (effective_risk_pips * 10))

    return {
        "risk_amount_jpy": round(risk_amount, 0),
        "effective_risk_pips": round(effective_risk_pips, 1),
        "recommended_lot": quantity_x1000 / 100,
        "input_quantity_x1000": quantity_x1000,
        "estimated_loss_jpy": round(quantity_x1000 * effective_risk_pips * 10, 0),
    }


# ============================================================
# 8. メイン実行(サンプル)
# ============================================================

import os

# Discord WebhookのURL。
# Colabで試す場合はここに直接貼ってOK。
# GitHub Actions移行後はSecrets(環境変数)から読み込む設計にしてあります。
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "ここにWebhook URLを貼る")
# ログ用チャンネル(見送りを含む全実行結果を記録。通知はミュート推奨)
DISCORD_LOG_WEBHOOK_URL = os.environ.get("DISCORD_LOG_WEBHOOK_URL", "")


def send_discord_log(score: dict, direction: str, sltp: dict, lot: dict,
                      gate_reasons: list, extra_info: dict, webhook_url: str):
    """毎回の実行結果をログ用チャンネルへ送る(見送りも含む)。
    後からバックテストや検証に使えるよう、判断根拠を残すのが目的。
    ログ送信の失敗は本処理を止めない(通知本体とは別扱い)。
    """
    if not webhook_url:
        return

    now_jst = datetime.now(timezone.utc) + timedelta(hours=9)
    direction_jp = "ロング" if direction == "long" else "ショート"

    if gate_reasons:
        verdict = "⛔ 見送り(ハードゲート)"
    elif score["合計"] >= 80:
        verdict = "🚨 エントリー推奨"
    else:
        verdict = "⏸ 見送り(点数不足)"

    lines = [
        f"`{now_jst.strftime('%m/%d %H:%M')}` **{verdict}**  合計 **{score['合計']}点**  方向: {direction_jp}",
        f"ロング{extra_info.get('long_total','-')}点 / ショート{extra_info.get('short_total','-')}点",
        "内訳: " + " / ".join(
            f"{k}{score[k]}" for k in
            ["VWAP位置", "セッション位置", "モメンタム", "重要ライン", "トレンド(1h)",
             "環境認識(上位足)", "DXY", "米2年債", "ボラティリティ", "経済指標"]
            if k in score
        ),
    ]

    if sltp.get("valid"):
        lines.append(
            f"E:{sltp['entry']} / SL:{sltp['stop_loss']} / TP:{sltp['take_profit']} "
            f"/ {sltp['risk_pips']}pips / 数量{lot.get('input_quantity_x1000','-')}"
        )
    else:
        lines.append(f"SL/TP算出不可: {sltp.get('reason','-')}")

    if gate_reasons:
        lines.append("ゲート: " + " ; ".join(gate_reasons))

    try:
        resp = requests.post(webhook_url, json={"content": "\n".join(lines)}, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ログ通知エラー(本処理には影響なし)] {e}")


def send_discord_notification(score: dict, direction: str, sltp: dict, lot: dict, webhook_url: str):
    """スコアが基準を満たした時にDiscordへ通知を送る。"""
    if not webhook_url or "ここに" in webhook_url:
        print("[通知スキップ] Webhook URLが未設定です")
        return

    direction_jp = "買い(ロング)" if direction == "long" else "売り(ショート)"

    lines = [
        f"🚨 **USDJPY エントリーシグナル** ({direction_jp})",
        f"合計スコア: **{score['合計']}点** → {score['推奨']}",
        "",
        "**スコア内訳**",
    ]
    for k in ["VWAP位置", "セッション位置", "モメンタム", "重要ライン", "トレンド(1h)",
              "環境認識(上位足)", "DXY", "米2年債", "ボラティリティ", "経済指標"]:
        if k in score:
            lines.append(f"・{k}: {score[k]}点")

    lines += [
        "",
        "**エントリープラン**",
        f"エントリー: {sltp['entry']}",
        f"損切り: {sltp['stop_loss']}",
        f"利確: {sltp['take_profit']}",
        f"RR比: 1:{sltp['rr_ratio']}",
        f"推奨ロット: {lot.get('recommended_lot', '-')}",
        f"SBI入力数値(×1,000単位): {lot.get('input_quantity_x1000', '-')}",
    ]

    message = {"content": "\n".join(lines)}
    try:
        resp = requests.post(webhook_url, json=message, timeout=10)
        resp.raise_for_status()
        print(f"[Discord通知] 送信成功 (ステータス: {resp.status_code})")
    except Exception as e:
        # 通知失敗を正常終了扱いにしない(GitHub Actions側で失敗として検知させる)
        print(f"[Discord通知エラー] {e}")
        raise


def main():
    print("=== USDJPY デイトレ支援ツール Phase1 ===\n")

    # --- データ取得 ---
    df_5m = fetch_usdjpy(interval="5m", period="5d")
    df_1h = fetch_usdjpy(interval="1h", period="1mo")
    df_1d = fetch_usdjpy(interval="1d", period="1y")
    dxy_1d = fetch_dxy(period="1mo")
    us2y = fetch_us2y_yield(days=30)

    # --- 未確定足(形成途中のローソク足)を除外 ---
    df_5m, drop5 = drop_unconfirmed_bar(df_5m, interval_minutes=5)
    df_1h, drop1h = drop_unconfirmed_bar(df_1h, interval_minutes=60)
    df_1d, drop1d = drop_unconfirmed_bar(df_1d, interval_minutes=1440)

    print("【データ品質チェック】")
    print(f"  未確定足を除外: 5分足{drop5['dropped']}本 / 1時間足{drop1h['dropped']}本 / 日足{drop1d['dropped']}本")

    fresh5 = check_data_freshness(df_5m, interval_minutes=5)
    print(f"  5分足の鮮度: 最新確定足は{fresh5['age_minutes']}分前 (許容{fresh5.get('max_allowed_minutes')}分) → {'OK' if fresh5['fresh'] else '古い'}")

    if df_5m.empty or df_1h.empty or df_1d.empty:
        print("\n[中断] 確定足が不足しているため判定できません")
        return

    # --- 指標計算 ---
    df_5m["ema20"] = calc_ema(df_5m, 20)
    df_5m["atr"] = calc_atr(df_5m, 14)
    df_5m["vwap"] = calc_vwap_daily(df_5m)

    # --- スイング・水平線 ---
    df_1h_sw = find_swing_points(df_1h, window=3)
    key_levels_long = extract_key_levels(df_1h_sw, lookback=100, top_n=5, timeframe="1h")

    # 直近セッション(5分足、直近300本≒25時間)の水平線を追加抽出。
    # 長期水平線だけだと現在価格から遠すぎることが多いため、
    # 直近で形成された新しいレンジの高安値も候補に加える。
    df_5m_sw = find_swing_points(df_5m, window=3)
    key_levels_short = extract_key_levels(df_5m_sw, lookback=300, top_n=5, timeframe="5m")

    key_levels = combine_key_levels(key_levels_long, key_levels_short, top_n=8)
    trend_1h = judge_trend_with_breakout(df_1h_sw, df_1h, lookback_swings=4, breakout_lookback=10)

    print("\n【重要水平線(1h長期 + 5分足直近セッションの統合)】")
    for lv in key_levels:
        origin = "高値由来" if lv.get("from_highs", 0) > lv.get("from_lows", 0) else "安値由来"
        print(f"  {lv['price']} 円  (タッチ{lv['touch_count']}回 / {origin} / {lv.get('timeframe','-')})")
    print(f"\n【1h足トレンド判定】 {trend_1h['trend']} ({trend_1h['reason']})")

    # --- 環境認識自動判定(1h/4h/日足) ---
    # 4時間足は構成する1時間足が4本揃っているものだけを使用(未完成バーを除外)
    df_4h = resample_ohlcv_complete(df_1h, "4h", bars_required=4)
    df_4h_sw = find_swing_points(df_4h, window=2)
    trend_4h = judge_trend_with_breakout(df_4h_sw, df_4h, lookback_swings=3, breakout_lookback=10)

    df_1d_sw = find_swing_points(df_1d, window=2)
    trend_1d = judge_trend_with_breakout(df_1d_sw, df_1d, lookback_swings=3, breakout_lookback=10)

    env = judge_environment(trend_1h, trend_4h, trend_1d)
    print(f"\n【環境認識(マルチタイムフレーム)】")
    print(f"  1h : {trend_1h['trend']}")
    print(f"  4h : {trend_4h['trend']}")
    print(f"  日足: {trend_1d['trend']}")
    print(f"  → {env['detail']}  (環境認識スコア: {env['score']}点)")

    # --- 基礎データの準備 ---
    latest_price = df_5m["close"].iloc[-1]
    latest_atr = df_5m["atr"].iloc[-1]
    atr_ratio_ok = 0.03 < latest_atr < 0.20  # 5分足ATRの適正レンジ(仮)

    # --- 経済指標カレンダー(発表前後30分は減点) ---
    try:
        econ_events = fetch_economic_calendar()
        econ_check = is_near_high_impact_event(econ_events, window_minutes=30)
    except Exception as e:
        print(f"\n[経済指標カレンダー取得失敗] {e} → 安全側に倒し、発表直前とみなします")
        econ_check = {"is_near": True, "fetched": False, "nearest_event": None, "nearest_diff_minutes": None}

    print(f"\n【経済指標カレンダー】")
    if econ_check["nearest_event"]:
        print(f"  直近の重要指標: {econ_check['nearest_event']} (前後{econ_check['nearest_diff_minutes']}分)")
    print(f"  → 発表前後30分以内: {econ_check['is_near']}")

    # --- COTレポート(投機筋の円先物ポジション、参考情報) ---
    try:
        cot_records = fetch_cot_report(weeks=2)
        cot_summary = summarize_cot(cot_records)
    except Exception as e:
        print(f"\n[COTレポート取得失敗] {e}")
        cot_summary = {"available": False}

    print(f"\n【COTレポート(投機筋の円先物ポジション・参考情報)】")
    if cot_summary.get("available"):
        net = cot_summary["net_position"]
        change = cot_summary["net_change"]
        position_jp = "円買い越し" if net > 0 else "円売り越し"
        print(f"  契約: {cot_summary['contract_name']}")
        print(f"  レポート日: {cot_summary['report_date']}")
        print(f"  レバレッジドファンドのネットポジション: {net:+.0f}枚 ({position_jp})")
        if change is not None:
            print(f"  前週比: {change:+.0f}枚")
    else:
        print("  データ取得不可(スコアには影響しません)")

    # --- ロング・ショート両方向を採点し、優位な方を採用する ---
    # 以前は「上昇トレンド以外は全てショート」という強制割り当てだったため、
    # 方向感がない相場でも常にショート前提で採点される重大なバグがあった。
    def evaluate_direction(dir_: str) -> dict:
        macro_d = judge_macro_alignment(dxy_1d, us2y, direction=dir_, lookback=5)
        vwap_d = calc_vwap_alignment(df_5m, dir_, atr=latest_atr)
        breakout_d = judge_session_breakout(df_5m, dir_, lookback=24, atr=latest_atr)
        momentum_d = judge_momentum(df_5m, dir_, lookback=12, atr=latest_atr)
        trend_d = calc_trend_score(trend_1h, df_1h, dir_)
        key_d = calc_key_level_score(latest_price, key_levels, latest_atr, dir_)
        score_d = calc_score(
            env_score=env["score"] if env["direction"] in (dir_, None) else 0,
            trend_score_result=trend_d,
            macro_result=macro_d,
            key_level_result=key_d,
            atr_ratio_ok=atr_ratio_ok,
            near_indicator_time=econ_check["is_near"],
            vwap_result=vwap_d,
            breakout_result=breakout_d,
            momentum_result=momentum_d,
        )
        return {
            "direction": dir_, "score": score_d, "macro": macro_d, "vwap": vwap_d,
            "breakout": breakout_d, "momentum": momentum_d, "trend": trend_d,
            "key_level": key_d,
        }

    eval_long = evaluate_direction("long")
    eval_short = evaluate_direction("short")

    print(f"\n【両方向の採点比較】")
    print(f"  ロング : {eval_long['score']['合計']}点")
    print(f"  ショート: {eval_short['score']['合計']}点")

    long_total = eval_long["score"]["合計"]
    short_total = eval_short["score"]["合計"]
    score_gap = abs(long_total - short_total)
    best = eval_long if long_total > short_total else eval_short
    direction_ambiguous = (long_total == short_total) or (score_gap < 5)
    direction = best["direction"]
    macro = best["macro"]
    vwap_result = best["vwap"]
    breakout_result = best["breakout"]
    momentum_result = best["momentum"]
    trend_score_result = best["trend"]
    key_level_result = best["key_level"]
    score = best["score"]

    direction_jp = "ロング(買い)" if direction == "long" else "ショート(売り)"
    print(f"  → 採用方向: {direction_jp}")

    print(f"\n【マクロ連動判定 ({direction})】")
    print(f"  DXYスロープ: {macro['dxy_slope']} ({macro['dxy_state']})  スコア: {macro['dxy_score']}")
    print(f"  米2年債スロープ: {macro['us2y_slope']} ({macro['us2y_state']})  スコア: {macro['us2y_score']}")

    print(f"\n【短期軸(5分足)の根拠】")
    print(f"  VWAP位置: 現在値-VWAP = {vwap_result['diff']}  ATR比: {vwap_result['ratio']}  スコア: {vwap_result['score']}")
    print(f"  重要ライン最近接距離: {key_level_result['nearest_distance']}円  ATR比: {key_level_result['ratio']}  スコア: {key_level_result['score']}")
    print(f"  セッション内位置: {breakout_result['state']} ({breakout_result['position_pct']}%)  スコア: {breakout_result['score']}")
    print(f"  モメンタム(直近1時間): {momentum_result['move_pips']}pips  {momentum_result['state']}  スコア: {momentum_result['score']}")
    print(f"  トレンド(1h): {trend_score_result['detail']}  スコア: {trend_score_result['score']}")

    print("\n【スコア内訳】")
    for k, v in score.items():
        print(f"  {k}: {v}")

    # --- SL/TP/ロット計算 ---
    sltp = suggest_sl_tp(
        entry_price=latest_price,
        direction=direction,
        atr=latest_atr,
        key_levels=key_levels,
    )

    print(f"\n【SL/TP提案 ({direction})】")
    if not sltp.get("valid"):
        print(f"  算出不可: {sltp.get('reason')}")
    else:
        for k, v in sltp.items():
            if k != "valid":
                print(f"  {k}: {v}")

    # --- 各種ハードゲートの判定 ---
    MAX_RISK_PIPS_FOR_DAYTRADE = 60
    gate_reasons = []

    if not sltp.get("valid"):
        gate_reasons.append(f"SL/TP算出不可({sltp.get('reason')})")
    else:
        if sltp["risk_pips"] > MAX_RISK_PIPS_FOR_DAYTRADE:
            gate_reasons.append(f"SL距離{sltp['risk_pips']}pipsが上限{MAX_RISK_PIPS_FOR_DAYTRADE}pipsを超過")
        if sltp.get("obstacle_before_tp") is not None:
            gate_reasons.append(f"TPに到達する前に障害となる水平線あり({sltp['obstacle_before_tp']}円)")

    if econ_check["is_near"]:
        gate_reasons.append("重要経済指標の発表前後30分以内")
    if not econ_check.get("fetched", True):
        gate_reasons.append("経済指標カレンダーの取得に失敗(安全側に停止)")
    if not fresh5["fresh"]:
        gate_reasons.append(f"5分足データが古い({fresh5['age_minutes']}分前)")
    if not macro.get("data_available", True):
        gate_reasons.append("DXYまたは米2年債のデータが取得できていない")
    if direction_ambiguous:
        gate_reasons.append(f"ロング{long_total}点とショート{short_total}点の差が小さく方向が不明瞭")

    # --- ロット計算(切り捨て。四捨五入だとリスク上限を超える可能性がある) ---
    if sltp.get("valid"):
        lot = calc_lot_size(account_balance=272_778, risk_percent=3.5, risk_pips=sltp["risk_pips"])
        print("\n【ロット計算(口座残高272,778円・リスク3.5%想定)】")
        for k, v in lot.items():
            print(f"  {k}: {v}")
        print(f"  → SBI画面(×1,000単位)への入力数値: {lot['input_quantity_x1000']}")
        print("  ※スプレッド0.3pips・スリッページ0.2pipsを実効リスクに算入済み")
        if lot["input_quantity_x1000"] < 1:
            gate_reasons.append("計算上のロットが最小取引単位未満")
    else:
        lot = {"recommended_lot": 0, "input_quantity_x1000": 0}

    # --- Discord通知(ハードゲート方式) ---
    if gate_reasons:
        print(f"\n[通知なし・ハードゲート作動] スコア{score['合計']}点でも以下の理由により無条件見送り:")
        for r in gate_reasons:
            print(f"  - {r}")
    elif score["合計"] >= 80:
        send_discord_notification(score, direction, sltp, lot, DISCORD_WEBHOOK_URL)
    else:
        print(f"\n[通知なし] スコア{score['合計']}点のため見送り(80点未満は通知しません)")

    # --- ログ用チャンネルへは毎回記録を送信(見送りも含む) ---
    send_discord_log(
        score, direction, sltp, lot, gate_reasons,
        extra_info={"long_total": long_total, "short_total": short_total},
        webhook_url=DISCORD_LOG_WEBHOOK_URL,
    )


if __name__ == "__main__":
    main()
