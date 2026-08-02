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
            swing_low.iloc[i] = True

    result = df.copy()
    result["swing_high"] = swing_high
    result["swing_low"] = swing_low
    return result


def extract_key_levels(df_with_swings: pd.DataFrame, lookback: int = 100, top_n: int = 5):
    """直近lookback本のスイング高値/安値から重要水平線を抽出する。
    近い価格帯(0.1円以内)は1本にまとめ、タッチ回数でランク付け。
    """
    recent = df_with_swings.tail(lookback)
    highs = recent.loc[recent["swing_high"], "high"].tolist()
    lows = recent.loc[recent["swing_low"], "low"].tolist()
    points = highs + lows

    if not points:
        return []

    points = sorted(points)
    clusters = []
    current_cluster = [points[0]]
    for p in points[1:]:
        if abs(p - current_cluster[-1]) <= 0.1:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]
    clusters.append(current_cluster)

    levels = [
        {"price": round(np.mean(c), 3), "touch_count": len(c)}
        for c in clusters
    ]
    levels.sort(key=lambda x: x["touch_count"], reverse=True)
    return levels[:top_n]


# ============================================================
# 4. トレンド / レンジ判定(ダウ理論ベース)
# ============================================================

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
    """1h/4h/日足のトレンド方向一致度から環境認識スコアを機械的に算出する。"""
    trends = [trend_1h["trend"], trend_4h["trend"], trend_1d["trend"]]
    up = trends.count("上昇トレンド")
    down = trends.count("下降トレンド")

    if up == 3:
        return {"score": 20, "direction": "long", "detail": "1h/4h/日足すべて上昇一致"}
    if down == 3:
        return {"score": 20, "direction": "short", "detail": "1h/4h/日足すべて下降一致"}
    if up == 2:
        return {"score": 10, "direction": "long", "detail": "3時間足中2つが上昇一致"}
    if down == 2:
        return {"score": 10, "direction": "short", "detail": "3時間足中2つが下降一致"}
    return {"score": 0, "direction": None, "detail": "方向性バラバラ(環境不一致)"}


def judge_trend(df_with_swings: pd.DataFrame, lookback_swings: int = 4) -> dict:
    """直近のスイング高値・安値の切り上げ/切り下げでトレンドを機械的に判定する。"""
    swing_highs = df_with_swings.loc[df_with_swings["swing_high"], "high"].tail(lookback_swings)
    swing_lows = df_with_swings.loc[df_with_swings["swing_low"], "low"].tail(lookback_swings)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"trend": "不明", "reason": "スイング点不足"}

    hh = swing_highs.is_monotonic_increasing
    hl = swing_lows.is_monotonic_increasing
    lh = swing_highs.is_monotonic_decreasing
    ll = swing_lows.is_monotonic_decreasing

    if hh and hl:
        return {"trend": "上昇トレンド", "reason": "高値・安値ともに切り上げ"}
    if lh and ll:
        return {"trend": "下降トレンド", "reason": "高値・安値ともに切り下げ"}
    return {"trend": "レンジ", "reason": "高値・安値の方向不一致"}


# ============================================================
# 5. DXY / 米2年債との連動判定
# ============================================================

def judge_macro_alignment(usdjpy_df: pd.DataFrame, dxy_df: pd.DataFrame, us2y_df: pd.DataFrame,
                            direction: str, lookback: int = 5) -> dict:
    """直近lookback本の変化方向が理論上の相関と一致しているかを判定する。
    USDJPYはDXYと正相関、米2年債利回りと正相関が理論値。
    direction: 'long' or 'short'(想定エントリー方向)
    """
    def recent_slope(series: pd.Series, n: int) -> float:
        s = series.dropna().tail(n)
        if len(s) < 2:
            return 0.0
        return s.iloc[-1] - s.iloc[0]

    dxy_slope = recent_slope(dxy_df["close"], lookback)
    y2_slope = recent_slope(us2y_df["yield"], lookback)

    dxy_ok = (dxy_slope > 0) == (direction == "long")
    y2_ok = (y2_slope > 0) == (direction == "long")

    return {
        "dxy_slope": round(float(dxy_slope), 4),
        "dxy_aligned": bool(dxy_ok),
        "us2y_slope": round(float(y2_slope), 4),
        "us2y_aligned": bool(y2_ok),
    }


# ============================================================
# 6. スコアリングエンジン(100点満点)
# ============================================================

def calc_score(env_score: int, trend_result: dict, macro_result: dict,
               key_levels_near: bool, atr_ratio_ok: bool, near_indicator_time: bool) -> dict:
    """
    配点:
      環境認識(上位足整合)   20点(1h/4h/日足の一致度で0/10/20点を判定関数側で算出)
      トレンド判定            20点
      重要ライン近接          15点
      DXY方向一致             15点
      米2年債方向一致         10点
      ボラティリティ適正      10点
      経済指標               10点(直前直後は減点)
    """
    scores = {}
    scores["環境認識"] = env_score
    scores["トレンド"] = 20 if trend_result["trend"] in ("上昇トレンド", "下降トレンド") else 5
    scores["重要ライン"] = 15 if key_levels_near else 5
    scores["DXY"] = 15 if macro_result["dxy_aligned"] else 0
    scores["米2年債"] = 10 if macro_result["us2y_aligned"] else 0
    scores["ボラティリティ"] = 10 if atr_ratio_ok else 3
    scores["経済指標"] = 0 if near_indicator_time else 10

    total = sum(scores.values())
    scores["合計"] = total
    scores["推奨"] = "エントリー推奨" if total >= 80 else "見送り推奨"
    return scores


# ============================================================
# 7. 損切り・利確・ロット計算
# ============================================================

def suggest_sl_tp(entry_price: float, direction: str, atr: float,
                   nearest_swing_low: float, nearest_swing_high: float,
                   rr_min: float = 2.0) -> dict:
    """直近スイング+ATRバッファでSL、RR比からTPを算出する。"""
    buffer = atr * 0.3
    if direction == "long":
        sl = nearest_swing_low - buffer
        risk = entry_price - sl
        tp = entry_price + risk * rr_min
    else:
        sl = nearest_swing_high + buffer
        risk = sl - entry_price
        tp = entry_price - risk * rr_min

    rr = rr_min if risk > 0 else 0
    return {
        "entry": round(entry_price, 3),
        "stop_loss": round(sl, 3),
        "take_profit": round(tp, 3),
        "risk_pips": round(risk * 100, 1),  # USDJPYは1pip=0.01円換算(簡易)
        "rr_ratio": rr,
    }


def calc_lot_size(account_balance: float, risk_percent: float, risk_pips: float,
                   pip_value_per_lot: float = 1000.0) -> dict:
    """
    account_balance: 口座残高(円)
    risk_percent: 1トレードあたりの許容リスク(%) 例: 1.0
    risk_pips: 損切りまでのpips数
    pip_value_per_lot: 1lot(10万通貨)あたり1pipの評価額(円) ※USDJPYは概ね1000円/lot
    """
    risk_amount = account_balance * (risk_percent / 100)
    if risk_pips <= 0:
        return {"lot": 0, "reason": "リスク幅が不正です"}
    lot = risk_amount / (risk_pips * pip_value_per_lot)
    return {
        "risk_amount_jpy": round(risk_amount, 0),
        "recommended_lot": round(lot, 2),
    }


# ============================================================
# 8. メイン実行(サンプル)
# ============================================================

import os

# Discord WebhookのURL。
# Colabで試す場合はここに直接貼ってOK。
# GitHub Actions移行後はSecrets(環境変数)から読み込む設計にしてあります。
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "ここにWebhook URLを貼る")


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
    for k in ["環境認識", "トレンド", "重要ライン", "DXY", "米2年債", "ボラティリティ", "経済指標"]:
        lines.append(f"・{k}: {score[k]}点")

    lines += [
        "",
        "**エントリープラン**",
        f"エントリー: {sltp['entry']}",
        f"損切り: {sltp['stop_loss']}",
        f"利確: {sltp['take_profit']}",
        f"RR比: 1:{sltp['rr_ratio']}",
        f"推奨ロット: {lot.get('recommended_lot', '-')}",
    ]

    message = {"content": "\n".join(lines)}
    try:
        resp = requests.post(webhook_url, json=message, timeout=10)
        print(f"[Discord通知] ステータスコード: {resp.status_code}")
    except Exception as e:
        print(f"[Discord通知エラー] {e}")


def main():
    print("=== USDJPY デイトレ支援ツール Phase1 ===\n")

    # --- データ取得 ---
    df_5m = fetch_usdjpy(interval="5m", period="5d")
    df_1h = fetch_usdjpy(interval="1h", period="60d")
    df_1d = fetch_usdjpy(interval="1d", period="1y")
    dxy_1d = fetch_dxy(period="30d")
    us2y = fetch_us2y_yield(days=30)

    # --- 指標計算 ---
    df_5m["ema20"] = calc_ema(df_5m, 20)
    df_5m["atr"] = calc_atr(df_5m, 14)
    df_5m["vwap"] = calc_vwap_daily(df_5m)

    # --- スイング・水平線 ---
    df_1h_sw = find_swing_points(df_1h, window=3)
    key_levels = extract_key_levels(df_1h_sw, lookback=100, top_n=5)
    trend_1h = judge_trend(df_1h_sw)

    print("【重要水平線(直近1hより)】")
    for lv in key_levels:
        print(f"  {lv['price']} 円  (タッチ回数: {lv['touch_count']})")
    print(f"\n【1h足トレンド判定】 {trend_1h['trend']} ({trend_1h['reason']})")

    # --- 環境認識自動判定(1h/4h/日足) ---
    df_4h = resample_ohlcv(df_1h, "4h")
    df_4h_sw = find_swing_points(df_4h, window=2)
    trend_4h = judge_trend(df_4h_sw, lookback_swings=3)

    df_1d_sw = find_swing_points(df_1d, window=2)
    trend_1d = judge_trend(df_1d_sw, lookback_swings=3)

    env = judge_environment(trend_1h, trend_4h, trend_1d)
    print(f"\n【環境認識(マルチタイムフレーム)】")
    print(f"  1h : {trend_1h['trend']}")
    print(f"  4h : {trend_4h['trend']}")
    print(f"  日足: {trend_1d['trend']}")
    print(f"  → {env['detail']}  (環境認識スコア: {env['score']}点)")

    # --- マクロ連動判定 ---
    # 方向は環境認識の一致方向を優先。一致がなければ1h足の方向で仮判定。
    direction = env["direction"] or ("long" if trend_1h["trend"] == "上昇トレンド" else "short")
    macro = judge_macro_alignment(df_5m, dxy_1d, us2y, direction=direction, lookback=5)
    print(f"\n【マクロ連動判定 ({direction})】")
    print(f"  DXYスロープ: {macro['dxy_slope']}  一致: {macro['dxy_aligned']}")
    print(f"  米2年債スロープ: {macro['us2y_slope']}  一致: {macro['us2y_aligned']}")

    # --- 簡易スコア計算(サンプル値、実運用では各種判定関数の結果を渡す) ---
    latest_price = df_5m["close"].iloc[-1]
    latest_atr = df_5m["atr"].iloc[-1]
    key_level_near = any(abs(latest_price - lv["price"]) < 0.15 for lv in key_levels)
    atr_ratio_ok = 0.03 < latest_atr < 0.20  # 5分足ATRの適正レンジ(仮)

    score = calc_score(
        env_score=env["score"],
        trend_result=trend_1h,
        macro_result=macro,
        key_levels_near=key_level_near,
        atr_ratio_ok=atr_ratio_ok,
        near_indicator_time=False,  # Phase2で経済指標カレンダー連携後に自動判定
    )
    print("\n【スコア内訳】")
    for k, v in score.items():
        print(f"  {k}: {v}")

    # --- SL/TP/ロット計算 ---
    if key_levels:
        levels_below = [lv["price"] for lv in key_levels if lv["price"] < latest_price]
        levels_above = [lv["price"] for lv in key_levels if lv["price"] > latest_price]
        # 「直近」の水平線 = 下側は最大値(entryに一番近い下のライン)、
        #                    上側は最小値(entryに一番近い上のライン)
        nearest_low = max(levels_below) if levels_below else latest_price - 0.5
        nearest_high = min(levels_above) if levels_above else latest_price + 0.5
    else:
        nearest_low, nearest_high = latest_price - 0.5, latest_price + 0.5

    sltp = suggest_sl_tp(
        entry_price=latest_price,
        direction=direction,
        atr=latest_atr,
        nearest_swing_low=nearest_low,
        nearest_swing_high=nearest_high,
    )
    print(f"\n【SL/TP提案 ({direction})】")
    for k, v in sltp.items():
        print(f"  {k}: {v}")

    lot = calc_lot_size(account_balance=1_000_000, risk_percent=1.0, risk_pips=sltp["risk_pips"])
    print("\n【ロット計算(口座残高100万円・リスク1%想定)】")
    for k, v in lot.items():
        print(f"  {k}: {v}")

    # --- Discord通知(80点以上のみ) ---
    if score["合計"] >= 80:
        send_discord_notification(score, direction, sltp, lot, DISCORD_WEBHOOK_URL)
    else:
        print(f"\n[通知なし] スコア{score['合計']}点のため見送り(80点未満は通知しません)")


if __name__ == "__main__":
    main()
