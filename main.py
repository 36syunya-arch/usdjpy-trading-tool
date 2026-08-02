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
    """新しいyfinanceが返すM
