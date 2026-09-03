"""Technical indicator engine built on pandas/numpy.

Every function takes a DataFrame that has at least 'close', 'high',
'low' and optionally 'volume' columns, and returns either a Series or
an enriched DataFrame. The API is vectorised and returns NaN for the
warm-up period so downstream code can reason about "ready" values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(series, fast) - ema(series, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def volume_profile(df: pd.DataFrame, period: int = 20) -> pd.Series:
    avg = df["volume"].rolling(window=period, min_periods=period).mean()
    ratio = df["volume"] / avg.replace(0, np.nan)
    return ratio


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a comprehensive set of indicators to a bars DataFrame."""
    out = df.copy()
    c = out["close"]
    out["ema_9"] = ema(c, 9)
    out["ema_21"] = ema(c, 21)
    out["ema_50"] = ema(c, 50)
    out["sma_50"] = sma(c, 50)
    out["sma_200"] = sma(c, 200)
    out["rsi_14"] = rsi(c, 14)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(c)
    out["atr_14"] = atr(out, 14)
    bb_mid, bb_upper, bb_lower = bollinger(c)
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower
    out["vol_ratio"] = volume_profile(out, 20)
    out["smoothed"] = ema(c, 8)
    return out