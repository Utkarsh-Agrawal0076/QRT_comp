"""
Technical Indicators Module
This module contains functions to calculate various technical indicators from OHLCV data.
All functions expect a pandas Series or DataFrame with the relevant price/volume data.
Supports parallel processing for efficient batch computation.
"""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def relative_strength_index(close, period=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Measures the speed and magnitude of recent price changes to assess whether 
    a stock is overbought or oversold. Values above 70 suggest overbought, 
    below 30 suggest oversold.
    
    Args:
        close: pandas Series of close prices
        period: lookback period (default 14)
    
    Returns:
        pandas Series of RSI values
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def williams_r(high, low, close, period=14):
    """
    Calculate Williams %R.
    
    Measures the level of closing price relative to the highest high over a period.
    Ranges from -100 to 0. Values below -80 indicate oversold, above -20 indicate overbought.
    
    Args:
        high: pandas Series of high prices
        low: pandas Series of low prices
        close: pandas Series of close prices
        period: lookback period (default 14)
    
    Returns:
        pandas Series of Williams %R values
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    return williams_r


def volatility_20(close, period=20):
    """
    Calculate 20-day volatility.
    
    Measures the standard deviation of returns over a 20-day period.
    Higher values indicate increased market uncertainty and risk.
    
    Args:
        close: pandas Series of close prices
        period: lookback period (default 20)
    
    Returns:
        pandas Series of volatility values
    """
    returns = close.pct_change()
    volatility = returns.rolling(window=period).std()
    
    return volatility


def volatility_60(close, period=60):
    """
    Calculate 60-day volatility.
    
    Measures the standard deviation of returns over a 60-day period.
    Reflects longer-term price fluctuations.
    
    Args:
        close: pandas Series of close prices
        period: lookback period (default 60)
    
    Returns:
        pandas Series of volatility values
    """
    returns = close.pct_change()
    volatility = returns.rolling(window=period).std()
    
    return volatility


def trend_1_3(close):
    """
    Calculate Trend (1-3) indicator.
    
    Calculates the difference between short-term EWMA (span=1) and long-term EWMA (span=3)
    of returns, normalized by exponentially weighted standard deviation (span=3).
    
    Args:
        close: pandas Series of close prices
    
    Returns:
        pandas Series of trend indicator values
    """
    returns = close.pct_change()
    
    ewma_1 = returns.ewm(span=1, adjust=False).mean()
    ewma_3 = returns.ewm(span=3, adjust=False).mean()
    
    ewstd_3 = returns.ewm(span=3, adjust=False).std()
    
    trend = (ewma_1 - ewma_3) / ewstd_3
    
    return trend


def trend_5_20(close):
    """
    Calculate Trend (5-20) indicator.
    
    Calculates the difference between short-term EWMA (span=5) and long-term EWMA (span=20)
    of returns, normalized by exponentially weighted standard deviation (span=20).
    
    Args:
        close: pandas Series of close prices
    
    Returns:
        pandas Series of trend indicator values
    """
    returns = close.pct_change()
    
    ewma_5 = returns.ewm(span=5, adjust=False).mean()
    ewma_20 = returns.ewm(span=20, adjust=False).mean()
    
    ewstd_20 = returns.ewm(span=20, adjust=False).std()
    
    trend = (ewma_5 - ewma_20) / ewstd_20
    
    return trend


def trend_20_60(close):
    """
    Calculate Trend (20-60) indicator.
    
    Calculates the difference between short-term EWMA (span=20) and long-term EWMA (span=60)
    of returns, normalized by exponentially weighted standard deviation (span=20).
    
    Args:
        close: pandas Series of close prices
    
    Returns:
        pandas Series of trend indicator values
    """
    returns = close.pct_change()
    
    ewma_20 = returns.ewm(span=20, adjust=False).mean()
    ewma_60 = returns.ewm(span=60, adjust=False).mean()
    
    ewstd_20 = returns.ewm(span=20, adjust=False).std()
    
    trend = (ewma_20 - ewma_60) / ewstd_20
    
    return trend


def average_true_range(high, low, close, period=14):
    """
    Calculate Average True Range (ATR).
    
    Measures market volatility by averaging the true range over a period.
    True range is the greatest of: high-low, |high-prev_close|, |low-prev_close|.
    
    Args:
        high: pandas Series of high prices
        low: pandas Series of low prices
        close: pandas Series of close prices
        period: lookback period (default 14)
    
    Returns:
        pandas Series of ATR values
    """
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def macd(close, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Calculates the difference between 12-day and 26-day EMAs, then smooths with 9-day EMA.
    Positive values indicate bullish momentum, negative values suggest bearish momentum.
    
    Args:
        close: pandas Series of close prices
        fast_period: fast EMA period (default 12)
        slow_period: slow EMA period (default 26)
        signal_period: signal line EMA period (default 9)
    
    Returns:
        tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def trix(close, period=15):
    """
    Calculate TRIX (Triple Exponential Moving Average Oscillator).
    
    Calculates the 1-day percentage change of the triple-smoothed EMA over a 15-day window.
    Positive values indicate increasing momentum, negative values suggest weakening momentum.
    
    Args:
        close: pandas Series of close prices
        period: EMA period (default 15)
    
    Returns:
        pandas Series of TRIX values
    """
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    
    trix = ema3.pct_change() * 10000  # Scale for readability
    
    return trix


def commodity_channel_index(high, low, close, period=20):
    """
    Calculate Commodity Channel Index (CCI).
    
    Measures deviation of typical price from its SMA, scaled by mean absolute deviation.
    Positive values indicate strong upward momentum, negative values suggest downward momentum.
    Overbought level typically at +100, oversold at -100.
    
    Args:
        high: pandas Series of high prices
        low: pandas Series of low prices
        close: pandas Series of close prices
        period: lookback period (default 20)
    
    Returns:
        pandas Series of CCI values
    """
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    
    mad = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x)))
    )
    
    cci = (typical_price - sma) / (0.015 * mad)
    
    return cci


def chande_momentum_oscillator(close, period=14):
    """
    Calculate Chande Momentum Oscillator (CMO).
    
    Measures the difference between sum of gains and sum of losses over a period,
    normalized to -100 to 100 scale. Higher values indicate strong upward momentum.
    Overbought at +50, oversold at -50.
    
    Args:
        close: pandas Series of close prices
        period: lookback period (default 14)
    
    Returns:
        pandas Series of CMO values
    """
    delta = close.diff()
    
    gains = delta.where(delta > 0, 0).rolling(window=period).sum()
    losses = -delta.where(delta < 0, 0).rolling(window=period).sum()
    
    cmo = ((gains - losses) / (gains + losses)) * 100
    
    return cmo


def ichimoku(high, low, close, conversion_period=9, base_period=26, lagging_period=26):
    """
    Calculate Ichimoku Cloud components.
    
    Calculates conversion line (9-period midpoint) and base line (26-period midpoint)
    to determine trends. Returns support, resistance, and trend direction.
    
    Args:
        high: pandas Series of high prices
        low: pandas Series of low prices
        close: pandas Series of close prices
        conversion_period: conversion line period (default 9)
        base_period: base line period (default 26)
        lagging_period: lagging span period (default 26)
    
    Returns:
        dict with 'conversion_line', 'base_line', 'leading_a', 'leading_b', 'lagging'
    """
    conversion_line = (
        (high.rolling(window=conversion_period).max() + 
         low.rolling(window=conversion_period).min()) / 2
    )
    
    base_line = (
        (high.rolling(window=base_period).max() + 
         low.rolling(window=base_period).min()) / 2
    )
    
    leading_a = (conversion_line + base_line) / 2
    leading_b = (
        (high.rolling(window=base_period*2).max() + 
         low.rolling(window=base_period*2).min()) / 2
    )
    
    lagging = close.shift(lagging_period)
    
    return {
        'conversion_line': conversion_line,
        'base_line': base_line,
        'leading_a': leading_a,
        'leading_b': leading_b,
        'lagging': lagging
    }


def know_sure_thing(close, roc_periods=[10, 15, 20, 30], sma_periods=[10, 10, 10, 15]):
    """
    Calculate Know Sure Thing (KST).
    
    Weighted sum of rate of change (ROC) over multiple periods, each smoothed with SMA.
    Positive values indicate bullish momentum, negative values suggest bearish momentum.
    
    Args:
        close: pandas Series of close prices
        roc_periods: periods for ROC calculation (default [10, 15, 20, 30])
        sma_periods: periods for SMA smoothing (default [10, 10, 10, 15])
    
    Returns:
        pandas Series of KST values
    """
    weights = [1, 2, 3, 4]
    kst = pd.Series(index=close.index, dtype=float)
    kst[:] = 0
    
    for roc_period, sma_period, weight in zip(roc_periods, sma_periods, weights):
        roc = close.pct_change(roc_period) * 100
        sma_roc = roc.rolling(window=sma_period).mean()
        kst += sma_roc * weight
    
    return kst


def ultimate_oscillator(high, low, close, period1=7, period2=14, period3=28):
    """
    Calculate Ultimate Oscillator.
    
    Weighted average of three different periods of buying pressure to true range ratio.
    Aims to reduce false signals by incorporating multiple timeframes.
    Values above 70 indicate overbought, below 30 suggest oversold.
    
    Args:
        high: pandas Series of high prices
        low: pandas Series of low prices
        close: pandas Series of close prices
        period1: first period (default 7)
        period2: second period (default 14)
        period3: third period (default 28)
    
    Returns:
        pandas Series of Ultimate Oscillator values
    """
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    bp = close - pd.concat([low, close.shift()], axis=1).min(axis=1)
    
    avg1 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
    avg2 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
    avg3 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()
    
    uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
    
    return uo


def aroon(high, low, period=26):
    """
    Calculate Aroon Indicator.
    
    Measures the difference between time since highest high and lowest low over a period
    to determine the strength and direction of a trend.
    
    Args:
        high: pandas Series of high prices
        low: pandas Series of low prices
        period: lookback period (default 26)
    
    Returns:
        dict with 'aroon_up', 'aroon_down', 'aroon_oscillator'
    """
    aroon_up = (high.rolling(window=period).apply(
        lambda x: period - np.argmax(x[::-1]) - 1
    ) / period) * 100
    
    aroon_down = (low.rolling(window=period).apply(
        lambda x: period - np.argmin(x[::-1]) - 1
    ) / period) * 100
    
    aroon_oscillator = aroon_up - aroon_down
    
    return {
        'aroon_up': aroon_up,
        'aroon_down': aroon_down,
        'aroon_oscillator': aroon_oscillator
    }


def stochastic_oscillator(high, low, close, k_period=14, d_period=3):
    """
    Calculate Stochastic Oscillator.
    
    Measures the position of closing price relative to the high-low range.
    Ranges from 0 to 100. Values above 80 indicate overbought, below 20 suggest oversold.
    
    Args:
        high: pandas Series of high prices
        low: pandas Series of low prices
        close: pandas Series of close prices
        k_period: period for %K calculation (default 14)
        d_period: period for %D (SMA of %K) (default 3)
    
    Returns:
        dict with 'k_percent', 'd_percent'
    """
    highest_high = high.rolling(window=k_period).max()
    lowest_low = low.rolling(window=k_period).min()
    
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return {
        'k_percent': k_percent,
        'd_percent': d_percent
    }


def on_balance_volume(close, volume):
    """
    Calculate On-Balance Volume (OBV).
    
    Tracks cumulative volume flow to measure buying and selling pressure.
    Adds volume on up days, subtracts on down days.
    Increasing OBV confirms uptrend, decreasing OBV signals downtrend.
    
    Args:
        close: pandas Series of close prices
        volume: pandas Series of volume data
    
    Returns:
        pandas Series of OBV values
    """
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = 0
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def ease_of_movement(high, low, close, volume, period=14):
    """
    Calculate Ease of Movement (EMV).
    
    Measures the relationship between price movement and volume.
    High EMV indicates price rising with little volume (strong upward momentum).
    Low/negative EMV signals downward pressure.
    
    Args:
        high: pandas Series of high prices
        low: pandas Series of low prices
        close: pandas Series of close prices
        volume: pandas Series of volume data
        period: lookback period for SMA (default 14)
    
    Returns:
        pandas Series of EMV values
    """
    emv_raw = (high + low) / 2 - (high + low).shift(1) / 2
    emv_raw = emv_raw / ((high - low) / volume)
    
    emv = emv_raw.rolling(window=period).mean()
    
    return emv


def chaikin_money_flow(high, low, close, volume, period=20):
    """
    Calculate Chaikin Money Flow (CMF).
    
    Measures money flow volume over a period to assess buying and selling pressure.
    Positive values indicate accumulation (buying pressure).
    Negative values suggest distribution (selling pressure).
    
    Args:
        high: pandas Series of high prices
        low: pandas Series of low prices
        close: pandas Series of close prices
        volume: pandas Series of volume data
        period: lookback period (default 20)
    
    Returns:
        pandas Series of CMF values
    """
    mfv = (((close - low) - (high - close)) / (high - low)) * volume
    cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    
    return cmf


def accumulation_distribution_index(high, low, close, volume):
    """
    Calculate Accumulation/Distribution Index (A/D).
    
    Measures cumulative money flow volume to determine strength of buying or selling pressure.
    Calculates Closeness Value (CLV) using high, low, and close prices, then multiplies by volume.
    Tracks whether money is flowing into (accumulation) or out of (distribution) an asset.
    
    Args:
        high: pandas Series of high prices
        low: pandas Series of low prices
        close: pandas Series of close prices
        volume: pandas Series of volume data
    
    Returns:
        pandas Series of A/D Index values
    """
    clv = ((close - low) - (high - close)) / (high - low)
    ad = (clv * volume).cumsum()
    
    return ad


def volume_feature(volume):
    """
    Return volume as a feature.
    
    Args:
        volume: pandas Series of volume data
    
    Returns:
        pandas Series of volume values
    """
    return volume


def calculate_all_indicators(data_matrix):
    """
    Calculate all technical indicators for a data matrix.
    
    Args:
        data_matrix: pandas DataFrame with MultiIndex columns (metric, ticker)
                    Expected metrics: Open, High, Low, Close, Adj Close, Volume
    
    Returns:
        dict mapping indicator names to DataFrames of values
    """
    indicators = {}
    
    # Extract price and volume data
    open_prices = data_matrix['Open']
    high_prices = data_matrix['High']
    low_prices = data_matrix['Low']
    close_prices = data_matrix['Close']
    adj_close_prices = data_matrix['Adj Close']
    volume_data = data_matrix['Volume']
    
    # Calculate indicators for each ticker
    for ticker in open_prices.columns:
        close = close_prices[ticker]
        high = high_prices[ticker]
        low = low_prices[ticker]
        volume = volume_data[ticker]
        
        if ticker not in indicators:
            indicators[ticker] = {}
        
        # Momentum indicators
        indicators[ticker]['relative_strength_index'] = relative_strength_index(close)
        indicators[ticker]['williams_r'] = williams_r(high, low, close)
        indicators[ticker]['rsi'] = relative_strength_index(close)
        
        # Volatility indicators
        indicators[ticker]['volatility_20'] = volatility_20(close)
        indicators[ticker]['volatility_60'] = volatility_60(close)
        
        # Trend indicators
        indicators[ticker]['trend_1_3'] = trend_1_3(close)
        indicators[ticker]['trend_5_20'] = trend_5_20(close)
        indicators[ticker]['trend_20_60'] = trend_20_60(close)
        
        # Average True Range
        indicators[ticker]['average_true_range'] = average_true_range(high, low, close)
        
        # MACD
        macd_line, signal_line, histogram = macd(close)
        indicators[ticker]['macd'] = macd_line
        indicators[ticker]['macd_signal'] = signal_line
        indicators[ticker]['macd_histogram'] = histogram
        
        # TRIX
        indicators[ticker]['trix'] = trix(close)
        
        # CCI
        indicators[ticker]['commodity_channel_index'] = commodity_channel_index(high, low, close)
        
        # CMO
        indicators[ticker]['chande_momentum_oscillator'] = chande_momentum_oscillator(close)
        
        # Ichimoku
        ichimoku_dict = ichimoku(high, low, close)
        indicators[ticker]['ichimoku_conversion'] = ichimoku_dict['conversion_line']
        indicators[ticker]['ichimoku_base'] = ichimoku_dict['base_line']
        indicators[ticker]['ichimoku_leading_a'] = ichimoku_dict['leading_a']
        indicators[ticker]['ichimoku_leading_b'] = ichimoku_dict['leading_b']
        
        # KST
        indicators[ticker]['know_sure_thing'] = know_sure_thing(close)
        
        # Ultimate Oscillator
        indicators[ticker]['ultimate_oscillator'] = ultimate_oscillator(high, low, close)
        
        # Aroon
        aroon_dict = aroon(high, low)
        indicators[ticker]['aroon_up'] = aroon_dict['aroon_up']
        indicators[ticker]['aroon_down'] = aroon_dict['aroon_down']
        indicators[ticker]['aroon_oscillator'] = aroon_dict['aroon_oscillator']
        
        # Stochastic
        stoch_dict = stochastic_oscillator(high, low, close)
        indicators[ticker]['stochastic_k'] = stoch_dict['k_percent']
        indicators[ticker]['stochastic_d'] = stoch_dict['d_percent']
        
        # Volume indicators
        indicators[ticker]['on_balance_volume'] = on_balance_volume(close, volume)
        indicators[ticker]['ease_of_movement'] = ease_of_movement(high, low, close, volume)
        indicators[ticker]['chaikin_money_flow'] = chaikin_money_flow(high, low, close, volume)
        indicators[ticker]['accumulation_distribution_index'] = accumulation_distribution_index(high, low, close, volume)
        indicators[ticker]['volume'] = volume_data[ticker]
    
    return indicators


def _calculate_indicators_for_ticker(ticker, data_matrix):
    """
    Helper function to calculate all indicators for a single ticker.
    Used for parallel processing.
    
    Args:
        ticker: ticker symbol
        data_matrix: pandas DataFrame with MultiIndex columns
    
    Returns:
        tuple of (ticker, indicators_dict)
    """
    open_prices = data_matrix['Open']
    high_prices = data_matrix['High']
    low_prices = data_matrix['Low']
    close_prices = data_matrix['Close']
    volume_data = data_matrix['Volume']
    
    close = close_prices[ticker]
    high = high_prices[ticker]
    low = low_prices[ticker]
    volume = volume_data[ticker]
    
    ticker_indicators = {}
    
    # Momentum indicators
    # ticker_indicators['relative_strength_index'] = relative_strength_index(close)
    # ticker_indicators['williams_r'] = williams_r(high, low, close)
    # ticker_indicators['rsi'] = relative_strength_index(close)
    
    # Volatility indicators
    # ticker_indicators['volatility_20'] = volatility_20(close)
    # ticker_indicators['volatility_60'] = volatility_60(close)
    
    # Trend indicators
    # ticker_indicators['trend_1_3'] = trend_1_3(close)
    # ticker_indicators['trend_5_20'] = trend_5_20(close)
    # ticker_indicators['trend_20_60'] = trend_20_60(close)
    
    # Average True Range
    ticker_indicators['average_true_range'] = average_true_range(high, low, close)
    
    # MACD
    # macd_line, signal_line, histogram = macd(close)
    # ticker_indicators['macd'] = macd_line
    # ticker_indicators['macd_signal'] = signal_line
    # ticker_indicators['macd_histogram'] = histogram
    
    # TRIX
    # ticker_indicators['trix'] = trix(close)
    
    # CCI
    # ticker_indicators['commodity_channel_index'] = commodity_channel_index(high, low, close)
    
    # CMO
    # ticker_indicators['chande_momentum_oscillator'] = chande_momentum_oscillator(close)
    
    # Ichimoku
    # ichimoku_dict = ichimoku(high, low, close)
    # ticker_indicators['ichimoku_conversion'] = ichimoku_dict['conversion_line']
    # ticker_indicators['ichimoku_base'] = ichimoku_dict['base_line']
    # ticker_indicators['ichimoku_leading_a'] = ichimoku_dict['leading_a']
    # ticker_indicators['ichimoku_leading_b'] = ichimoku_dict['leading_b']
    
    # KST
    # ticker_indicators['know_sure_thing'] = know_sure_thing(close)
    
    # Ultimate Oscillator
    # ticker_indicators['ultimate_oscillator'] = ultimate_oscillator(high, low, close)
    
    # Aroon
    # aroon_dict = aroon(high, low)
    # ticker_indicators['aroon_up'] = aroon_dict['aroon_up']
    # ticker_indicators['aroon_down'] = aroon_dict['aroon_down']
    # ticker_indicators['aroon_oscillator'] = aroon_dict['aroon_oscillator']
    
    # Stochastic
    # stoch_dict = stochastic_oscillator(high, low, close)
    # ticker_indicators['stochastic_k'] = stoch_dict['k_percent']
    # ticker_indicators['stochastic_d'] = stoch_dict['d_percent']
    
    # Volume indicators
    # ticker_indicators['on_balance_volume'] = on_balance_volume(close, volume)
    # ticker_indicators['ease_of_movement'] = ease_of_movement(high, low, close, volume)
    # ticker_indicators['chaikin_money_flow'] = chaikin_money_flow(high, low, close, volume)
    # ticker_indicators['accumulation_distribution_index'] = accumulation_distribution_index(high, low, close, volume)
    # ticker_indicators['volume'] = volume_data[ticker]
    
    return ticker, ticker_indicators


def calculate_all_indicators_parallel(data_matrix, n_jobs=-1, verbose=1):
    """
    Calculate all technical indicators for all tickers using parallel processing.
    
    This version is much faster for large datasets with many tickers.
    
    Args:
        data_matrix: pandas DataFrame with MultiIndex columns (metric, ticker)
                    Expected metrics: Open, High, Low, Close, Adj Close, Volume
        n_jobs: number of jobs for parallel processing
               -1 = use all processors
               1 = no parallel processing
               2, 3, etc. = number of processors to use
        verbose: verbosity level (0=silent, 1=progress bar, 10=detailed)
    
    Returns:
        dict mapping indicator names to DataFrames (dates x tickers)
        Structure: {indicator_name: DataFrame[dates x tickers], ...}
    """
    tickers = data_matrix['Close'].columns.tolist()
    
    # Use Parallel with tqdm for progress tracking
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_calculate_indicators_for_ticker)(ticker, data_matrix)
        for ticker in tqdm(tickers, desc="Computing indicators", unit="ticker")
    )
    
    # Convert results to ticker-indexed dictionary first
    ticker_indicators = {ticker: ind_dict for ticker, ind_dict in results}
    
    # Transform to indicator-indexed format with DataFrames
    # Get all unique indicator names
    sample_ticker = next(iter(ticker_indicators.keys()))
    indicator_names = list(ticker_indicators[sample_ticker].keys())
    
    # Build indicator -> DataFrame structure
    indicators_dfs = {}
    
    for indicator_name in indicator_names:
        # Collect all Series for this indicator across all tickers
        data_dict = {}
        
        for ticker in ticker_indicators.keys():
            indicator_series = ticker_indicators[ticker].get(indicator_name)
            if indicator_series is not None:
                data_dict[ticker] = indicator_series
        
        # Create DataFrame from dict (automatically aligns by index)
        if data_dict:
            indicator_df = pd.DataFrame(data_dict)
            indicators_dfs[indicator_name] = indicator_df
    
    return indicators_dfs


def _process_date_features(date, all_indicators, tickers):
    """
    Helper function to process features for a single date.
    Used for parallel processing in build_features_dataframe_parallel.
    
    Args:
        date: single date to process
        all_indicators: dict with all indicators
        tickers: list of ticker symbols
    
    Returns:
        dict with features for this date
    """
    row = {'Date': date}
    
    for ticker in tickers:
        if ticker not in all_indicators:
            continue
            
        ticker_inds = all_indicators[ticker]
        
        for indicator_name, series in ticker_inds.items():
            if date in series.index:
                try:
                    value = series.loc[date]
                    col_name = f"{ticker}_{indicator_name}"
                    row[col_name] = value
                except (KeyError, TypeError):
                    pass
    
    return row


def build_features_dataframe_parallel(data_matrix, all_indicators, n_jobs=-1, show_progress=True):
    """
    Convert all indicators to a wide format DataFrame suitable for saving to parquet.
    Uses parallel processing for efficient computation on large datasets.
    
    Args:
        data_matrix: pandas DataFrame with dates as index and tickers as columns
        all_indicators: dict from calculate_all_indicators or calculate_all_indicators_parallel
        n_jobs: number of processors for parallel processing
               -1 = use all available (default)
               1 = serial processing
               2, 4, 8, etc. = specific number
        show_progress: whether to show progress bar
    
    Returns:
        pandas DataFrame with features as columns and dates as rows
    """
    dates = data_matrix.index.tolist()
    tickers = list(all_indicators.keys())
    
    if show_progress:
        print(f"Building features DataFrame with {len(dates)} dates and {len(tickers)} tickers...")
    
    # Process dates in parallel
    feature_data = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_process_date_features)(date, all_indicators, tickers)
        for date in tqdm(dates, desc="Processing dates", unit="date", disable=not show_progress)
    )
    
    features_df = pd.DataFrame(feature_data)
    
    return features_df


def save_features_to_parquet(features_df, filepath='features.parquet', show_progress=True):
    """
    Save features DataFrame to parquet file.
    
    Args:
        features_df: pandas DataFrame to save
        filepath: output file path
        show_progress: whether to show progress message
    
    Returns:
        None (saves to file)
    """
    if show_progress:
        print(f"Saving features to {filepath}...")
    
    features_df.to_parquet(filepath, index=False)
    
    if show_progress:
        print(f"✓ Successfully saved {len(features_df)} rows with {len(features_df.columns)} features")
        print(f"  File size: {filepath}")
        print(f"  Estimated memory: {features_df.memory_usage(deep=True).sum() / 1e6:.2f} MB")


def transform_indicators_to_dataframes(all_indicators, presence_df, show_progress=True):
    """
    Transform all_indicators from ticker->indicators format to indicators->dataframe format.
    
    Converts from:
        {ticker: {indicator_name: Series, ...}, ...}
    To:
        {indicator_name: DataFrame(dates x tickers), ...}
    
    Args:
        all_indicators: dict with structure {ticker: {indicator: Series, ...}, ...}
        presence_df: reference DataFrame with dates as index and tickers as columns
                    used to establish the shape (dates x tickers)
        show_progress: whether to show progress bar
    
    Returns:
        dict with structure {indicator_name: DataFrame(dates x tickers), ...}
    """
    if not all_indicators:
        return {}
    
    # Get all unique indicator names
    sample_ticker = next(iter(all_indicators.keys()))
    indicator_names = list(all_indicators[sample_ticker].keys())
    
    # Transform to indicator-based structure
    indicators_dfs = {}
    
    iterator = tqdm(indicator_names, desc="Transforming indicators", unit="indicator") if show_progress else indicator_names
    
    for indicator_name in iterator:
        # Create DataFrame for this indicator with dates as index and tickers as columns
        data_dict = {}
        
        for ticker in all_indicators.keys():
            indicator_series = all_indicators[ticker].get(indicator_name)
            if indicator_series is not None:
                data_dict[ticker] = indicator_series
        
        # Create DataFrame from dict (automatically aligns by index)
        indicator_df = pd.DataFrame(data_dict)
        
        # Reindex to match presence_df structure (for consistency)
        indicator_df = indicator_df.reindex(index=presence_df.index, columns=presence_df.columns)
        
        indicators_dfs[indicator_name] = indicator_df
    
    return indicators_dfs


def save_all_indicators_to_parquet(all_indicators, directory='stores/indicators', show_progress=True):
    """
    Save all indicators to parquet files.
    
    Each indicator is saved as a separate parquet file:
        directory/indicator_name.parquet
    
    Args:
        all_indicators: dict mapping indicator names to DataFrames
                       Structure: {indicator_name: DataFrame(dates x tickers), ...}
        directory: output directory path (will be created if doesn't exist)
        show_progress: whether to show progress with tqdm
    
    Returns:
        dict with save statistics {indicator_name: file_path}
    """
    import os
    
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        if show_progress:
            print(f"Created directory: {directory}")
    
    saved_files = {}
    
    iterator = tqdm(all_indicators.items(), desc="Saving indicators", unit="indicator") if show_progress else all_indicators.items()
    
    for indicator_name, df in iterator:
        # Create safe filename
        safe_name = indicator_name.replace('/', '_').replace('\\', '_')
        filepath = os.path.join(directory, f"{safe_name}.parquet")
        
        # Save to parquet
        df.to_parquet(filepath)
        saved_files[indicator_name] = filepath
    
    if show_progress:
        total_size = sum(os.path.getsize(fp) for fp in saved_files.values()) / 1e6
        print(f"\n✓ Successfully saved {len(saved_files)} indicators")
        print(f"  Directory: {directory}")
        print(f"  Total size: {total_size:.2f} MB")
        print(f"\nSaved files:")
        for i, (indicator_name, filepath) in enumerate(sorted(saved_files.items()), 1):
            size_mb = os.path.getsize(filepath) / 1e6
            print(f"  {i:2d}. {indicator_name:40s} - {size_mb:8.2f} MB")
    
    return saved_files


def load_indicator_from_parquet(indicator_name, directory='stores/indicators'):
    """
    Load a single indicator from parquet file.
    
    Args:
        indicator_name: name of the indicator to load
        directory: directory where indicator files are stored
    
    Returns:
        pandas DataFrame with the indicator data
    """
    import os
    
    safe_name = indicator_name.replace('/', '_').replace('\\', '_')
    filepath = os.path.join(directory, f"{safe_name}.parquet")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Indicator file not found: {filepath}")
    
    return pd.read_parquet(filepath)

