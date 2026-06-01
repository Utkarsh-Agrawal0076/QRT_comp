"""
Build the input panel consumed by the 101 alpha functions.

All fields are aligned to a single (date x ticker) grid taken from the
competition universe/returns files so that signals, returns and the universe
mask line up exactly when the pipeline evaluates them.

Field sources
-------------
open/high/low/close/volume : top_5000_yf_data.pkl  (daily OHLCV)
vwap                       : surrogate typical price (high+low+close)/3
                             (no intraday data; same surrogate used elsewhere
                             in this repo, e.g. alpha_pipeline.py)
returns                    : stores/returns.parquet (official close-to-close),
                             ±inf cleaned to NaN
adv{n}                     : (close * volume) rolled n days (avg daily $ volume)
cap                        : static market cap from the marketcap CSV (proxy)
sector / industry          : ticker->group from the marketcap CSV
subindustry                : falls back to `industry` (no finer classification)
"""

import os
import numpy as np
import pandas as pd

DATA_PICKLE = "top_5000_yf_data.pkl"
RETURNS_PARQUET = os.path.join("stores", "returns.parquet")
UNIVERSE_PARQUET = os.path.join("stores", "universe_5m.parquet")
MARKETCAP_CSV = os.path.join("phase2_qrt_challenge", "top_5000_us_by_marketcap.csv")


class Data:
    """Container of aligned OHLCV/vwap/returns panels + adv cache and groups."""

    def __init__(self, open, high, low, close, volume, vwap, returns,
                 cap, sector, industry, subindustry):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.vwap = vwap
        self.returns = returns
        self.cap = cap
        self.sector = sector
        self.industry = industry
        self.subindustry = subindustry
        self._dollar_vol = close * volume
        self._adv_cache = {}

    def adv(self, n):
        """Average daily dollar volume over the past n days (cached)."""
        n = int(n)
        if n not in self._adv_cache:
            self._adv_cache[n] = self._dollar_vol.rolling(n, min_periods=n).mean()
        return self._adv_cache[n]


def load_panel(start=None, end=None, base_dir=".", verbose=True):
    """
    Load and align every input field. `start`/`end` optionally clip the date
    range (strings 'YYYY-MM-DD') to bound compute.
    """
    def p(msg):
        if verbose:
            print(msg)

    p("Loading OHLCV pickle...")
    ohlcv = pd.read_pickle(os.path.join(base_dir, DATA_PICKLE))

    p("Loading returns / universe...")
    returns = pd.read_parquet(os.path.join(base_dir, RETURNS_PARQUET))
    universe = pd.read_parquet(os.path.join(base_dir, UNIVERSE_PARQUET))

    # canonical grid = universe grid (dates x tickers)
    dates = universe.index
    tickers = universe.columns

    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]

    def field(name):
        f = ohlcv[name]
        f = f.loc[:, ~f.columns.duplicated()]
        return f.reindex(index=dates, columns=tickers)

    open_ = field("Open")
    high = field("High")
    low = field("Low")
    close = field("Close")
    volume = field("Volume")
    vwap = (high + low + close) / 3.0

    returns = returns.reindex(index=dates, columns=tickers)
    returns = returns.replace([np.inf, -np.inf], np.nan)

    p("Loading market cap / industry classification...")
    cap, sector, industry = _load_static(os.path.join(base_dir, MARKETCAP_CSV), tickers)
    subindustry = industry  # no finer classification available

    p(f"Panel ready: {len(dates)} dates x {len(tickers)} tickers "
      f"({dates[0].date()} -> {dates[-1].date()})")

    data = Data(open_, high, low, close, volume, vwap, returns,
                cap, sector, industry, subindustry)
    return data, returns, universe.reindex(index=dates, columns=tickers)


def _load_static(csv_path, tickers):
    """Static per-ticker market cap (Series) and sector/industry group maps."""
    df = pd.read_csv(csv_path)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df = df.drop_duplicates(subset="symbol").set_index("symbol")

    # ticker-indexed Series; `returns * cap` broadcasts it across dates (the
    # Series index aligns to the DataFrame columns), which is what Alpha#56 wants.
    cap_series = pd.to_numeric(df["marketCap"], errors="coerce").reindex(tickers)
    sector = df["sector"].reindex(tickers)
    industry = df["industry"].reindex(tickers)
    return cap_series, sector, industry
