"""
Vectorized implementations of the operators used by the 101 Formulaic Alphas
(Kakushadze 2015, arXiv:1601.00991).

Every operator works on a pandas DataFrame whose index is the trading date and
whose columns are tickers (a "panel"). Scalars broadcast normally. This keeps
each alpha a one-to-one transcription of the paper's formula while running
cross-sectionally / time-serially across the whole universe at once.

Conventions
-----------
- rank(x)           : cross-sectional percentile rank in [0, 1] (per row / day).
- ts_*(x, d)        : time-series operator over the trailing d days (per column).
- correlation/cov   : rolling time-series statistic computed with the moment
                      formula (vectorised across all columns simultaneously).
- d is floored to an int as the paper specifies (many windows are non-integer).
"""

import numpy as np
import pandas as pd
import bottleneck as bn


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _d(d):
    """Floor a (possibly float) window length to a positive int, per the paper."""
    return max(int(np.floor(d)), 1)


def _as_float(x):
    """Cast booleans (from comparisons) to float so they compose arithmetically."""
    if isinstance(x, pd.DataFrame):
        if x.dtypes.eq(bool).any():
            return x.astype(float)
    return x


# --------------------------------------------------------------------------- #
#  Elementwise math
# --------------------------------------------------------------------------- #
def abs_(x):
    return np.abs(x)


def log(x):
    # guard against log of non-positive values
    return np.log(x.where(x > 0)) if isinstance(x, pd.DataFrame) else np.log(x)


def sign(x):
    return np.sign(x)


def signedpower(x, a):
    """x^a but preserving sign: sign(x) * |x|^a (community-standard reading)."""
    return np.sign(x) * (np.abs(x) ** a)


def power(x, a):
    """The '^' operator as written in the paper (plain exponentiation)."""
    return signedpower(x, a)


# --------------------------------------------------------------------------- #
#  Cross-sectional operators (operate per row / day)
# --------------------------------------------------------------------------- #
def rank(x):
    """Cross-sectional percentile rank in [0, 1]."""
    x = _as_float(x)
    return x.rank(axis=1, pct=True)


def scale(x, a=1.0):
    """Rescale each row so that sum(|x|) == a."""
    x = _as_float(x)
    norm = x.abs().sum(axis=1).replace(0, np.nan)
    return x.mul(a).div(norm, axis=0)


def indneutralize(x, groups):
    """
    Cross-sectionally demean x within each group g.

    `groups` is a pandas Series mapping ticker -> group label (sector / industry).
    Tickers with no group are demeaned against the global cross-section. If
    `groups` is None the operator is the identity (used when no industry data is
    available — those alphas are flagged 'approximate' in the report).
    """
    if groups is None:
        return x
    x = _as_float(x)
    g = groups.reindex(x.columns)
    g = g.fillna("__UNKNOWN__")
    # group means per day: transpose so groups live on the row axis
    grp_mean = x.T.groupby(g).transform("mean").T
    return x - grp_mean


# --------------------------------------------------------------------------- #
#  Time-series operators (operate per column over a trailing window)
# --------------------------------------------------------------------------- #
def delay(x, d):
    return x.shift(_d(d))


def delta(x, d):
    return x - x.shift(_d(d))


def ts_sum(x, d):
    return x.rolling(_d(d), min_periods=_d(d)).sum()


# alias used as plain `sum(x, d)` in the paper
sum_ = ts_sum


def product(x, d):
    """
    Rolling product over d days. Implemented as exp(rolling-sum(log|x|)) * sign,
    which is fully vectorised. In the paper `product` is only ever applied to
    ranks (strictly positive), so the sign term is a no-op there.
    """
    d = _d(d)
    x = _as_float(x)
    sign_prod = np.sign(x).rolling(d, min_periods=d).apply(np.prod, raw=True) \
        if (x.values < 0).any() else 1.0
    log_abs = np.log(x.abs().replace(0, np.nan))
    mag = np.exp(log_abs.rolling(d, min_periods=d).sum())
    return mag * sign_prod


def stddev(x, d):
    d = _d(d)
    return x.rolling(d, min_periods=d).std()


def ts_min(x, d):
    d = _d(d)
    return x.rolling(d, min_periods=d).min()


def ts_max(x, d):
    d = _d(d)
    return x.rolling(d, min_periods=d).max()


# `min`/`max` in the paper mean ts_min/ts_max
def min_(x, d):
    return ts_min(x, d)


def max_(x, d):
    return ts_max(x, d)


def ts_rank(x, d):
    """
    Rolling rank of the most recent value within the trailing d-day window,
    normalised to [0, 1]. Uses bottleneck.move_rank (returns [-1, 1]) for speed.
    """
    d = _d(d)
    arr = bn.move_rank(np.asarray(x, dtype=float), window=d, axis=0)
    out = (arr + 1.0) / 2.0  # map [-1, 1] -> [0, 1]
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _ts_arg(x, d, want_max):
    """
    Position (0 = oldest, d-1 = most recent) within the trailing window at which
    the rolling max (want_max=True) or min occurred. Matches np.argmax/argmin
    over the window ordered oldest->newest. O(d) vectorised frame ops.
    """
    d = _d(d)
    extreme = ts_max(x, d) if want_max else ts_min(x, d)
    pos = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    # iterate oldest (shift d-1) -> newest (shift 0); keep first/oldest extreme
    found = pd.DataFrame(False, index=x.index, columns=x.columns)
    for k in range(d - 1, -1, -1):
        cand = x.shift(k)
        is_ext = (cand == extreme) & (~found)
        position = (d - 1 - k)  # oldest k=d-1 -> 0
        pos = pos.mask(is_ext, position)
        found = found | is_ext
    # only valid where the window is full
    valid = extreme.notna()
    return pos.where(valid)


def ts_argmax(x, d):
    return _ts_arg(x, d, want_max=True)


def ts_argmin(x, d):
    return _ts_arg(x, d, want_max=False)


def _rolling_moment(x, y, d, cov=False):
    """Rolling correlation (cov=False) or covariance (cov=True) of two panels."""
    d = _d(d)
    x = _as_float(x)
    y = _as_float(y)
    # align
    x, y = x.align(y, join="inner")
    sx = x.rolling(d, min_periods=d).sum()
    sy = y.rolling(d, min_periods=d).sum()
    sxy = (x * y).rolling(d, min_periods=d).sum()
    sxx = (x * x).rolling(d, min_periods=d).sum()
    syy = (y * y).rolling(d, min_periods=d).sum()
    cov_xy = (sxy - sx * sy / d) / (d - 1)
    if cov:
        return cov_xy
    var_x = (sxx - sx * sx / d) / (d - 1)
    var_y = (syy - sy * sy / d) / (d - 1)
    denom = np.sqrt(var_x * var_y)
    denom = denom.replace(0, np.nan)
    return cov_xy / denom


def correlation(x, y, d):
    return _rolling_moment(x, y, d, cov=False)


def covariance(x, y, d):
    return _rolling_moment(x, y, d, cov=True)


def decay_linear(x, d):
    """
    Weighted moving average over the trailing d days with linearly decaying
    weights (d, d-1, ..., 1) rescaled to sum to 1. Vectorised as a weighted sum
    of shifted frames.
    """
    d = _d(d)
    x = _as_float(x)
    weights = np.arange(d, 0, -1, dtype=float)
    weights /= weights.sum()
    out = None
    for i, w in enumerate(weights):
        term = x.shift(i) * w  # i=0 -> most recent gets the largest weight (d)
        out = term if out is None else out + term
    # require a full window
    mask = x.shift(d - 1).notna()
    return out.where(mask)


# --------------------------------------------------------------------------- #
#  Logical / ternary helpers
# --------------------------------------------------------------------------- #
def emax(a, b):
    """Elementwise maximum of two panels (paper's max(x, y) with two expressions)."""
    return np.maximum(a, b)


def emin(a, b):
    """Elementwise minimum of two panels (paper's min(x, y) with two expressions)."""
    return np.minimum(a, b)


def iif(cond, a, b):
    """Vectorised ternary  cond ? a : b  preserving the panel index/columns."""
    if isinstance(cond, pd.DataFrame):
        a_df = a if isinstance(a, pd.DataFrame) else pd.DataFrame(a, index=cond.index, columns=cond.columns)
        b_df = b if isinstance(b, pd.DataFrame) else pd.DataFrame(b, index=cond.index, columns=cond.columns)
        out = a_df.where(cond.astype(bool), b_df)
        return out
    return np.where(cond, a, b)
