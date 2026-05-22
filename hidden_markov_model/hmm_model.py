"""Frozen 3-state Gaussian HMM regime detector for SPY.

Fit once with `python hmm_model.py fit`, then call `load_regime_scalar(dates)`
from the master pipeline to get the daily HMM_cont = P(bull) + 0.5 * P(chop).
"""
from __future__ import annotations
import os, pickle
import numpy as np
import pandas as pd
from scipy.special import logsumexp

PARAMS_PATH = os.path.join(os.path.dirname(__file__), 'hmm_params.pkl')


def _log_gauss(X, mu, sigma):
    diff = X[:, None, :] - mu[None, :, :]
    inv_var = 1.0 / (sigma ** 2)
    log_norm = -0.5 * (X.shape[1] * np.log(2 * np.pi) + np.sum(2 * np.log(sigma), axis=1))
    quad = -0.5 * np.sum((diff ** 2) * inv_var[None, :, :], axis=2)
    return log_norm[None, :] + quad


def _forward(log_pi, log_A, log_B):
    T, K = log_B.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        log_alpha[t] = log_B[t] + logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)
    return log_alpha


def _backward(log_A, log_B):
    T, K = log_B.shape
    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(log_A + log_B[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
    return log_beta


class GaussianHMM:
    def __init__(self, n_states=3, n_iter=80, tol=1e-4, min_sigma=1e-4, seed=0):
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.min_sigma = min_sigma
        self.seed = seed

    def _init(self, X):
        order = np.argsort(X[:, 0])
        chunks = np.array_split(order, self.K)
        self.mu = np.stack([X[c].mean(axis=0) for c in chunks])
        self.sigma = np.stack([X[c].std(axis=0) + self.min_sigma for c in chunks])
        self.pi = np.full(self.K, 1.0 / self.K)
        self.A = np.full((self.K, self.K), 0.05 / (self.K - 1))
        np.fill_diagonal(self.A, 0.95)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self._init(X)
        prev_ll = -np.inf
        for it in range(self.n_iter):
            log_pi, log_A = np.log(self.pi + 1e-300), np.log(self.A + 1e-300)
            log_B = _log_gauss(X, self.mu, self.sigma)
            log_alpha = _forward(log_pi, log_A, log_B)
            log_beta = _backward(log_A, log_B)
            ll = logsumexp(log_alpha[-1])
            log_gamma = log_alpha + log_beta - ll
            gamma = np.exp(log_gamma)
            log_xi = (log_alpha[:-1, :, None] + log_A[None, :, :]
                      + log_B[1:, None, :] + log_beta[1:, None, :]) - ll
            xi = np.exp(log_xi)
            self.pi = gamma[0] / gamma[0].sum()
            self.A = xi.sum(axis=0) / xi.sum(axis=(0, 2))[:, None]
            Nk = gamma.sum(axis=0)[:, None]
            self.mu = (gamma.T @ X) / Nk
            diff2 = (X[:, None, :] - self.mu[None, :, :]) ** 2
            self.sigma = np.sqrt((gamma[:, :, None] * diff2).sum(axis=0) / Nk) + self.min_sigma
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        self.log_likelihood_ = ll
        self.n_iter_ = it + 1
        return self

    def filter(self, X):
        X = np.asarray(X, dtype=float)
        log_pi, log_A = np.log(self.pi + 1e-300), np.log(self.A + 1e-300)
        log_B = _log_gauss(X, self.mu, self.sigma)
        log_alpha = _forward(log_pi, log_A, log_B)
        return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))


# -----------------------------------------------------------------------------
# Feature pipeline (must match what was used at fit time)
# -----------------------------------------------------------------------------
def build_features(spy_adj_close: pd.Series) -> pd.DataFrame:
    """Two-channel feature: daily log return + 20-day realized vol of log returns."""
    px = spy_adj_close.astype(float).sort_index()
    ret = np.log(px / px.shift(1))
    rv = ret.rolling(20).std()
    return pd.concat({'ret': ret, 'rv': rv}, axis=1).dropna()


# -----------------------------------------------------------------------------
# Fit-and-save (run once)
# -----------------------------------------------------------------------------
def fit_and_save(spy_adj_close: pd.Series, out_path: str = PARAMS_PATH) -> dict:
    feats = build_features(spy_adj_close)
    X = feats.values
    mu_s, sd_s = X.mean(axis=0), X.std(axis=0)
    Xs = (X - mu_s) / sd_s

    hmm = GaussianHMM(n_states=3, n_iter=120, seed=0).fit(Xs)

    # Label states by annualized return so 'bull' always means the high-return state
    mu_orig = hmm.mu * sd_s + mu_s            # un-standardize the means
    ann_ret = mu_orig[:, 0] * 252
    order = np.argsort(ann_ret)               # ascending: low -> high
    state_labels = {int(order[0]): 'bear', int(order[1]): 'chop', int(order[2]): 'bull'}

    params = {
        'pi': hmm.pi, 'A': hmm.A, 'mu': hmm.mu, 'sigma': hmm.sigma,
        'feat_mean': mu_s, 'feat_std': sd_s,
        'state_labels': state_labels,
        'fit_log_likelihood': hmm.log_likelihood_,
        'fit_n_iter': hmm.n_iter_,
        'fit_T': len(feats),
        'fit_last_date': str(feats.index[-1].date()),
    }
    with open(out_path, 'wb') as f:
        pickle.dump(params, f)
    print(f"Saved HMM params to {out_path}")
    print(f"  T={params['fit_T']}, iters={params['fit_n_iter']}, ll={params['fit_log_likelihood']:.1f}")
    print(f"  state -> label: {state_labels}")
    print(f"  ann_ret per state: {dict(zip(range(3), (mu_orig[:,0]*252).round(3)))}")
    return params


# -----------------------------------------------------------------------------
# Inference (called from the master pipeline)
# -----------------------------------------------------------------------------
def load_params(path: str = PARAMS_PATH) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_regime_probs(spy_adj_close: pd.Series, params: dict | None = None) -> pd.DataFrame:
    """Return daily filtered P(bull), P(chop), P(bear) using frozen params."""
    if params is None:
        params = load_params()
    feats = build_features(spy_adj_close)
    Xs = (feats.values - params['feat_mean']) / params['feat_std']

    hmm = GaussianHMM(n_states=3)
    hmm.pi = params['pi']
    hmm.A = params['A']
    hmm.mu = params['mu']
    hmm.sigma = params['sigma']

    alpha = hmm.filter(Xs)
    cols = [params['state_labels'][k] for k in range(3)]
    return pd.DataFrame(alpha, index=feats.index, columns=cols)[['bull', 'chop', 'bear']]


def compute_regime_scalar(spy_adj_close: pd.Series,
                          chop_weight: float = 0.5,
                          params: dict | None = None) -> pd.Series:
    """HMM_cont = P(bull) + chop_weight * P(chop). Returns a daily series."""
    probs = compute_regime_probs(spy_adj_close, params=params)
    return (probs['bull'] + chop_weight * probs['chop']).rename('regime_scalar')


if __name__ == '__main__':
    import sys
    import yfinance as yf
    if len(sys.argv) > 1 and sys.argv[1] == 'fit':
        cache = os.path.join(os.path.dirname(__file__), 'spy_cache.parquet')
        if os.path.exists(cache):
            spy = pd.read_parquet(cache)
        else:
            spy = yf.download('SPY', start='2005-01-01', auto_adjust=False, progress=False)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy.to_parquet(cache)
        px = spy['Adj Close']
        if hasattr(px.index, 'tz') and px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        fit_and_save(px)
    else:
        print("Usage: python hmm_model.py fit")
