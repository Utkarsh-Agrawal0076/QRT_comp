"""
ric_resolver.py
Robust ticker -> RIC resolution for QRT submissions.

Resolution order for each ticker:
  1. Use ric_exchange_map.csv (authoritative, QRT-report-derived + yfinance-augmented).
  2. If absent and allow_fetch: query yfinance listing exchange, derive the RIC
     suffix, and CACHE the result back into the map so it is consistent next time.
  3. If still unresolved: DO NOT silently default to .OQ. Return it in `unresolved`
     so the caller can drop it loudly (an invalid RIC is silently rejected by QRT,
     which is exactly the failure this module exists to prevent).
"""
from pathlib import Path
import pandas as pd

# Listing-exchange code -> RIC suffix. Validated against the QRT-report-derived map.
EXCH_TO_SUFFIX = {
    "NMS": ".OQ", "NGM": ".OQ", "NCM": ".OQ", "NSM": ".OQ", "NGS": ".OQ", "NAS": ".OQ",
    "NYQ": ".N",  "NYS": ".N",
    "ASE": ".A",  "AMX": ".A",
    "PCX": ".P",  "ARC": ".P",
    "BTS": ".Z",  "BATS": ".Z",
}


def _load_map(ric_map_csv):
    if not Path(ric_map_csv).exists():
        return {}
    m = pd.read_csv(ric_map_csv, index_col=0)["ric"].to_dict()
    return {str(k): v for k, v in m.items() if isinstance(v, str)}


def _save_map(mapping, ric_map_csv):
    out = pd.DataFrame({"ric": pd.Series(mapping)})
    out.to_csv(ric_map_csv)


def _fetch_suffix(ticker):
    """Return RIC suffix for a ticker via yfinance listing exchange, or None."""
    try:
        import yfinance as yf
        ex = getattr(yf.Ticker(ticker).fast_info, "exchange", None)
    except Exception:
        return None, None
    return EXCH_TO_SUFFIX.get(ex), ex


def resolve(tickers, ric_map_csv="ric_exchange_map.csv", allow_fetch=True, verbose=True):
    """Map an iterable of base tickers to RICs.

    Returns (codes, unresolved):
      codes: dict ticker -> RIC for every resolvable ticker
      unresolved: list of tickers with no valid RIC (caller should exclude these)
    Newly fetched RICs are persisted back to ric_map_csv.
    """
    mapping = _load_map(ric_map_csv)
    codes, unresolved, newly_fetched = {}, [], 0

    for t in tickers:
        ric = mapping.get(str(t))
        if isinstance(ric, str) and "." in ric:
            codes[t] = ric
            continue
        if allow_fetch:
            suf, ex = _fetch_suffix(t)
            if suf is not None:
                ric = f"{t}{suf}"
                mapping[str(t)] = ric
                codes[t] = ric
                newly_fetched += 1
                continue
            if verbose:
                print(f"    UNRESOLVED {t}: yfinance exchange={ex!r} has no known RIC suffix")
        unresolved.append(t)

    if newly_fetched:
        _save_map(mapping, ric_map_csv)
        if verbose:
            print(f"  resolved {newly_fetched} new ticker(s) via yfinance and cached to {ric_map_csv}")
    if unresolved and verbose:
        print(f"  WARNING: {len(unresolved)} ticker(s) unresolved and will be EXCLUDED: {unresolved[:20]}"
              + (" ..." if len(unresolved) > 20 else ""))
    return codes, unresolved


if __name__ == "__main__":
    # quick self-test
    c, u = resolve(["KEY", "ZION", "EG", "SPY"], allow_fetch=True)
    print("codes:", c)
    print("unresolved:", u)
