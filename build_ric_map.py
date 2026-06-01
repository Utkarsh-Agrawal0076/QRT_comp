"""
build_ric_map.py
Build a robust RIC exchange map by fetching each ticker's listing exchange from
yfinance and learning QRT's exchange->RIC-suffix convention from names already
correctly mapped in ric_exchange_map.csv (ground truth derived from QRT reports).

Output: rebuilt ric_exchange_map.csv (old one backed up), plus a diagnostics CSV.
"""
import sys, time, json
from pathlib import Path
import pandas as pd
import yfinance as yf

RIC_MAP_CSV = "ric_exchange_map.csv"
SUB_CSV = "qrt_academy_IND22_20260528-1809.csv"
EXCH_CACHE = "ticker_exchange_cache.csv"   # raw yfinance exchange per ticker (resumable)

# Static fallback exchange-code -> RIC suffix (used only if a code is unseen in ground truth)
STATIC_EXCH_TO_SUFFIX = {
    "NMS": ".OQ", "NGM": ".OQ", "NCM": ".OQ", "NSM": ".OQ", "NGS": ".OQ", "NAS": ".OQ",
    "NYQ": ".N", "NYS": ".N",
    "ASE": ".A", "AMX": ".A",
    "PCX": ".P", "ARC": ".P",
    "BTS": ".Z", "BATS": ".Z",
}


def fetch_exchanges(tickers):
    """Return dict ticker -> exchange code, resuming from cache if present."""
    cache = {}
    if Path(EXCH_CACHE).exists():
        c = pd.read_csv(EXCH_CACHE)
        cache = dict(zip(c["ticker"].astype(str), c["exchange"].astype(str)))
        print(f"  resume: {len(cache)} cached exchanges loaded")

    todo = [t for t in tickers if t not in cache]
    print(f"  need to fetch {len(todo)} of {len(tickers)} tickers")
    CHUNK = 200
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        ts = yf.Tickers(" ".join(chunk))
        for s in chunk:
            try:
                ex = getattr(ts.tickers[s].fast_info, "exchange", None)
            except Exception:
                ex = None
            cache[s] = ex if ex else "UNKNOWN"
        # persist after each chunk (resumable)
        pd.DataFrame({"ticker": list(cache), "exchange": list(cache.values())}).to_csv(EXCH_CACHE, index=False)
        print(f"    fetched {min(i+CHUNK, len(todo))}/{len(todo)} (last chunk {chunk[0]}..{chunk[-1]})", flush=True)
    return cache


def main():
    sub = pd.read_csv(SUB_CSV)
    sub["internal_code"] = sub["internal_code"].str.strip()
    sub["base"] = sub["internal_code"].str.replace(r"\.[A-Z]+$", "", regex=True)

    old_map = pd.read_csv(RIC_MAP_CSV, index_col=0)["ric"].to_dict()
    # ground-truth suffix per base ticker (from existing QRT-derived map)
    gt_suffix = {}
    for base, ric in old_map.items():
        if isinstance(ric, str) and "." in ric and str(base) != "nan":
            gt_suffix[str(base)] = "." + ric.split(".")[-1]

    base_tickers = sorted(sub["base"].unique())
    print(f"Submission unique base tickers: {len(base_tickers)}")

    print("Fetching exchanges from yfinance...")
    exch = fetch_exchanges(base_tickers)

    # ---- Learn exchange-code -> suffix from ground truth ----
    learn_rows = []
    for base, suf in gt_suffix.items():
        if base in exch:
            learn_rows.append({"exchange": exch[base], "suffix": suf})
    learn = pd.DataFrame(learn_rows)
    learned = {}
    if len(learn):
        # for each exchange code, dominant suffix among ground-truth names
        tab = learn.groupby("exchange")["suffix"].agg(lambda s: s.value_counts())
        ct = learn.value_counts(["exchange", "suffix"]).reset_index(name="n")
        dom = ct.sort_values("n", ascending=False).drop_duplicates("exchange")
        learned = dict(zip(dom["exchange"], dom["suffix"]))
    print("\nLearned exchange -> suffix (from correctly-mapped names):")
    for ex_code in sorted(set(list(learned) + list(STATIC_EXCH_TO_SUFFIX))):
        ln = learned.get(ex_code); st = STATIC_EXCH_TO_SUFFIX.get(ex_code)
        n = int((learn["exchange"] == ex_code).sum()) if len(learn) else 0
        print(f"  {str(ex_code):10} learned={ln} (n={n})  static={st}")

    def suffix_for(ex_code):
        if ex_code in learned:
            return learned[ex_code]
        return STATIC_EXCH_TO_SUFFIX.get(ex_code, None)

    # ---- Validate: derived suffix vs ground-truth suffix on mapped names ----
    conflicts = []
    agree = 0
    for base, gts in gt_suffix.items():
        if base not in exch:
            continue
        ds = suffix_for(exch[base])
        if ds is None:
            continue
        if ds == gts:
            agree += 1
        else:
            conflicts.append({"ticker": base, "exchange": exch[base],
                              "ground_truth": gts, "derived": ds})
    print(f"\nValidation on ground-truth names: {agree} agree, {len(conflicts)} conflict")
    if conflicts:
        print("  conflicts (ground truth wins, kept as-is):")
        for c in conflicts[:25]:
            print(f"    {c['ticker']:8} ex={c['exchange']:6} gt={c['ground_truth']} derived={c['derived']}")

    # ---- Build the complete map ----
    new_map = dict(old_map)              # keep all existing (ground truth) entries
    unresolved = []
    for base in base_tickers:
        if base in old_map and isinstance(old_map[base], str) and "." in str(old_map[base]):
            continue  # already correctly mapped — trust ground truth
        ex_code = exch.get(base, "UNKNOWN")
        suf = suffix_for(ex_code)
        if suf is None:
            unresolved.append({"ticker": base, "exchange": ex_code})
            continue
        new_map[base] = f"{base}{suf}"

    print(f"\nUnresolved (unknown exchange code, NOT added — generator will warn): {len(unresolved)}")
    for u in unresolved[:30]:
        print(f"    {u['ticker']:8} exchange={u['exchange']}")

    # backup + write
    Path(RIC_MAP_CSV).replace(RIC_MAP_CSV + ".bak")
    out = pd.DataFrame({"ric": pd.Series(new_map)})
    out.index.name = None
    out.to_csv(RIC_MAP_CSV)
    print(f"\n[wrote] {RIC_MAP_CSV}: {len(new_map)} entries (old backed up to {RIC_MAP_CSV}.bak)")

    # coverage report for the current submission
    cov = sum(1 for b in base_tickers if b in new_map and isinstance(new_map[b], str) and "." in str(new_map[b]))
    print(f"  submission base tickers now mapped: {cov}/{len(base_tickers)}")
    pd.DataFrame(unresolved).to_csv("ric_unresolved.csv", index=False)
    print("[wrote] ric_unresolved.csv")


if __name__ == "__main__":
    main()
