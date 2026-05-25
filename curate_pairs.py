"""curate_pairs.py — Manual fundamental review of pairs from find_pairs.py.

Removes pairs that are STATISTICALLY cointegrating but ECONOMICALLY mismatched:
  - Wrong industry classification (metadata error)
  - Same industry bucket but very different business models
  - Negative-beta pairs (usually reflect one-time corporate events, not true cointegration)
  - Massive market-cap mismatch within an industry where size matters

Outputs kalman_universe_config_curated.csv. Each REMOVE has a short rationale
in the inline comments so the decision is auditable.
"""
import pandas as pd

INPUT  = "kalman_universe_config_expanded.csv"
OUTPUT = "kalman_universe_config_curated.csv"

# Pairs to REMOVE. Each is (asset_y, asset_x) — order matches the CSV.
# Rationale grouped by industry.
REMOVE = [
    # Major Banks: KEY (commercial bank) vs NTRS (wealth/trust) — different business
    ("KEY",  "NTRS"),

    # Industrial Machinery: cross-industry matches (chem, healthcare, semis)
    ("NDSN", "NOVT"),  # adhesives vs precision lasers
    ("IR",   "WSO"),   # industrial compressors vs HVAC distribution
    ("NDSN", "TMO"),   # adhesives vs life sciences (12x mcap)
    ("NOVT", "TMO"),   # lasers vs life sciences (44x)
    ("ETN",  "WSO"),   # electrical vs HVAC dist
    ("BMI",  "ETN"),   # meters vs electrical (32x)
    ("ALG",  "AME"),   # mowing equipment vs precision instruments (24x)
    ("ITW",  "LRCX"),  # industrial conglom vs semis
    ("ALG",  "IR"),    # mowing equipment vs industrial
    ("IR",   "ONTO"),  # industrial vs semis
    ("FTV",  "WWD"),   # industrial vs aerospace

    # REITs: heterogeneous bucket — only keep within-subtype pairs
    ("HST",  "SKT"),   # hotels vs outlet malls
    ("LXP",  "SUI"),   # industrial vs manufactured-home REIT
    ("CDP",  "LAMR"),  # office vs billboards
    ("HR",   "SUI"),   # healthcare vs manufactured-home
    ("HST",  "SPG"),   # hotels vs malls
    ("HR",   "SBAC"),  # healthcare vs cell towers
    ("KIM",  "STWD"),  # shopping centers vs mortgage REIT
    ("DX",   "NSA"),   # mortgage REIT vs self-storage
    ("AMH",  "STAG"),  # residential vs industrial
    ("REG",  "RHP"),   # shopping centers vs hotels

    # P&C Insurers: mortgage insurance is its own sub-industry, not P&C
    ("MKL",  "MTG"),   # specialty P&C vs mortgage insurance
    ("CNA",  "MTG"),   # commercial P&C vs mortgage
    ("MKL",  "RDN"),   # P&C vs mortgage
    ("L",    "MTG"),   # holding co vs mortgage
    ("AXS",  "RDN"),   # reinsurance vs mortgage
    ("CNA",  "RDN"),   # P&C vs mortgage

    # Software: highly idiosyncratic — kill cross-segment pairs
    ("DBX",  "DT"),    # storage vs APM
    ("SSNC", "VRNS"),  # financial SW vs data security
    ("ADSK", "SHOP"),  # CAD vs e-commerce
    ("DSGX", "INTU"),  # logistics SW vs tax/accounting (19x)
    ("DSGX", "MSFT"),  # logistics SW vs Microsoft (463x mcap!)
    ("CHKP", "ORCL"),  # security vs database
    ("CHKP", "SAP"),   # security vs ERP
    ("MSFT", "SNPS"),  # general SW vs EDA
    ("CHKP", "PTC"),   # security vs CAD/IoT

    # "Real Estate" — catch-all metadata bucket, mostly mis-classified
    ("BR",   "RELX"),  # financial SW vs information services
    ("BR",   "RBA"),   # financial SW vs auctions
    ("MMS",  "V"),     # govt services vs payments (163x mcap)
    ("INVH", "MSCI"),  # residential REIT vs index provider
    ("ACN",  "FSV"),   # consulting vs real-estate services (20x)
    ("MMS",  "TNET"),  # govt services vs HR services
    ("AKAM", "CSGP"),  # CDN vs real-estate data
    ("ACN",  "MSCI"),  # consulting vs index provider
    ("MMS",  "TRNO"),  # govt services vs REIT

    # Biotech Pharma: negative-beta pairs typically reflect one-time events, not cointegration
    ("NVS",  "PRGO"),  # negative beta, 199x mcap
    ("JNJ",  "TAK"),   # negative beta
    ("TAK",  "TBPH"),  # 68x mcap, small biotech vs big pharma
    ("SNY",  "VTRS"),  # negative beta
    ("JNJ",  "PRGO"),  # negative beta, 395x mcap
    ("GSK",  "TAK"),   # negative beta

    # IBs/Brokers: kill exchange/broker cross-pairs and asset-mgr/exchange mixes
    ("BLK",  "TW"),    # asset mgr vs trading platform
    ("CME",  "SF"),    # exchange vs broker
    ("CME",  "RJF"),   # exchange vs broker

    # Major Chemicals: cross-business
    ("EMN",  "MTX"),   # chem vs minerals (sect mismatch)
    ("BCPC", "LIN"),   # specialty chem vs industrial gases (42x mcap)
    ("IOSP", "LIN"),   # specialty chem vs industrial gases (126x mcap)

    # Trucking: international parcel doesn't pair with US trucking
    ("WERN", "ZTO"),   # US trucking vs Chinese parcel delivery

    # Finance Consumer Services: mortgage servicer / lender don't pair with info services
    ("MCO",  "PFSI"),  # Moody's vs PennyMac mortgage
    ("ARCC", "AXP"),   # BDC vs American Express

    # EDP Services: banking SW doesn't pair with utility or defense IT
    ("JKHY", "NEE"),   # banking SW vs NextEra utility
    ("JKHY", "LDOS"),  # banking SW vs govt IT

    # Diversified Commercial Services: ABM (facilities) isn't ADP/PAYX (payroll)
    ("ABM",  "PAYX"),  # facilities vs payroll
    ("ABM",  "ADP"),   # facilities vs payroll
    ("ALLE", "UPBD"),  # locks vs rent-to-own

    # Marine Transportation: defense shipbuilding isn't bulk shipping
    ("HII",  "SFL"),   # naval defense vs ship leasing
    ("GD",   "SBLK"),  # General Dynamics defense vs bulk shipping

    # Medical Specialities: lab/insurer/distributor cross-pairs
    ("DGX",  "UNH"),   # lab vs health insurer
    ("HSIC", "HUM"),   # dental supplies vs insurer

    # Containers/Packaging
    ("AVY",  "WMS"),   # labels vs drainage pipes

    # Auto Manufacturing
    ("FSS",  "RACE"),  # sirens/safety equipment vs Ferrari (12x mcap)
]

# ---- apply curation ----
df = pd.read_csv(INPUT)
print(f"Loaded {len(df)} pairs from {INPUT}")

remove_set = set(tuple(p) for p in REMOVE)
print(f"REMOVE list: {len(remove_set)} pairs (auditable rationale in script)")

# Build mask
def to_key(r):
    return (r["asset_y"], r["asset_x"])
mask_remove = df.apply(to_key, axis=1).isin(remove_set)
print(f"Matched to REMOVE: {mask_remove.sum()}")

if mask_remove.sum() < len(remove_set):
    # Diagnostic: which REMOVE entries didn't match anything in the CSV
    matched_keys = set(df[mask_remove].apply(to_key, axis=1))
    missing = remove_set - matched_keys
    if missing:
        print(f"WARNING: {len(missing)} REMOVE entries don't match the CSV:")
        for m in missing:
            print(f"  {m}")

curated = df[~mask_remove].reset_index(drop=True)
print(f"\nKEPT: {len(curated)} pairs across {curated['industry'].nunique()} industries")
print()
print("Top 15 industries in curated set:")
print(curated["industry"].value_counts().head(15).to_string())

curated.to_csv(OUTPUT, index=False)
print(f"\nWrote {OUTPUT}")
