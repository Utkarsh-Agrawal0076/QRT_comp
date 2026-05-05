# Phase 2 - QRT Quant Challenge India 2026

Small collection of notebooks and helper scripts used for Phase 2 of the QRT Quant Challenge India 2026.

## Repository layout

- `top_5000_us_by_marketcap.csv` - universe of US tickers by market cap used for experiments.
- `requirements.txt` - Python dependencies for running the notebooks and scripts.
- `scripts/` - main analysis notebooks and helper modules:
	- `1_create_universe_and_returns.ipynb` — build the trading universe and compute returns.
	- `2_create_features.ipynb` — compute technical/feature sets from price data.
	- `3_testing_features.ipynb` — test and validate feature predictive power.
	- `4_creating_portfolios.ipynb` — construct test portfolios and evaluate performance.
	- `technical_indicators.py` — implementations of indicator/feature functions.
	- `utils.py` — utility helpers used by the notebooks.
- `stores/` - stored artifacts (also contains `store_links.txt` for the in-person workshop).

## Quick start

1. Google Colab setup (recommended):

- In a Colab Notebook, create 'New Notebook' and select 'GitHub' as the option. Authenticate with your Github and load this repo. Finally, select the notebook you wish to run.


1. For local setup, open the notebooks with Jupyter Lab / Notebook and run them in order (1 → 2 → 3 → 4). The notebooks are written to be executed sequentially so outputs from earlier notebooks are available to later ones.

## Data

- The repository includes `top_5000_us_by_marketcap.csv` which lists the tickers used to build the universe.
- Downloaded price data and persistent artifacts (parquet files) produced by the notebooks are expected to be stored in the `stores/` folder (or another data folder configured in the notebooks).

## Notes

- Notebooks contain the main analysis and are the recommended entrypoint for exploration.
- If you modify or re-run data-download steps, re-run notebooks from `1_create_universe_and_returns.ipynb` to ensure later notebooks see updated inputs.

## Requirements

All Python dependencies are listed in `requirements.txt`.

## Contact

Contact the repository owner for permission and questions. Redistribution is strictly prohibited.