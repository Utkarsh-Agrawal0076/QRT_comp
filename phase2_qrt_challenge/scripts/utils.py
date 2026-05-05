import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys
import datetime
from tqdm import tqdm

########################### PLOTTING FUNCTIONS ##############################

def plot_series_with_names(series_list, names=None, title="Time Series Plot", yaxis_title="Value", xaxis_title="Time"):
    if names and len(names)!=len(series_list):
        raise ValueError("Length of 'names' must match the number of series provided")
    fig = go.Figure()

    for i, series in enumerate(series_list):
        series_name = names[i] if names else f"Series {i+1}"
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode='lines',
                name=series_name
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        showlegend=True
    )

    fig.show()

def plot_series(*series_list, title="Time Series Plot", yaxis_title="Value", xaxis_title="Time"):
    fig = go.Figure()

    for i, series in enumerate(series_list):
        series_name = f"Series {i+1}"
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode='lines',
                name=series_name
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        showlegend=True
    )

    fig.show()

def plot_dataframe(df, title="DataFrame Plot", yaxis_title="Value", xaxis_title="Time"):
    fig = go.Figure()

    for column in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[column],
                mode='lines',
                name=column
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        showlegend=True
    )

    fig.show()

def plot_series_bar(series, title="Series Bar Plot", yaxis_title="Value", xaxis_title="Index"):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=series.index,
            y=series.values,
            mode='lines',
            name=series.name if series.name else "Series"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        showlegend=True
    )

    fig.show()


################################ PORTFOLIO UTILITY FUNCTIONS ###############################

def scale_weights_to_one(input_series: pd.Series) -> pd.Series:
    """
    Demeans the series, divides by the absolute sum of the series values,
    and retains NaN values.

    Parameters:
    -----------
    input_series : pd.Series
        The input pandas Series.

    Returns:
    --------
    pd.Series
        The processed Series.
    """
    # Demean the series
    demeaned = input_series - input_series.mean()

    # Calculate the absolute sum (ignoring NaN)
    abs_sum = demeaned.abs().sum()

    # Divide by the absolute sum, keeping NaN values
    result = demeaned / abs_sum if abs_sum != 0 else demeaned

    return result

def get_universe_adjusted_series(weights: pd.Series, universe: pd.Series) -> pd.Series:
    """
    Adjusts the weights series based on the universe series by setting weights 
    to NaN where the universe is 0. Ensures both inputs have the same index.

    Parameters:
    -----------
    weights : pd.Series
        A pandas Series representing the weights.

    universe : pd.Series
        A pandas Series representing the universe, where 1 indicates inclusion 
        in the universe and 0 indicates exclusion.

    Returns:
    --------
    pd.Series
        A Series with weights set to NaN where universe is 0.

    Raises:
    -------
    ValueError
        If the indices of weights and universe do not match.
    """
    # Ensure the indices match
    if not weights.index.equals(universe.index):
        raise ValueError("Indices of weights and universe must match.")
    
    # Set weights to NaN where universe is 0
    adjusted_series = weights.where(universe != 0, other=np.nan)
    
    return adjusted_series

def scale_to_book_long_short(alpha: pd.Series) -> pd.Series:
    """
    Scales a given alpha series to maintain a balanced long-short position,
    ensuring that the sum of positive values scales to 0.5 and the sum of negative values scales to -0.5.

    Parameters:
    alpha (pd.Series): A Pandas Series containing alpha values (signals) for trading.

    Returns:
    pd.Series: A scaled alpha series where long positions sum to 0.5 and short positions sum to -0.5.
    """
    # Sum of positive and negative values separately
    sum_pos = alpha[alpha > 0].sum()
    sum_neg = alpha[alpha < 0].sum()

    # Scale positive and negative values to ensure balanced exposure
    df_pos = alpha[alpha > 0] / sum_pos * 0.5 if sum_pos != 0 else 0
    df_neg = alpha[alpha < 0] / abs(sum_neg) * 0.5 if sum_neg != 0 else 0

    # Create a new series to store scaled values
    alpha_copy = pd.Series(np.nan, index=alpha.index)
    alpha_copy[alpha > 0] = df_pos
    alpha_copy[alpha < 0] = df_neg
    
    # Replace NaN values with zero
    alpha_copy = alpha_copy.fillna(0)
    
    return alpha_copy

############################## PORTFOLIO FUNCTIONS ##############################

def generate_portfolio(
    contestant_get_weights: callable,
    entire_features: pd.DataFrame,
    universe: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:

    """
    Generate a portfolio by applying the contestant's strategy function
    over a specified date range. Ensures compliance with various constraints before returning 
    the portfolio.

    Parameters:
    -----------
    contestant_get_weights : callable
        A get_weights type function that computes portfolio weights based on historical features 
        and the daily stock universe.

    entire_features : pd.DataFrame
        A DataFrame containing historical stock features.
        - Index: Trading days.
        - Columns: MultiIndex with feature names and stock identifiers.

    universe : pd.DataFrame
        A DataFrame indicating which stocks are tradable on a given day.
        - Index: Trading days.
        - Columns: Stock identifiers.
        - Values: 1 if the stock is tradable, 0 otherwise.

    start_date : str
        The start date for the portfolio in "YYYY-MM-DD" format.

    end_date : str
        The end date for the portfolio in "YYYY-MM-DD" format.

    Returns:
    --------
    portfolio : pd.DataFrame
        A DataFrame containing daily stock weights in the portfolio.
        - Index: Trading days.
        - Columns: Stock identifiers.
        - Values: Stock weight allocations (float).
    """
    
    # Validate date format
    try:
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        cutoff_date = datetime.datetime.strptime('2005-01-01', '%Y-%m-%d')
    except ValueError:
        raise ValueError("start_date and end_date must be strings in 'YYYY-MM-DD' format.")

    # Ensure start_date is before end_date
    if start_dt >= end_dt:
        raise ValueError("start_date must be earlier than end_date.")

    # Ensure start_date is not before '2005-01-01'
    if start_dt < cutoff_date:
        raise ValueError("start_date must be later than '2005-01-01'.")

    # Get trading days within the date range
    trading_days = universe.index[(universe.index >= start_dt) & (universe.index <= end_dt)]

    if len(trading_days) == 0:
        raise ValueError("No Trading Days in the specified dates")
    
    # Initialize portfolio DataFrame with stock identifiers as columns
    portfolio = pd.DataFrame(index=trading_days, columns=universe.columns)

    # Iterate over trading days to compute portfolio weights
    for day in tqdm(trading_days, total=len(trading_days)):
        current_universe = universe.loc[day]

        # Filter feature data up to the previous day
        filtered_features = entire_features[entire_features.index < day]

        # Compute portfolio weights using the contestant's function
        current_weights = pd.Series(contestant_get_weights(filtered_features, current_universe))

        stocks_with_weight = set(current_weights.keys())
        stocks_not_in_universe = set(current_universe[current_universe == 0].index.tolist())

        # Validate that all selected stocks are in the universe
        if len(stocks_not_in_universe & stocks_with_weight) != 0:
            raise ValueError(f"Your returned weights dictionary has a stock which is not in the universe on {day}")
        
        # Check for dollar neutrality constraint
        if abs(current_weights.sum()) > 1e-2:
            raise ValueError(f"On {day}, Dollar Neutral Constraint is violated")
        
        # Check for unit capital constraint
        if (current_weights.abs().sum() - 1) > 1e-2:
            raise ValueError(f"On {day}, Unit Capital Constraint is violated")
        
        # Check for maximum weight constraint
        if current_weights.abs().max() > 0.1:
            raise ValueError(f"On {day}, Maximum Weight Constraint is violated")

        # Store the computed weights for the day
        portfolio.loc[day] = current_weights.reindex(current_universe.index, fill_value=np.nan)

    # Replace NaN values with 0 to finalize portfolio
    portfolio = portfolio.fillna(0)

    return portfolio

def backtest_portfolio(portfolio: pd.DataFrame, returns: pd.DataFrame, universe: pd.DataFrame, plot_: bool, print_: bool):

    """
    Computes performance metrics from a given portfolio DataFrame.

    This function calculates the net Sharpe ratio, along with other portfolio metrics,
    ensuring that certain constraints are met. It performs checks for:
    - Shape alignment of input DataFrames.
    - portfolio weights only in stocks that are part of the universe.
    - Dollar neutrality.
    - Unit capital constraint.
    - Maximum weight constraint.

    Parameters:
    ----------
    portfolio : pd.DataFrame
        DataFrame representing portfolio over time.
    returns : pd.DataFrame
        DataFrame containing stock returns corresponding to the portfolio.
    universe : pd.DataFrame
        Boolean DataFrame indicating whether a stock is part of the investable universe.
    plot_: bool
        Boolean Flag which decides whether to plot the cumulative PnL
    print_: bool
        Boolean Flag which decides whether to print the metrics of backtest

    Raises:
    ------
    ValueError
        If the input DataFrames do not have matching shapes.
        If portfolio contain stocks that are not in the universe.
        If the portfolio violates the dollar neutrality constraint.
        If the unit capital constraint is violated.
        If the maximum weight constraint is exceeded.

    Returns
    -------
    tuple
        - Net Sharpe Ratio (rounded to 3 decimal places).
        - Pandas Series containing gross PnL over time.

    Additional Outputs:
    -------------------
    - Prints the gross and net Sharpe ratios.
    - Prints the turnover percentage.
    - Plots cumulative Gross and Net PnL.

    Notes:
    ------
    - The turnover is calculated as the average traded capital divided by the average book value.
    - The gross and net Sharpe ratios are annualized using a factor of √252.
    - The net PnL accounts for trading costs (assumed to be 0.01% per unit traded).
    """

    universe = universe.astype(bool)
    
    if not (portfolio.shape == returns.shape == universe.shape):
        raise ValueError("Shapes of portfolio, returns and universe are not algined")

    if ((portfolio.replace(0, np.nan))[~universe].notna().sum().sum() != 0):
        raise ValueError("Your portfolio are present for a stock not present in the universe")

    abs_sum = portfolio.abs().sum(axis=1)
    non_zero_rows = abs_sum > 1e-10
    
    
    if ((abs_sum - 1).abs() > 0.01)[non_zero_rows].sum() > 0:
        raise ValueError("Unit capital constraint violated")
    
    if (((portfolio.abs().sum(1) - 1) > 0.01).sum() > 0):
        raise ValueError("Unit Capital Constraint is violated")

    if ((portfolio.abs().max(1) > 0.1).sum() > 0):
        raise ValueError("Maximum Weight Constraint is violated")

    portfolio = portfolio.fillna(0)

    rets = returns.fillna(0)

    gross_pnl = (portfolio * rets).sum(axis=1)

    traded = portfolio.diff(1).abs().sum(axis=1).fillna(0)

    book_value = portfolio.abs().sum(axis=1)

    turnover = (traded.mean() / book_value.mean()) * 100

    net_pnl = gross_pnl - traded * 1e-4

    gross_sharpe_ratio = (gross_pnl.mean() / gross_pnl.std()) * np.sqrt(252)

    net_sharpe_ratio = (net_pnl.mean() / net_pnl.std()) * np.sqrt(252)

    if print_:
        print("Gross Sharpe Ratio: ", round(gross_sharpe_ratio, 3))
        print("Net Sharpe Ratio: ", round(net_sharpe_ratio, 3))
        print("Turnover %: ", round(turnover, 3))
    
    if plot_:
        plot_series_with_names([gross_pnl.cumsum(), net_pnl.cumsum()], ["Gross PnL", "Net PnL"], "Cumulative PnL Plot", yaxis_title="PnL", xaxis_title="Date")

    return round(net_sharpe_ratio, 3), gross_pnl

def match_implementations(
    contestant_get_weights: callable,
    contestant_vectorized_portfolio: pd.DataFrame,
    entire_features: pd.DataFrame,
    universe: pd.DataFrame,
    returns: pd.DataFrame
):
    """
    Compare the iterative and vectorized implementations of portfolio generation 
    by running both and checking the correlation of their PnL (Profit and Loss).

    Parameters:
    -----------
    contestant_get_weights : callable
        A get_weights type function that computes portfolio weights based on historical features 
        and the daily stock universe.
    
    contestant_vectorized_portfolio: pd.DataFrame
        Portfolio DataFrame generated using vectorized operations. You can generate this using generate_vectorized_portfolio function

    entire_features : pd.DataFrame
        A DataFrame containing feature data.
        - Index: Datetime (chronological order).
        - Columns: MultiIndex with two levels:
          - Level 0: Feature names (e.g., "macd", "ichimoku", ..., "volatility_60").
          - Level 1: Stock identifiers (e.g., "1", "2", ..., "2167").
    
    universe : pd.DataFrame
        A DataFrame indicating which stocks can be traded on a given day.
        - Index: Datetime (trading days).
        - Columns: Stock identifiers.
        - Values: 0 or 1, where 1 means the stock is in the universe and 0 means it is not.

    returns : pd.DataFrame
        A DataFrame containing daily stock returns.
        - Index: Datetime (trading days).
        - Columns: Stock identifiers.
        - Values: Daily returns for each stock.

    Raises:
    -------
    ValueError
        - If the correlation between the PnL of the two implementations is NaN or infinite.
        - If the correlation is below 0.98, indicating a mismatch between implementations.
    
    Notes:
    ------
    - This function randomly selects a start date within the universe index range.
    - It generates portfolios using both iterative and vectorized methods.
    - It backtests both portfolios and computes the correlation between their PnLs.
    - If the correlation is high (≥ 0.98), the implementations are considered equivalent.
    """

    # Randomly select a start index within a safe range
    start_index = np.random.randint(500, 3000)

    # Determine start and end dates for the backtest period
    start_date = universe.index[start_index].strftime("%Y-%m-%d")
    end_date = universe.index[start_index + 40].strftime("%Y-%m-%d")

    print("Starting to generate Iterative Portfolio")

    # Generate portfolio using the iterative method
    iterative_portfolio = generate_portfolio(
        contestant_get_weights,
        entire_features,
        universe,
        start_date,
        end_date,
    )

    print("Iterative Portfolio Generated")

    vectorized_portfolio = contestant_vectorized_portfolio.loc[start_date:end_date]

    # Backtest both portfolios
    _, iterative_pnl = backtest_portfolio(
        iterative_portfolio.loc[start_date:end_date], 
        returns.loc[start_date:end_date], 
        universe.loc[start_date:end_date], 
        False, 
        False
    )
    _, vectorized_pnl = backtest_portfolio(
        vectorized_portfolio.loc[start_date:end_date], 
        returns.loc[start_date:end_date], 
        universe.loc[start_date:end_date], 
        False, 
        False
    )

    # Compute correlation between the PnLs of both implementations
    correlation = iterative_pnl.corr(vectorized_pnl)

    # Validate correlation to ensure both implementations match
    if correlation in [np.nan, np.inf, -np.inf]:
        raise ValueError(f"Invalid Correlation of {correlation} between PnL of Iterative and Vectorized Implementations!")
    elif correlation < 0.98:
        raise ValueError(f"Correlation {correlation} between PnL of Iterative and Vectorized Implementations is less than 0.98. Both Implementations don't match!")
    else:
        print(f"Correlation of {correlation} between Iterative and Vectorized Implementations. Both implementations match!")
