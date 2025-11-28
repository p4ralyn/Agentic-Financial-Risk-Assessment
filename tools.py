import yfinance as yf
from langchain_core.tools import tool

@tool
def get_industry_financial_data(industry: str):
    """
    Retrieve financial data for all tickers in a given industry using Yahoo Finance.
    Returns revenue, net income, operating income, and gross profit for each ticker.
    """

    # Industry → Tickers mapping
    industry_map = {
        "Telecommunications": ["T", "VZ", "NOK"],
        "Technology": ["AAPL", "MSFT", "GOOGL"],
        "Automotive": ["TSLA", "F", "GM"],
        "Finance": ["JPM", "BAC", "WFC"],
        "Energy": ["XOM", "CVX", "BP"],
    }

    tickers = industry_map.get(industry, [])
    if not tickers:
        return {"error": f"No tickers mapped for industry '{industry}'"}

    financials = []

    for t in tickers:
        try:
            print(f"Fetching Yahoo Finance data for {t}...")

            yf_ticker = yf.Ticker(t)
            inc = yf_ticker.financials

            if inc is None or inc.empty:
                print(f"No income statement for {t}")
                continue

            latest = inc.iloc[:, 0]  # Most recent column

            financials.append({
                "ticker": t,
                "revenue": float(latest.get("Total Revenue", 0)),
                "net_income": float(latest.get("Net Income", 0)),
                "operating_income": float(latest.get("Operating Income", 0)),
                "gross_profit": float(latest.get("Gross Profit", 0)),
            })

        except Exception as e:
            print(f"Error fetching {t}: {e}")

    if not financials:
        return {"error": f"Could not retrieve financial data for industry '{industry}' using Yahoo Finance."}

    return {
        "industry": industry,
        "tickers": financials,
    }
