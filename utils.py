import pandas as pd
from typing import Dict

ALLOWED_TICKERS = [
    "AAPL", "TSLA", "AMZN", "MSFT", "NVDA", "GOOGL", "META", "NFLX", "JPM", "V",
    "BAC", "AMD", "PYPL", "DIS", "T", "PFE", "COST", "INTC", "KO", "TGT", "NKE",
    "SPY", "BA", "BABA", "XOM", "WMT", "GE", "CSCO", "VZ", "JNJ", "CVX", "PLTR",
    "SQ", "SHOP", "SBUX", "SOFI", "HOOD", "RBLX", "SNAP", "UBER", "FDX", "ABBV",
    "ETSY", "MRNA", "LMT", "GM", "F", "RIVN", "LCID", "CCL", "DAL", "UAL", "AAL",
    "TSM", "SONY", "ET", "NOK", "MRO", "COIN", "RIOT", "CPRX", "VWO", "SPYG",
    "ROKU", "VIAC", "ATVI", "BIDU", "DOCU", "ZM", "PINS", "TLRY", "WBA", "MGM",
    "NIO", "C", "GS", "WFC", "ADBE", "PEP", "UNH", "CARR", "FUBO", "HCA", "BILI",
    "SIRI", "RKT"
]

TICKER_INDUSTRY_MAP = {
    # Technology/Software/Semiconductors
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Technology",
    "META": "Technology", "NFLX": "Technology", "AMD": "Technology", "PYPL": "Technology",
    "PLTR": "Technology", "SQ": "Technology", "SHOP": "Technology", "SOFI": "Technology",
    "HOOD": "Technology", "RBLX": "Technology", "SNAP": "Technology", "UBER": "Technology",
    "ETSY": "Technology", "TSM": "Technology", "SONY": "Technology", "COIN": "Technology",
    "RIOT": "Technology", "ROKU": "Technology", "ADBE": "Technology", "CSCO": "Technology",
    "BIDU": "Technology", "DOCU": "Technology", "ZM": "Technology", "PINS": "Technology",
    "FUBO": "Technology", "BILI": "Technology", "SIRI": "Technology", "ATVI": "Technology",

    # Automotive (and related EVs)
    "TSLA": "Automotive", "GM": "Automotive", "F": "Automotive", "RIVN": "Automotive",
    "LCID": "Automotive", "NIO": "Automotive",

    # Retail/Consumer Goods/Food & Beverage
    "AMZN": "Retail", "COST": "Retail", "TGT": "Retail", "NKE": "Retail", "WMT": "Retail",
    "SBUX": "Retail", "KO": "Food & Beverage", "PEP": "Food & Beverage",

    # Financials
    "JPM": "Financials", "V": "Financials", "BAC": "Financials", "GS": "Financials",
    "WFC": "Financials", "C": "Financials", "RKT": "Financials",

    # Healthcare/Pharma
    "PFE": "Healthcare", "JNJ": "Healthcare", "MRNA": "Healthcare", "ABBV": "Healthcare",
    "UNH": "Healthcare", "HCA": "Healthcare", "CPRX": "Healthcare",

    # Industrials/Manufacturing/Aerospace & Defense
    "BA": "Industrials", "GE": "Industrials", "MMM": "Industrials", "LMT": "Industrials",
    "CAT": "Industrials", "CARR": "Industrials",

    # Telecom
    "T": "Telecommunications", "VZ": "Telecommunications", "NOK": "Telecommunications",

    # Energy
    "XOM": "Energy", "CVX": "Energy", "ET": "Energy", "MRO": "Energy",

    # Travel & Leisure
    "DIS": "Travel & Leisure", "CCL": "Travel & Leisure", "DAL": "Travel & Leisure",
    "UAL": "Travel & Leisure", "AAL": "Travel & Leisure", "MGM": "Travel & Leisure",

    # ETFs (broad market, not specific industry for company analysis)
    "SPY": "ETFs", "VWO": "ETFs", "SPYG": "ETFs",

    # Others / Uncategorized for now
    "TLRY": "Other", "WBA": "Other", "VIAC": "Other", "NFLX": "Other", # Netflix is in Tech/Entertainment
    "TWTR": "Other", # Now X, part of Tech/Social Media
}

qualitative_insights_map = {
        "Technology": {
            "growth_outlook": "High (Rapid innovation, digital transformation)",
            "regulatory_risk": "Moderate to High (Antitrust, privacy, AI ethics)",
            "competitive_intensity": "Very High (Global competition, rapid obsolescence)"
        },
        "Retail": {
            "growth_outlook": "Moderate to Low (E-commerce disruption, consumer sentiment sensitivity)",
            "regulatory_risk": "Low to Moderate (Labor laws, consumer protection)",
            "competitive_intensity": "Very High (E-commerce giants, pricing pressure)"
        },
        "Manufacturing": {
            "growth_outlook": "Moderate (Supply chain resilience, automation, reshoring trends)",
            "regulatory_risk": "High (Environmental, labor, trade policies, safety standards)",
            "competitive_intensity": "Moderate to High (Globalized supply chains, cost efficiency)"
        },
        "Automotive": {
            "growth_outlook": "Moderate (EV transition, autonomous driving R&D)",
            "regulatory_risk": "High (Emissions, safety, EV mandates)",
            "competitive_intensity": "High (New entrants, traditional OEMs adapting)"
        },
        "Healthcare": {
            "growth_outlook": "High (Aging population, technological advancements, chronic diseases)",
            "regulatory_risk": "Very High (FDA approvals, drug pricing, insurance, data privacy)",
            "competitive_intensity": "Moderate (Consolidation, R&D costs)"
        },
        "Financials": {
            "growth_outlook": "Moderate (Interest rate environment, digital banking trends)",
            "regulatory_risk": "Very High (Strict compliance, capital requirements, systemic risk)",
            "competitive_intensity": "High (Fintech disruption, traditional banks vs. new players)"
        },
        "Telecommunications": {
            "growth_outlook": "Stable (5G rollout, infrastructure investment)",
            "regulatory_risk": "High (Spectrum regulation, net neutrality, data privacy)",
            "competitive_intensity": "High (Few large players, high capital expenditure)"
        },
        "Energy": {
            "growth_outlook": "Cyclical (Commodity prices, renewable transition)",
            "regulatory_risk": "High (Environmental regulations, climate policy, geopolitical factors)",
            "competitive_intensity": "Moderate (OPEC+ influence, large integrated players)"
        },
        "Travel & Leisure": {
            "growth_outlook": "Moderate (Post-pandemic recovery, discretionary spending sensitivity)",
            "regulatory_risk": "Moderate (Travel advisories, health regulations, labor issues)",
            "competitive_intensity": "High (Online travel agencies, diverse offerings)"
        },
        "Food & Beverage": {
            "growth_outlook": "Stable (Consumer staples, population growth)",
            "regulatory_risk": "Moderate (Food safety, labeling, advertising standards)",
            "competitive_intensity": "Moderate to High (Brand loyalty, health trends, private labels)"
        },
        # Add more if needed
        "ETFs": { # ETFs are not an industry for fundamental analysis
             "growth_outlook": "Depends on underlying assets",
             "regulatory_risk": "Moderate",
             "competitive_intensity": "Moderate"
        },
        "Other": {
            "growth_outlook": "Varied",
            "regulatory_risk": "Varied",
            "competitive_intensity": "Varied"
        }
    }


def calculate_ratios_from_df_row(row: pd.Series) -> Dict[str, float]:
    """
    Calculates key financial ratios for a single company from a Pandas Series (row data)
    based on FMP TTM Income Statement and Balance Sheet data structure.
    """
    ratios = {}

    # --- Data Extraction ---
    # Income Statement
    revenue = row.get("revenue", 0)
    cost_of_revenue = row.get("costOfRevenue", 0)
    gross_profit_fmp = row.get("grossProfit", 0) # Direct from FMP
    operating_expenses_fmp = row.get("operatingExpenses", 0) # Direct from FMP (often R&D + SG&A)
    operating_income_fmp = row.get("operatingIncome", 0) # Direct from FMP (EBIT)
    interest_expense = row.get("interestExpense", 0)
    net_income = row.get("netIncome", 0)

    # Balance Sheet
    cash_and_equivalents = row.get("cashAndCashEquivalents", 0)
    current_assets = row.get("currentAssets", 0)
    current_liabilities = row.get("currentLiabilities", 0)
    inventory = row.get("inventory", 0)
    net_receivables = row.get("netReceivables", 0) # Used for Accounts Receivable
    total_liabilities = row.get("totalLiabilities", 0)
    shareholders_equity = row.get("totalStockholdersEquity", 0) # FMP uses totalEquity
    total_assets = row.get("totalAssets", 0)

    # --- Ratio Calculations ---

    # 1. Profitability Ratios
    if revenue > 0:
        # Gross Profit Margin (using FMP's grossProfit directly)
        ratios["gross_profit_margin"] = gross_profit_fmp / revenue
        
        # Operating Profit Margin (using FMP's operatingIncome directly as EBIT)
        ratios["operating_profit_margin"] = operating_income_fmp / revenue
        
        # Net Profit Margin
        ratios["net_profit_margin"] = net_income / revenue
    else:
        ratios["gross_profit_margin"] = 0.0
        ratios["operating_profit_margin"] = 0.0
        ratios["net_profit_margin"] = 0.0

    # 2. Liquidity Ratios
    if current_liabilities > 0:
        ratios["current_ratio"] = current_assets / current_liabilities
        ratios["quick_ratio"] = (current_assets - inventory) / current_liabilities
        ratios["cash_ratio"] = cash_and_equivalents / current_liabilities
    else:
        ratios["current_ratio"] = float('inf')
        ratios["quick_ratio"] = float('inf')
        ratios["cash_ratio"] = float('inf')

    # 3. Solvency/Leverage Ratios
    if shareholders_equity > 0:
        ratios["debt_to_equity_ratio"] = total_liabilities / shareholders_equity
    else:
        ratios["debt_to_equity_ratio"] = float('inf') # Handles zero or negative equity

    if total_assets > 0:
        ratios["debt_to_assets_ratio"] = total_liabilities / total_assets
    else:
        ratios["debt_to_assets_ratio"] = float('inf')

    # Interest Coverage Ratio (using FMP's operatingIncome as EBIT)
    if interest_expense > 0:
        ratios["interest_coverage_ratio"] = operating_income_fmp / interest_expense
    else:
        # Infinite if no interest expense or if EBIT is positive and no interest expense
        ratios["interest_coverage_ratio"] = float('inf')

    # 4. Efficiency/Activity Ratios
    # Assuming annual figures, or that TTM revenue/COGS aligns with TTM inventory/AR for turnover
    if inventory > 0:
        ratios["inventory_turnover"] = cost_of_revenue / inventory
    else:
        ratios["inventory_turnover"] = float('inf')

    # Days Sales Outstanding
    # Using netReceivables as Accounts Receivable. Assumes revenue is for 365 days.
    if revenue > 0:
        ratios["days_sales_outstanding"] = (net_receivables / revenue) * 365
    else:
        ratios["days_sales_outstanding"] = 0.0

    if total_assets > 0:
        ratios["asset_turnover"] = revenue / total_assets
    else:
        ratios["asset_turnover"] = 0.0
    
    print(f"    DEBUG Ratios Calc for {row.get('symbol', 'UNKNOWN_SYMBOL')}:")
    print(f"      Revenue: {revenue}, Cost of Revenue: {cost_of_revenue}, Gross Profit FMP: {gross_profit_fmp}")
    print(f"      Operating Income FMP: {operating_income_fmp}, Net Income: {net_income}")
    print(f"      Current Assets: {current_assets}, Current Liabilities: {current_liabilities}, Inventory: {inventory}")
    print(f"      Net Receivables: {net_receivables}, Total Liabilities: {total_liabilities}, Shareholders Equity: {shareholders_equity}, Total Assets: {total_assets}")
    print(f"      Interest Expense: {interest_expense}")

    return ratios


SYSTEM_PROMPT_AGENT = """You are an expert financial risk assessment AI.
Your goal is to provide a concise, high-level financial risk assessment for a given industry.

Here's your process:
1.  **Information Gathering:** To get financial data for an industry, you MUST use the `get_industry_financial_data` tool.
2.  **Analysis:** Once you have the necessary data, analyze the following key indicators:
    -   Profitability (Median Gross Margin, Median Net Profit Margin, Median Operating Profit Margin)
    -   Leverage (Median Debt-to-Equity Ratio, Median Debt-to-Assets Ratio)
    -   Liquidity (Median Current Ratio, Median Quick Ratio, Median Cash Ratio)
    -   Efficiency (Median Inventory Turnover, Median Days Sales Outstanding, Median Asset Turnover)
    -   Solvency (Median Interest Coverage Ratio)
    -   Growth Outlook (from tool)
    -   Regulatory Risk (from tool)
    -   Competitive Intensity (from tool)

    Interpret values of `inf` for ratios like Interest Coverage, Current Ratio, Quick Ratio, etc., as **extremely strong** or **no relevant liability**, indicating very low risk in that specific area.
3.  **Assessment:** For each category, briefly explain what the provided values imply for a typical company operating in this industry, considering industry context.
4.  **Conclusion:** Conclude with an overall risk assessment (e.g., Low, Moderate, High) for a typical company in this industry, justifying your conclusion based on all the data.
5.  **Final Output:** Once you have completed your assessment and are ready to provide the final report, output the report directly as your final answer. Do NOT call any more tools after giving the final report.
"""
