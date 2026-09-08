"""yfinance access and peer-group assembly."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import pandas as pd
import yfinance as yf

from ratios import altman_z, compute_ratios, median_ratios, pick, z_band


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

QUALITATIVE_INSIGHTS = {
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


# yfinance renames statement rows between releases; each field is resolved
# against every spelling we have seen.
BALANCE_FIELDS = {
    "total_assets": ["Total Assets"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "inventory": ["Inventory"],
    "accounts_receivable": ["Accounts Receivable", "Receivables", "Net Receivables"],
    "cash": ["Cash And Cash Equivalents",
             "Cash Cash Equivalents And Short Term Investments"],
    "total_liabilities": ["Total Liabilities Net Minority Interest",
                          "Total Liabilities"],
    "equity": ["Stockholders Equity", "Total Stockholder Equity"],
    "retained_earnings": ["Retained Earnings"],
    "working_capital": ["Working Capital"],
}

INCOME_FIELDS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "cost_of_revenue": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported"],
    "ebit": ["EBIT", "Operating Income"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
}


def industry_for(ticker: str) -> str | None:
    return TICKER_INDUSTRY_MAP.get(ticker.upper())


def peers_for(ticker: str, limit: int = 6) -> list[str]:
    industry = industry_for(ticker)
    if industry is None:
        return []
    return [t for t, ind in TICKER_INDUSTRY_MAP.items()
            if ind == industry and t != ticker.upper()][:limit]


def _latest(frame) -> "pd.Series | None":
    if frame is None or frame.empty:
        return None
    return frame.iloc[:, 0]


def extract_fields(ticker_obj) -> dict:
    fields = {}

    balance = _latest(ticker_obj.balance_sheet)
    for name, aliases in BALANCE_FIELDS.items():
        fields[name] = pick(balance, aliases) if balance is not None else None

    income = _latest(ticker_obj.financials)
    for name, aliases in INCOME_FIELDS.items():
        fields[name] = pick(income, aliases) if income is not None else None

    # Subscript, not .get: fast_info's keys are camelCase and only
    # __getitem__/__getattr__ alias snake_case. .get('market_cap') is None.
    try:
        fields["market_cap"] = float(ticker_obj.fast_info["market_cap"])
    except (KeyError, TypeError, ValueError):
        fields["market_cap"] = None

    return fields


@lru_cache(maxsize=64)
def fetch_company(ticker: str) -> dict:
    """Statements are published quarterly, so a process-lifetime cache is
    sufficient; a TTL would be more code than the staleness it prevents."""
    obj = yf.Ticker(ticker)
    fields = extract_fields(obj)
    name = ticker
    try:
        name = obj.info.get("longName") or ticker
    except Exception:
        pass
    z = altman_z(fields)
    return {
        "ticker": ticker,
        "name": name,
        "fields": fields,
        "ratios": compute_ratios(fields),
        "z": z,
        "band": z_band(z),
    }


def _safe_fetch(ticker: str):
    """A failing peer degrades the benchmark; it must not abort the run."""
    try:
        return fetch_company(ticker)
    except Exception as exc:
        return exc


def assess(ticker: str) -> dict:
    ticker = ticker.upper()
    errors: list[str] = []

    try:
        company = fetch_company(ticker)
    except Exception as exc:
        raise ValueError(f"Could not retrieve financial data for {ticker}: {exc}")

    peers = peers_for(ticker)
    peer_ratios = []
    if peers:
        with ThreadPoolExecutor(max_workers=len(peers)) as pool:
            for symbol, result in zip(peers, pool.map(_safe_fetch, peers)):
                if isinstance(result, Exception):
                    errors.append(f"{symbol}: {result}")
                else:
                    peer_ratios.append(result["ratios"])

    industry = industry_for(ticker)
    return {
        "ticker": ticker,
        "company_name": company["name"],
        "industry": industry,
        "peer_tickers": peers,
        "company_ratios": company["ratios"],
        "industry_median_ratios": median_ratios(peer_ratios) if peer_ratios else {},
        "z": company["z"],
        "band": company["band"],
        "qualitative": QUALITATIVE_INSIGHTS.get(industry or "", {}),
        "data_errors": errors,
    }
