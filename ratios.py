"""Financial ratio math. Pure functions, no I/O.

A missing input or an unusable denominator yields None, never 0 and never
inf. v2 returned float('inf') and instructed the model to read it as
"extremely strong" -- but debt_to_equity reaches inf exactly when equity is
zero or negative, which is severe distress. That inverted the signal on the
companies this tool exists to catch.
"""
from statistics import median

import pandas as pd

REASONS = {
    "current_ratio": "current liabilities are zero or missing",
    "quick_ratio": "current liabilities or inventory unavailable",
    "cash_ratio": "current liabilities are zero or missing",
    "debt_to_equity": "shareholders equity is zero or negative",
    "debt_to_assets": "total assets are zero or missing",
    "interest_coverage": "no interest expense reported",
    "gross_margin": "revenue is zero or missing",
    "operating_margin": "revenue is zero or missing",
    "net_margin": "revenue is zero or missing",
    "roa": "total assets are zero or missing",
    "roe": "shareholders equity is zero or negative",
    "asset_turnover": "total assets are zero or missing",
    "inventory_turnover": "inventory is zero or missing",
    "days_sales_outstanding": "revenue is zero or missing",
}


def pick(series: pd.Series, aliases: list[str]) -> float | None:
    """First alias present in the series with a real value.

    yfinance renames rows between releases (Current Assets vs Total Current
    Assets), so every field is looked up against a list of known spellings.
    """
    for name in aliases:
        if name in series.index:
            value = series[name]
            if value is not None and not pd.isna(value):
                return float(value)
    return None


def _div(numerator: float | None, denominator: float | None,
         require_positive: bool = True) -> float | None:
    if numerator is None or denominator is None:
        return None
    if require_positive and denominator <= 0:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def compute_ratios(fields: dict) -> dict[str, float | None]:
    f = fields.get
    revenue = f("revenue")
    assets = f("total_assets")
    equity = f("equity")
    current_liabilities = f("current_liabilities")

    quick_assets = None
    if f("current_assets") is not None and f("inventory") is not None:
        quick_assets = f("current_assets") - f("inventory")

    dso_base = _div(f("accounts_receivable"), revenue)

    return {
        "current_ratio": _div(f("current_assets"), current_liabilities),
        "quick_ratio": _div(quick_assets, current_liabilities),
        "cash_ratio": _div(f("cash"), current_liabilities),
        "debt_to_equity": _div(f("total_liabilities"), equity),
        "debt_to_assets": _div(f("total_liabilities"), assets),
        "interest_coverage": _div(f("ebit"), f("interest_expense")),
        "gross_margin": _div(f("gross_profit"), revenue),
        "operating_margin": _div(f("operating_income"), revenue),
        "net_margin": _div(f("net_income"), revenue),
        "roa": _div(f("net_income"), assets),
        "roe": _div(f("net_income"), equity),
        "asset_turnover": _div(revenue, assets),
        "inventory_turnover": _div(f("cost_of_revenue"), f("inventory")),
        "days_sales_outstanding": None if dso_base is None else dso_base * 365,
    }


def altman_z(fields: dict) -> float | None:
    """Altman Z-score, public-company coefficients.

    Z = 1.2(WC/TA) + 1.4(RE/TA) + 3.3(EBIT/TA) + 0.6(MVE/TL) + 1.0(Rev/TA)
    """
    terms = [
        (1.2, _div(fields.get("working_capital"), fields.get("total_assets"))),
        (1.4, _div(fields.get("retained_earnings"), fields.get("total_assets"))),
        (3.3, _div(fields.get("ebit"), fields.get("total_assets"))),
        (0.6, _div(fields.get("market_cap"), fields.get("total_liabilities"))),
        (1.0, _div(fields.get("revenue"), fields.get("total_assets"))),
    ]
    if any(value is None for _, value in terms):
        return None
    return sum(weight * value for weight, value in terms)


def z_band(z: float | None) -> str:
    if z is None:
        return "unknown"
    if z < 1.81:
        return "distress"
    if z <= 2.99:
        return "grey"
    return "safe"


def median_ratios(peer_ratios: list[dict]) -> dict[str, float | None]:
    """Median of each ratio across peers, ignoring peers missing that ratio."""
    names = {name for peer in peer_ratios for name in peer}
    out = {}
    for name in names:
        values = [p[name] for p in peer_ratios if p.get(name) is not None]
        out[name] = median(values) if values else None
    return out
