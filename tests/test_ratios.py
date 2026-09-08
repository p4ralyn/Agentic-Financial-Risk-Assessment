import math
import pandas as pd
import pytest

from ratios import pick, compute_ratios, altman_z, z_band, median_ratios

HEALTHY = {
    "total_assets": 1000.0, "current_assets": 400.0, "current_liabilities": 200.0,
    "inventory": 100.0, "accounts_receivable": 80.0, "cash": 50.0,
    "total_liabilities": 600.0, "equity": 400.0, "retained_earnings": 150.0,
    "working_capital": 200.0, "revenue": 800.0, "cost_of_revenue": 500.0,
    "gross_profit": 300.0, "operating_income": 120.0, "ebit": 120.0,
    "interest_expense": 30.0, "net_income": 60.0, "market_cap": 1200.0,
}


def test_pick_returns_first_matching_alias():
    s = pd.Series({"Total Current Assets": 400.0})
    assert pick(s, ["Current Assets", "Total Current Assets"]) == 400.0


def test_pick_returns_none_when_no_alias_matches():
    s = pd.Series({"Something Else": 1.0})
    assert pick(s, ["Current Assets", "Total Current Assets"]) is None


def test_pick_returns_none_for_nan():
    s = pd.Series({"Inventory": float("nan")})
    assert pick(s, ["Inventory"]) is None


def test_liquidity_ratios():
    r = compute_ratios(HEALTHY)
    assert r["current_ratio"] == pytest.approx(2.0)
    assert r["quick_ratio"] == pytest.approx(1.5)
    assert r["cash_ratio"] == pytest.approx(0.25)


def test_leverage_and_coverage_ratios():
    r = compute_ratios(HEALTHY)
    assert r["debt_to_equity"] == pytest.approx(1.5)
    assert r["debt_to_assets"] == pytest.approx(0.6)
    assert r["interest_coverage"] == pytest.approx(4.0)


def test_profitability_ratios():
    r = compute_ratios(HEALTHY)
    assert r["gross_margin"] == pytest.approx(0.375)
    assert r["operating_margin"] == pytest.approx(0.15)
    assert r["net_margin"] == pytest.approx(0.075)
    assert r["roa"] == pytest.approx(0.06)
    assert r["roe"] == pytest.approx(0.15)


def test_efficiency_ratios():
    r = compute_ratios(HEALTHY)
    assert r["asset_turnover"] == pytest.approx(0.8)
    assert r["inventory_turnover"] == pytest.approx(5.0)
    assert r["days_sales_outstanding"] == pytest.approx(36.5)


def test_negative_equity_yields_none_not_infinity():
    """The v2 bug: negative equity produced inf, which the prompt read as
    'extremely strong'. Negative equity is severe distress."""
    fields = {**HEALTHY, "equity": -100.0}
    r = compute_ratios(fields)
    assert r["debt_to_equity"] is None
    assert r["roe"] is None


def test_zero_denominator_yields_none_not_infinity():
    fields = {**HEALTHY, "current_liabilities": 0.0}
    r = compute_ratios(fields)
    assert r["current_ratio"] is None
    assert not any(
        isinstance(v, float) and math.isinf(v) for v in r.values()
    ), "no ratio may ever be infinite"


def test_missing_field_yields_none_not_zero():
    fields = {**HEALTHY, "inventory": None}
    r = compute_ratios(fields)
    assert r["quick_ratio"] is None
    assert r["inventory_turnover"] is None


def test_altman_z_on_known_fixture():
    # 1.2(.2) + 1.4(.15) + 3.3(.12) + 0.6(2.0) + 1.0(.8)
    # = .24 + .21 + .396 + 1.2 + .8
    assert altman_z(HEALTHY) == pytest.approx(2.846)


def test_altman_z_returns_none_when_market_cap_missing():
    assert altman_z({**HEALTHY, "market_cap": None}) is None


@pytest.mark.parametrize("z,expected", [
    (1.80, "distress"), (1.81, "grey"), (2.99, "grey"),
    (3.00, "safe"), (None, "unknown"),
])
def test_z_band_boundaries(z, expected):
    assert z_band(z) == expected


def test_median_ratios_odd_count():
    peers = [{"current_ratio": 1.0}, {"current_ratio": 3.0}, {"current_ratio": 2.0}]
    assert median_ratios(peers)["current_ratio"] == pytest.approx(2.0)


def test_median_ratios_even_count_averages_middle_two():
    peers = [{"current_ratio": 1.0}, {"current_ratio": 2.0},
             {"current_ratio": 3.0}, {"current_ratio": 4.0}]
    assert median_ratios(peers)["current_ratio"] == pytest.approx(2.5)


def test_median_ratios_ignores_none_values():
    peers = [{"current_ratio": 2.0}, {"current_ratio": None}, {"current_ratio": 4.0}]
    assert median_ratios(peers)["current_ratio"] == pytest.approx(3.0)


def test_median_ratios_all_none_yields_none():
    assert median_ratios([{"current_ratio": None}])["current_ratio"] is None
