import pandas as pd
import pytest

import financial_data as fd


def _frame(rows: dict) -> pd.DataFrame:
    """A yfinance-shaped statement: rows are line items, columns are periods,
    most recent first."""
    return pd.DataFrame({pd.Timestamp("2024-12-31"): rows})


BALANCE = {
    "Total Assets": 1000.0, "Current Assets": 400.0,
    "Current Liabilities": 200.0, "Inventory": 100.0,
    "Accounts Receivable": 80.0, "Cash And Cash Equivalents": 50.0,
    "Total Liabilities Net Minority Interest": 600.0,
    "Stockholders Equity": 400.0, "Retained Earnings": 150.0,
    "Working Capital": 200.0,
}

INCOME = {
    "Total Revenue": 800.0, "Cost Of Revenue": 500.0,
    "Gross Profit": 300.0, "Operating Income": 120.0,
    "EBIT": 120.0, "Interest Expense": 30.0, "Net Income": 60.0,
}


class FakeFastInfo(dict):
    """fast_info exposes camelCase keys but aliases snake_case on __getitem__.
    .get('market_cap') therefore returns None -- the trap this stands in for."""
    def __getitem__(self, key):
        return super().__getitem__("marketCap" if key == "market_cap" else key)


class FakeTicker:
    def __init__(self, market_cap=1200.0, fail=False, balance=None):
        self._fail = fail
        self._balance = balance if balance is not None else BALANCE
        self.fast_info = FakeFastInfo({"marketCap": market_cap})

    def _check(self):
        if self._fail:
            raise RuntimeError("network down")

    @property
    def balance_sheet(self):
        self._check()
        return _frame(self._balance)

    @property
    def financials(self):
        self._check()
        return _frame(INCOME)

    @property
    def info(self):
        return {"longName": "Fixture Corp"}


def test_industry_for_known_ticker():
    assert fd.industry_for("VZ") == "Telecommunications"


def test_industry_for_unknown_ticker():
    assert fd.industry_for("ZZZZ") is None


def test_peers_exclude_the_company_itself():
    peers = fd.peers_for("VZ")
    assert "VZ" not in peers
    assert "T" in peers


def test_peers_for_unknown_ticker_is_empty():
    assert fd.peers_for("ZZZZ") == []


def test_peers_respect_limit():
    assert len(fd.peers_for("AAPL", limit=2)) == 2


def test_extract_fields_reads_yfinance_labels():
    fields = fd.extract_fields(FakeTicker())
    assert fields["total_assets"] == 1000.0
    assert fields["total_liabilities"] == 600.0
    assert fields["equity"] == 400.0
    assert fields["market_cap"] == 1200.0


def test_extract_fields_uses_subscript_for_market_cap():
    """Regression: fast_info.get('market_cap') returns None because the keys
    are camelCase. Only subscript/attribute access aliases snake_case."""
    assert fd.extract_fields(FakeTicker(market_cap=5.0))["market_cap"] == 5.0


def test_extract_fields_tolerates_alias_spelling():
    renamed = dict(BALANCE)
    renamed["Total Current Assets"] = renamed.pop("Current Assets")
    assert fd.extract_fields(FakeTicker(balance=renamed))["current_assets"] == 400.0


def test_fetch_company_is_cached(monkeypatch):
    calls = []

    def fake_ticker(symbol):
        calls.append(symbol)
        return FakeTicker()

    monkeypatch.setattr(fd.yf, "Ticker", fake_ticker)
    fd.fetch_company.cache_clear()
    fd.fetch_company("VZ")
    fd.fetch_company("VZ")
    assert len(calls) == 1, "second call should hit the lru_cache"


def test_assess_records_failing_peer_without_aborting(monkeypatch):
    def fake_ticker(symbol):
        return FakeTicker(fail=(symbol == "T"))

    monkeypatch.setattr(fd.yf, "Ticker", fake_ticker)
    fd.fetch_company.cache_clear()
    result = fd.assess("VZ")
    assert result["company_ratios"]["current_ratio"] == pytest.approx(2.0)
    assert any("T" in err for err in result["data_errors"])
    assert result["industry_median_ratios"]["current_ratio"] is not None


def test_assess_fails_loudly_when_target_company_unavailable(monkeypatch):
    monkeypatch.setattr(fd.yf, "Ticker", lambda s: FakeTicker(fail=True))
    fd.fetch_company.cache_clear()
    with pytest.raises(ValueError, match="VZ"):
        fd.assess("VZ")
