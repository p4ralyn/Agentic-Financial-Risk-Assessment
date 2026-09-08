import pytest

import graph as g


def _state(score, attempts):
    return {"validation": {"final_score": score, "logic_flags": ["x"]},
            "validation_attempts": attempts}


def test_route_retries_when_score_below_threshold():
    assert g.route_after_validator(_state(50, 1)) == "retry"


def test_route_finishes_when_score_meets_threshold():
    assert g.route_after_validator(_state(70, 1)) == "done"


def test_route_finishes_when_attempts_exhausted():
    """A stubborn model must not loop forever and burn quota."""
    assert g.route_after_validator(_state(10, g.MAX_ATTEMPTS)) == "done"


def test_route_finishes_when_validation_missing():
    assert g.route_after_validator({"validation": None,
                                    "validation_attempts": 1}) == "done"


def test_retry_loop_terminates_with_always_rejecting_validator():
    """The bound is what makes the loop safe; assert it directly."""
    state = {"validation": {"final_score": 0, "logic_flags": []},
             "validation_attempts": 0}
    hops = 0
    while g.route_after_validator(state) == "retry" and hops < 50:
        hops += 1
        state["validation_attempts"] += 1
    assert hops == g.MAX_ATTEMPTS, "must stop after MAX_ATTEMPTS retries"


def test_fetch_node_populates_ratios(monkeypatch):
    monkeypatch.setattr(g, "assess", lambda t: {
        "ticker": t, "company_name": "Fixture Corp", "industry": "Telecommunications",
        "peer_tickers": ["T"], "company_ratios": {"current_ratio": 2.0},
        "industry_median_ratios": {"current_ratio": 1.5}, "z": 2.8,
        "band": "grey", "qualitative": {}, "data_errors": [],
    })
    out = g.fetch_financials({"ticker": "VZ", "company_ratios": None,
                              "messages": []})
    assert out["company_ratios"] == {"current_ratio": 2.0}
    assert out["industry_median_ratios"] == {"current_ratio": 1.5}


def test_fetch_node_skips_when_ratios_already_present_for_ticker():
    """Follow-up chat turns must not refetch."""
    state = {"ticker": "VZ", "company_ratios": {"current_ratio": 2.0},
             "fetched_ticker": "VZ", "messages": []}
    assert g.fetch_financials(state) == {}


def test_validator_output_does_not_enter_messages():
    """v2 appended the validator JSON to messages, polluting every later turn."""
    class FakeValidator:
        def invoke(self, _):
            return g.ValidationReport(
                accuracy_checks=[], completeness_checks=[], logic_flags=[],
                bankruptcy_risk_validation="ok", summary_assessment="ok",
                final_score=90)

    out = g.make_validator_node(FakeValidator())({
        "messages": [], "company_ratios": {}, "industry_median_ratios": {},
        "final_risk_assessment_report": "report text", "validation_attempts": 0,
    })
    assert "messages" not in out
    assert out["validation"]["final_score"] == 90
    assert out["validation_attempts"] == 1


def test_correction_node_names_the_specific_flags():
    out = g.correction_node({"validation": {"logic_flags": ["claimed strong solvency"]}})
    assert "claimed strong solvency" in out["messages"][0].content


class _FakeLLM:
    """Stands in for the analyst. Counts invocations so the retry bound is
    observable end to end."""
    def __init__(self, counter):
        self.counter = counter

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        self.counter["agent"] += 1
        return AIMessage(content=f"report v{self.counter['agent']}")


class _FakeValidator:
    def __init__(self, counter, score):
        self.counter = counter
        self.score = score

    def invoke(self, messages):
        self.counter["validator"] += 1
        return g.ValidationReport(final_score=self.score,
                                  logic_flags=["claimed strong solvency"])


@pytest.fixture
def stub_assess(monkeypatch):
    monkeypatch.setattr(g, "assess", lambda t: {
        "ticker": t, "company_name": "Fixture Corp",
        "industry": "Telecommunications", "peer_tickers": ["T"],
        "company_ratios": {"current_ratio": 2.0},
        "industry_median_ratios": {"current_ratio": 1.5},
        "z": 1.2, "band": "distress", "qualitative": {}, "data_errors": [],
    })


def test_graph_runs_end_to_end_when_validation_passes(stub_assess):
    counter = {"agent": 0, "validator": 0}
    app = g.build_graph(llm=_FakeLLM(counter),
                        validator_llm=_FakeValidator(counter, score=95))
    out = app.invoke(g.initial_state("VZ", "assess"),
                     config={"configurable": {"thread_id": "pass"},
                             "recursion_limit": 25})
    assert out["final_risk_assessment_report"] == "report v1"
    assert out["validation"]["final_score"] == 95
    assert counter["agent"] == 1, "a passing report must not be retried"
    assert out["company_ratios"] == {"current_ratio": 2.0}


def test_graph_stops_retrying_against_an_always_rejecting_validator(stub_assess):
    """Without the bound this loops until recursion_limit and burns quota."""
    counter = {"agent": 0, "validator": 0}
    app = g.build_graph(llm=_FakeLLM(counter),
                        validator_llm=_FakeValidator(counter, score=5))
    out = app.invoke(g.initial_state("VZ", "assess"),
                     config={"configurable": {"thread_id": "reject"},
                             "recursion_limit": 50})
    assert counter["validator"] == g.MAX_ATTEMPTS
    assert out["validation_attempts"] == g.MAX_ATTEMPTS
    assert out["final_risk_assessment_report"], "the best attempt is still returned"
