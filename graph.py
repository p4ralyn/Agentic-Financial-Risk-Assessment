"""Agent graph.

    START -> fetch_financials -> agent <-> tool_executor -> validator -+-> agent
                                                                       +-> END

fetch_financials runs before the model, so the ratios the validator checks
against are guaranteed present. In v2 the ratios came from Streamlit number
inputs defaulting to 0.0, so the validator compared the report to zeros.
"""
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from financial_data import assess

load_dotenv()

VALIDATION_THRESHOLD = 70
MAX_ATTEMPTS = 3


class AgentState(TypedDict):
    ticker: str
    company_name: str | None
    industry_name: str | None
    company_ratios: dict | None
    industry_median_ratios: dict | None
    peer_tickers: list[str]
    qualitative_context: dict | None
    z_score: float | None
    z_band: str | None
    fetched_ticker: str | None
    messages: Annotated[List[BaseMessage], add_messages]
    final_risk_assessment_report: str | None
    validation: dict | None
    validation_attempts: int
    data_errors: list[str]


class ValidationReport(BaseModel):
    accuracy_checks: List[str] = Field(default_factory=list)
    completeness_checks: List[str] = Field(default_factory=list)
    logic_flags: List[str] = Field(default_factory=list)
    bankruptcy_risk_validation: str = ""
    summary_assessment: str = ""
    final_score: int = 0


SYSTEM_PROMPT_AGENT = """You are a financial risk analyst.

You are given a company's computed financial ratios and the median ratios of
its industry peers. Argue from the gap between them.

Cover liquidity, leverage, coverage, profitability, efficiency, and the
Altman Z-score, then give an overall bankruptcy risk verdict.

A ratio shown as null is NOT computable, which is not the same as good. In
particular, a null debt-to-equity or return on equity means shareholders'
equity is zero or negative -- treat that as severe distress, never strength.

Be factual and concise. Cite the numbers you were given; do not invent any.
"""

SYSTEM_PROMPT_VALIDATOR = """You are a financial validation agent.

Check the analyst's report against the ratios actually supplied. Flag any
claim the numbers do not support, any material ratio left unaddressed, and
any internal contradiction (for example calling solvency strong while
debt-to-equity is elevated).

Score 0-100 for how well the report is supported by the data.
"""


@tool
def get_company_financials(ticker: str) -> dict:
    """Fetch computed financial ratios and peer medians for a stock ticker."""
    return assess(ticker)


TOOLS = [get_company_financials]


def fetch_financials(state: AgentState) -> dict:
    """Deterministic prefetch. No-ops on follow-up turns for the same ticker."""
    ticker = state.get("ticker")
    if not ticker:
        return {}
    if state.get("fetched_ticker") == ticker and state.get("company_ratios"):
        return {}

    data = assess(ticker)
    summary = (
        f"Company: {data['company_name']} ({data['ticker']})\n"
        f"Industry: {data['industry']}\n"
        f"Peers: {', '.join(data['peer_tickers']) or 'none'}\n"
        f"Altman Z-score: {data['z']} ({data['band']})\n"
        f"Company ratios: {data['company_ratios']}\n"
        f"Industry median ratios: {data['industry_median_ratios']}\n"
        f"Qualitative context: {data['qualitative']}\n"
        f"Data gaps: {data['data_errors'] or 'none'}"
    )
    return {
        "company_name": data["company_name"],
        "industry_name": data["industry"],
        "company_ratios": data["company_ratios"],
        "industry_median_ratios": data["industry_median_ratios"],
        "peer_tickers": data["peer_tickers"],
        "qualitative_context": data["qualitative"],
        "z_score": data["z"],
        "z_band": data["band"],
        "data_errors": data["data_errors"],
        "fetched_ticker": ticker,
        "messages": [HumanMessage(content=summary)],
    }


def make_agent_node(llm_with_tools):
    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        update = {"messages": [response]}
        if isinstance(response, AIMessage) and not response.tool_calls:
            update["final_risk_assessment_report"] = response.content
        return update
    return agent_node


def make_validator_node(validator_llm):
    def validator_node(state: AgentState) -> dict:
        prompt = (
            f"COMPANY RATIOS:\n{state.get('company_ratios')}\n\n"
            f"INDUSTRY MEDIAN RATIOS:\n{state.get('industry_median_ratios')}\n\n"
            f"ANALYST REPORT:\n{state.get('final_risk_assessment_report')}"
        )
        report = validator_llm.invoke([
            HumanMessage(content=SYSTEM_PROMPT_VALIDATOR),
            HumanMessage(content=prompt),
        ])
        # Deliberately no "messages" key: v2 appended this JSON to the
        # transcript, so it became context for every subsequent turn.
        return {
            "validation": report.model_dump(),
            "validation_attempts": state.get("validation_attempts", 0) + 1,
        }
    return validator_node


def route_after_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "call_tool"
    return "validate"


def route_after_validator(state: AgentState) -> str:
    validation = state.get("validation")
    if not validation:
        return "done"
    if state.get("validation_attempts", 0) >= MAX_ATTEMPTS:
        return "done"
    if validation.get("final_score", 0) < VALIDATION_THRESHOLD:
        return "retry"
    return "done"


def correction_node(state: AgentState) -> dict:
    flags = (state.get("validation") or {}).get("logic_flags", [])
    return {"messages": [HumanMessage(content=(
        "Your report failed validation. Fix these specific problems and "
        "reissue the full report:\n- " + "\n- ".join(flags or ["unsupported claims"])
    ))]}


def build_graph(llm=None, validator_llm=None):
    llm = llm or ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    validator_llm = validator_llm or ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=0
    ).with_structured_output(ValidationReport)

    workflow = StateGraph(AgentState)
    workflow.add_node("fetch", fetch_financials)
    workflow.add_node("agent", make_agent_node(llm.bind_tools(TOOLS)))
    workflow.add_node("tool_executor", ToolNode(tools=TOOLS))
    workflow.add_node("validator", make_validator_node(validator_llm))
    workflow.add_node("correction", correction_node)

    workflow.add_edge(START, "fetch")
    workflow.add_edge("fetch", "agent")
    workflow.add_conditional_edges("agent", route_after_tools,
                                   {"call_tool": "tool_executor",
                                    "validate": "validator"})
    workflow.add_edge("tool_executor", "agent")
    workflow.add_conditional_edges("validator", route_after_validator,
                                   {"retry": "correction", "done": END})
    workflow.add_edge("correction", "agent")

    return workflow.compile(checkpointer=MemorySaver())


def initial_state(ticker: str, question: str) -> dict:
    return {
        "ticker": ticker.upper(),
        "messages": [
            HumanMessage(content=SYSTEM_PROMPT_AGENT),
            HumanMessage(content=question),
        ],
        "validation_attempts": 0,
        "data_errors": [],
    }
