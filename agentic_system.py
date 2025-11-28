import uuid
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END, add_messages, START
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from tools import get_industry_financial_data
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    industry_name: str
    company_name: str | None
    company_ratios: dict | None
    industry_median_ratios: dict | None
    messages: Annotated[List[BaseMessage], add_messages]
    final_risk_assessment_report: str | None
    validator_json: str | None

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
tools = [get_industry_financial_data]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState) -> AgentState:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

SYSTEM_PROMPT_VALIDATOR = """
You are the Financial Validation Agent. Your job is to evaluate whether the primary agent's financial-risk analysis is accurate, complete, consistent, and aligned with the provided financial ratios.

You MUST return your evaluation ONLY in JSON, in the following format:
{
  "accuracy_checks": [],
  "completeness_checks": [],
  "logic_flags": [],
  "bankruptcy_risk_validation": "",
  "summary_assessment": "",
  "final_score": 0
}

Evaluate:
- liquidity
- leverage
- profitability
- efficiency
- coverage
- cash flow
- Z-score
- overall bankruptcy logic

ONLY return JSON.
"""

def validator_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    last_ai_text = ""
    if messages:
        last_ai_text = getattr(messages[-1], "content", "")

    ratios = state.get("company_ratios", {})

    validation_prompt = f"""
FINANCIAL RATIOS:
{ratios}

MODEL ANALYSIS:
{last_ai_text}

FOLLOW THE VALIDATION RULES STRICTLY. RETURN ONLY JSON.
"""

    validator_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    validation_response = validator_llm.invoke([
        HumanMessage(content=SYSTEM_PROMPT_VALIDATOR),
        HumanMessage(content=validation_prompt)
    ])

    return {
        "messages": [validation_response],
        "final_risk_assessment_report": last_ai_text,
        "validator_json": validation_response.content
    }

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "call_tool"
    return "__end__"

tool_executor_node = ToolNode(tools=tools)

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("validator", validator_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "call_tool": "tool_executor",
        "__end__": "validator"
    }
)

workflow.add_edge("tool_executor", "agent")
workflow.add_edge("validator", END)

THREAD_ID = str(uuid.uuid4())
checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)

SYSTEM_PROMPT_AGENT = """
You are a financial risk assessment agent.
You analyze industries, detect potential risks, list threats, evaluate macro- and micro-financial factors,
and generate clear, structured risk reports.
Always be factual, analytical, and concise.
"""

def get_thread_id():
    return THREAD_ID
