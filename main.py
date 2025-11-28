from tools import get_industry_financial_data
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, AIMessage
import datetime
import os

from agentic_system import app, AgentState, SYSTEM_PROMPT_AGENT, THREAD_ID

if __name__ == "__main__":

    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found. Please export it.")
        exit()

    industry_to_assess = "Telecommunications"

    initial_messages = [
        HumanMessage(content=SYSTEM_PROMPT_AGENT),
        HumanMessage(content=f"Assess the financial risk of the {industry_to_assess} industry.")
    ]

    initial_state = {
        "industry_name": industry_to_assess,
        "company_ratios": None,
        "industry_median_ratios": None,
        "messages": initial_messages,
        "final_risk_assessment_report": None
    }

    final_state = None

    for s in app.stream(initial_state, config={"configurable": {"thread_id": THREAD_ID}}):
        print(s)
        final_state = s

    if final_state:
        msgs = (
            final_state.get("messages")
            or final_state.get("agent", {}).get("messages")
        )

        if msgs:
            final_ai = None
            for msg in reversed(msgs):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    final_ai = msg
                    break

            if final_ai:
                print("\n--- FINAL RISK ASSESSMENT REPORT ---")
                print(final_ai.content)
            else:
                print("\n--- Could not find final AI report. ---")
    print("\n--- Done ---")
