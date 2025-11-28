import streamlit as st
from agentic_system import SYSTEM_PROMPT_AGENT, app, get_thread_id
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Financial Risk AI", page_icon="📈")

st.title("Financial Risk Assessment AI")

st.sidebar.header("Configure Your Analysis")

industry_options = [
    "Telecommunications", "Technology", "Automotive", "Banking",
    "Pharmaceuticals", "Energy", "Retail", "Consumer Goods"
]

industry = st.sidebar.selectbox("Select Industry", industry_options)
company_name = st.sidebar.text_input("Company Name", "")

st.sidebar.subheader("Financial Ratios")

ratios_sidebar = {
    "current_ratio": st.sidebar.number_input("Current Ratio", value=0.0),
    "quick_ratio": st.sidebar.number_input("Quick Ratio", value=0.0),
    "cash_ratio": st.sidebar.number_input("Cash Ratio", value=0.0),
    "debt_to_equity": st.sidebar.number_input("Debt to Equity", value=0.0),
    "interest_coverage": st.sidebar.number_input("Interest Coverage", value=0.0),
    "operating_margin": st.sidebar.number_input("Operating Margin (%)", value=0.0),
    "net_margin": st.sidebar.number_input("Net Profit Margin (%)", value=0.0),
    "roa": st.sidebar.number_input("ROA (%)", value=0.0),
    "roe": st.sidebar.number_input("ROE (%)", value=0.0),
    "asset_turnover": st.sidebar.number_input("Asset Turnover", value=0.0),
    "inventory_turnover": st.sidebar.number_input("Inventory Turnover", value=0.0),
    "receivables_days": st.sidebar.number_input("Receivables Days", value=0),
    "payable_days": st.sidebar.number_input("Payable Days", value=0),
    "z_score": st.sidebar.number_input("Altman Z-Score", value=0.0),
    "ocf": st.sidebar.number_input("Operating Cash Flow (M)", value=0.0),
    "fcf": st.sidebar.number_input("Free Cash Flow (M)", value=0.0),
}

# Session state sync
st.session_state.industry = industry
st.session_state.company_name = company_name
st.session_state.company_ratios = ratios_sidebar

if "thread_id" not in st.session_state:
    st.session_state.thread_id = get_thread_id()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask questions or request analysis...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):

            # Build messages for LangGraph
            initial_messages = [
                HumanMessage(content=SYSTEM_PROMPT_AGENT),
                HumanMessage(content=f"Industry: {st.session_state.industry}"),
                HumanMessage(content=f"Company Name: {st.session_state.company_name}"),
                HumanMessage(content=f"Financial Ratios: {st.session_state.company_ratios}"),
                HumanMessage(content=prompt),
            ]

            # Build graph state
            initial_state = {
                "industry_name": st.session_state.industry,
                "company_name": st.session_state.company_name,
                "company_ratios": st.session_state.company_ratios,
                "industry_median_ratios": None,
                "messages": initial_messages,
                "final_risk_assessment_report": None,
            }

            # Stream agent
            ai_output_text = ""
            for s in app.stream(
                initial_state,
                config={"configurable": {"thread_id": st.session_state.thread_id}}
            ):
                if "agent" in s:
                    msg = s["agent"]["messages"][-1]
                    if isinstance(msg, AIMessage):
                        ai_output_text = msg.content

            # Display result
            st.markdown(ai_output_text)
            st.session_state.messages.append({"role": "assistant", "content": ai_output_text})

            try:
                full_state = app.get_state(config={"configurable": {"thread_id": st.session_state.thread_id}})
                validator_json = full_state.values.get("validator_json", None)
            except:
                validator_json = None

            if validator_json:
                st.markdown("### 🧮 Validator Report")
                with st.expander("View Validator JSON"):
                    st.code(validator_json, language="json")
