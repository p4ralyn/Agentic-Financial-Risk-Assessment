import uuid

import streamlit as st

from graph import build_graph, initial_state

st.set_page_config(page_title="Financial risk", page_icon="📉", layout="wide")

BANDS = {
    "distress": ("#A33B32", "Elevated risk"),
    "grey": ("#B08334", "Mixed signals"),
    "safe": ("#2F6F5E", "Low risk"),
    "unknown": ("#6B7A88", "Risk not scored"),
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root { --ink:#101A24; --rule:#D3DAE0; --muted:#5A6B7B; }

html, body, [class*="css"] { font-family:'IBM Plex Sans',system-ui,sans-serif; }

.identity { display:flex; align-items:baseline; gap:.75rem; flex-wrap:wrap;
  border-bottom:1px solid var(--rule); padding-bottom:.5rem; margin-bottom:1.5rem; }
.identity .name { font-family:'Newsreader',Georgia,serif; font-size:1.6rem;
  color:var(--ink); }
.identity .meta { color:var(--muted); font-size:.9rem;
  border-left:1px solid var(--rule); padding-left:.75rem; }

.verdict { display:flex; align-items:baseline; justify-content:space-between;
  gap:1rem; flex-wrap:wrap; margin-bottom:.25rem; }
.verdict .label { font-family:'Newsreader',Georgia,serif; font-size:2.6rem;
  line-height:1.1; }
.verdict .z { font-size:1rem; color:var(--muted); }
.verdict .z b { font-family:'Newsreader',Georgia,serif; font-size:2rem;
  color:var(--ink); font-weight:500; font-variant-numeric:tabular-nums; }

.scale { position:relative; height:8px; border-radius:4px; margin:1.5rem 0 .4rem;
  background:linear-gradient(90deg,#A33B32 0%,#A33B32 36.2%,#B08334 36.2%,
    #B08334 59.8%,#2F6F5E 59.8%,#2F6F5E 100%); }
.marker { position:absolute; top:-6px; width:2px; height:20px; background:var(--ink);
  transition:left .6s cubic-bezier(.22,.61,.36,1); }
.marker::after { content:''; position:absolute; left:-4px; top:-4px; width:10px;
  height:10px; border-radius:50%; background:var(--ink); }
.ticks { display:flex; justify-content:space-between; color:var(--muted);
  font-size:.78rem; font-variant-numeric:tabular-nums; }

.report { font-family:'Newsreader',Georgia,serif; font-size:1.05rem;
  line-height:1.65; max-width:68ch; }

.empty { font-family:'Newsreader',Georgia,serif; font-size:1.15rem;
  color:var(--muted); max-width:52ch; line-height:1.6; }

@media (prefers-reduced-motion:reduce) { .marker { transition:none; } }
</style>
""", unsafe_allow_html=True)

if "thread_id" not in st.session_state:
    # One thread per browser session. A module-level id would be shared by
    # every visitor to a container, leaking one person's history into
    # another's context.
    st.session_state.thread_id = str(uuid.uuid4())
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "result" not in st.session_state:
    st.session_state.result = None

with st.sidebar:
    st.markdown("### Assess a company")
    ticker = st.text_input("Ticker", value="", placeholder="VZ").strip().upper()
    question = st.text_area(
        "Question",
        value="Assess this company's bankruptcy risk against its industry peers.",
        height=90,
    )
    run = st.button("Assess risk", type="primary", use_container_width=True)

if run and not ticker:
    st.warning("Enter a ticker symbol to run an assessment.")
elif run:
    with st.spinner(f"Fetching {ticker} and its peer group…"):
        try:
            st.session_state.result = st.session_state.graph.invoke(
                initial_state(ticker, question),
                config={"configurable": {"thread_id": st.session_state.thread_id},
                        "recursion_limit": 25},
            )
        except ValueError as exc:
            st.session_state.result = None
            st.error(str(exc))
        except Exception as exc:
            st.session_state.result = None
            st.error(f"The assessment could not be completed: {exc}")

result = st.session_state.result

if result is None:
    st.markdown("<div class='identity'><span class='name'>Financial risk</span>"
                "</div>", unsafe_allow_html=True)
    st.markdown("<p class='empty'>Enter a ticker to assess bankruptcy risk "
                "against its industry peers.</p>", unsafe_allow_html=True)
    st.stop()

band = result.get("z_band") or "unknown"
colour, verdict = BANDS[band]
z = result.get("z_score")

st.markdown(
    f"<div class='identity'><span class='name'>{result.get('company_name')}</span>"
    f"<span class='meta'>{result.get('ticker')} &nbsp; "
    f"{result.get('industry_name') or 'Industry unmapped'}</span></div>",
    unsafe_allow_html=True)

z_text = f"<b>{z:.2f}</b>" if z is not None else "<b>—</b>"
st.markdown(
    f"<div class='verdict'><span class='label' style='color:{colour}'>{verdict}"
    f"</span><span class='z'>Altman Z {z_text}</span></div>",
    unsafe_allow_html=True)

# Marker position: 0 at Z=0, full width at Z=5, clamped. The gradient stops
# at 36.2% and 59.8% are 1.81/5 and 2.99/5, so the bands line up with the ticks.
pct = 0.0 if z is None else max(0.0, min(z / 5.0, 1.0)) * 100
st.markdown(
    f"<div class='scale'><div class='marker' style='left:{pct:.1f}%'></div></div>"
    f"<div class='ticks'><span>0</span><span>1.81 distress</span>"
    f"<span>2.99 grey</span><span>5+ safe</span></div>",
    unsafe_allow_html=True)

st.markdown("")

company = result.get("company_ratios") or {}
peers = result.get("industry_median_ratios") or {}
if company:
    rows = []
    for name in sorted(company):
        mine, theirs = company[name], peers.get(name)
        delta = None if mine is None or theirs is None else mine - theirs
        rows.append({
            "Ratio": name.replace("_", " ").capitalize(),
            "Company": "—" if mine is None else round(mine, 3),
            "Peer median": "—" if theirs is None else round(theirs, 3),
            "Δ": "—" if delta is None else f"{delta:+.3f}",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("A dash means the ratio is not computable from the filed "
               "statements — most often because equity is zero or negative.")

report = result.get("final_risk_assessment_report")
if report:
    st.markdown(f"<div class='report'>{report}</div>", unsafe_allow_html=True)

validation = result.get("validation")
if validation:
    score = validation.get("final_score", 0)
    attempts = result.get("validation_attempts", 0)
    if score >= 70:
        when = "the first attempt" if attempts <= 1 else f"attempt {attempts}"
        st.success(f"Validated {score}/100 on {when}.")
    else:
        st.warning(f"Failed validation at {score}/100 after {attempts} attempts. "
                   "The report is shown with the validator's objections.")
    flags = validation.get("logic_flags") or []
    if flags:
        with st.expander("What the validator flagged"):
            for flag in flags:
                st.markdown(f"- {flag}")

for err in result.get("data_errors") or []:
    st.caption(f"Peer data unavailable — {err}")
