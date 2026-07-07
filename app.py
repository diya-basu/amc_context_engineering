import concurrent.futures
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

import config, retrieval, graph_store, faiss_store
from compare_view import render_traditional_panel, render_contextgraph_panel

faiss_store.INDEXES_DIR = config.FAISS_DIR

# --- NEW: Custom CSS to match the target UI ---
CUSTOM_CSS = """
<style>
/* Faint dot grid background */
.stApp {
    background-color: #F8F7F3;
    background-image: radial-gradient(#E5E0D8 1px, transparent 0);
    background-size: 24px 24px;
}

/* Make top Streamlit header transparent */
header {
    background-color: transparent !important;
}

/* Style the standard Streamlit tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    padding-top: 1rem;
    padding-bottom: 0.5rem;
    color: #8A8378;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #A8412C !important;
    border-bottom-color: #A8412C !important;
}

/* Style the main search input box */
.stTextInput input {
    background-color: #EFEAE0 !important;
    border: 1px solid #E7E1D4 !important;
    border-radius: 8px !important;
    color: #5C574C !important;
}
.stTextInput input:focus {
    border-color: #A8412C !important;
    box-shadow: 0 0 0 1px #A8412C !important;
}

/* Style standard (secondary) buttons - Example buttons */
.stButton > button[kind="secondary"] {
    background-color: #FFFFFF;
    border: 1px solid #E7E1D4;
    border-radius: 8px;
    color: #5C574C;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #A8412C;
    color: #A8412C;
    background-color: #FFF;
}

/* Style primary button - Run query button */
.stButton > button[kind="primary"] {
    background-color: #A8412C;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover {
    background-color: #8C3523;
    color: white;
}

/* Metrics styling adjustments */
[data-testid="stMetricValue"] { color: #231F1C; }
[data-testid="stMetricLabel"] { color: #8A8378; }
</style>
"""

@st.cache_resource
def get_master_store():
    return faiss_store.BrochureFAISSStore("amc_master")

if "comparisons" not in st.session_state:
    st.session_state.comparisons = []
if "compare_query" not in st.session_state:
    st.session_state.compare_query = ""

st.set_page_config(page_title="MF Context Engine", layout="wide")

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
tab_compare, tab_analytics = st.tabs(["⚖️ Compare", "📊 Analytics"])

with tab_compare:
    st.caption("🟢 Backend: Live")

    col_ex1, col_ex2, col_q, col_run = st.columns([1.3, 1.3, 3.4, 1])
    if col_ex1.button("Example: concentration", use_container_width=True):
        st.session_state.compare_query = "What's our exposure to Adani Group across all schemes?"
    if col_ex2.button("Example: compliance", use_container_width=True):
        st.session_state.compare_query = "Which schemes need exit load disclosure updates after SEBI's latest circular?"
    query = col_q.text_input("query", value=st.session_state.compare_query,
                              label_visibility="collapsed",
                              placeholder="Type your query and run it against both search modes…")
    run = col_run.button("Run query", type="primary", use_container_width=True)

    if run and query:
        try:
            store = get_master_store()
        except FileNotFoundError:
            store = None
            st.warning("No FAISS index found — run `python build_index.py` first.")

        if store is not None:
            col_left, col_right = st.columns(2)
            ph_trad = col_left.empty()
            ph_ctx = col_right.empty()
            ph_trad.info("Running traditional vector search…")
            ph_ctx.info("Running graph + vector retrieval…")

            results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(retrieval.traditional_rag, query, store): "traditional",
                    pool.submit(retrieval.hybrid_graphrag, query, store): "hybrid",
                }
                for future in concurrent.futures.as_completed(futures):
                    kind = futures[future]
                    results[kind] = future.result()
                    if kind == "traditional":
                        with ph_trad.container():
                            components.html(render_traditional_panel(results["traditional"]),
                                             height=520, scrolling=True)
                    else:
                        entity_summary = graph_store.get_entity_type_summary(results["hybrid"]["active_labels"])
                        with ph_ctx.container():
                            components.html(render_contextgraph_panel(results["hybrid"], entity_summary),
                                             height=520, scrolling=True)

            st.session_state.comparisons.append({
                "query": query,
                "traditional_time": results["traditional"]["total_time"],
                "hybrid_time": results["hybrid"]["total_time"],
                "traditional_relevancy": retrieval.relevancy_score(results["traditional"]),
                "hybrid_relevancy": retrieval.relevancy_score(results["hybrid"]),
            })

with tab_analytics:
    st.subheader("Traditional vs. ContextGraph — run history")
    if not st.session_state.comparisons:
        st.info("Run a query in the Compare tab first.")
    else:
        df = pd.DataFrame(st.session_state.comparisons)
        st.dataframe(df, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg traditional latency", f"{df['traditional_time'].mean():.2f}s")
        c2.metric("Avg hybrid latency", f"{df['hybrid_time'].mean():.2f}s",
                   delta=f"{df['hybrid_time'].mean() - df['traditional_time'].mean():+.2f}s")
        c3.metric("Avg relevancy gain", f"{(df['hybrid_relevancy'] - df['traditional_relevancy']).mean():+.2f}")
        st.bar_chart(df.set_index("query")[["traditional_time", "hybrid_time"]])
        st.bar_chart(df.set_index("query")[["traditional_relevancy", "hybrid_relevancy"]])