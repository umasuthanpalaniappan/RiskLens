"""RiskLens Streamlit dashboard.

Single-file, read-only dashboard over the three JSON artifacts produced by
the pipeline: results/lstm_results.json, results/gnn_results.json and
results/portfolio_weights.json. No browser storage is used -- every reload
re-reads the on-disk JSON, so the dashboard always reflects the latest
committed results (e.g. after the daily GitHub Actions run).

Run with: streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

REGIME_STYLE = {
    "calm": {"color": "#1a7f37", "emoji": "🟢", "label": "CALM"},
    "normal": {"color": "#9a6700", "emoji": "🟡", "label": "NORMAL"},
    "crisis": {"color": "#cf222e", "emoji": "🔴", "label": "CRISIS"},
}


def load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


st.set_page_config(page_title="RiskLens", page_icon="📈", layout="wide")

lstm_results = load_json(config.LSTM_RESULTS_PATH)
gnn_results = load_json(config.GNN_RESULTS_PATH)
portfolio_results = load_json(config.PORTFOLIO_RESULTS_PATH)

title_col, badge_col = st.columns([4, 1])
with title_col:
    st.title("📈 RiskLens")
    st.caption("LSTM + dynamic-GNN volatility forecasting and minimum-variance portfolio construction")

with badge_col:
    if portfolio_results:
        regime = portfolio_results.get("regime", "normal")
        style = REGIME_STYLE.get(regime, REGIME_STYLE["normal"])
        st.markdown(
            f"""
            <div style="text-align:center; padding:14px; border-radius:10px;
                        border:2px solid {style['color']}; margin-top:10px;">
                <div style="font-size:1.6rem;">{style['emoji']}</div>
                <div style="font-weight:700; color:{style['color']}; letter-spacing:1px;">
                    {style['label']}
                </div>
                <div style="font-size:0.75rem; color:gray;">today's regime</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("No portfolio results yet")

st.divider()

if not (lstm_results and gnn_results and portfolio_results):
    st.error(
        "Missing results files. Run scripts/run_lstm.py, scripts/run_gnn.py and "
        "scripts/run_portfolio.py first (see README.md)."
    )
    st.stop()

tickers = portfolio_results["tickers"]
weights = portfolio_results["weights"]
predicted_vol = portfolio_results["predicted_volatility"]
gnn_alpha = gnn_results.get("alpha", {})

# --- Portfolio weights + summary table -----------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Portfolio weights")
    weights_df = pd.DataFrame(
        {"ticker": tickers, "weight": [weights[t] for t in tickers]}
    ).sort_values("weight", ascending=False)
    chart_kind = st.radio("Chart type", ["Bar", "Pie"], horizontal=True, label_visibility="collapsed")
    if chart_kind == "Bar":
        fig = px.bar(weights_df, x="ticker", y="weight", text_auto=".1%", color="weight",
                     color_continuous_scale="Blues")
        fig.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False, xaxis_title="", yaxis_title="Weight")
    else:
        fig = px.pie(weights_df, names="ticker", values="weight", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Stock-level detail")
    table_df = pd.DataFrame(
        {
            "Stock": tickers,
            "Predicted Volatility": [predicted_vol[t] for t in tickers],
            "Portfolio Weight": [weights[t] for t in tickers],
            "GNN Gate Alpha": [gnn_alpha.get(t) for t in tickers],
        }
    ).sort_values("Portfolio Weight", ascending=False)
    st.dataframe(
        table_df.style.format(
            {"Predicted Volatility": "{:.1%}", "Portfolio Weight": "{:.1%}", "GNN Gate Alpha": "{:.2f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# --- Direction accuracy comparison ---------------------------------------
st.subheader("Direction accuracy: LSTM vs. LSTM + GNN")
acc_df = pd.DataFrame(
    {
        "model": ["LSTM (baseline)", "LSTM + GNN"],
        "direction_accuracy": [
            lstm_results.get("test_direction_accuracy"),
            gnn_results.get("test_direction_accuracy_blended"),
        ],
    }
)
fig_acc = go.Figure(
    go.Bar(
        x=acc_df["model"],
        y=acc_df["direction_accuracy"],
        text=[f"{v:.1%}" if v is not None else "n/a" for v in acc_df["direction_accuracy"]],
        textposition="outside",
        marker_color=["#8c8c8c", "#1a7f37"],
    )
)
fig_acc.update_layout(yaxis_tickformat=".0%", yaxis_title="Test-set direction accuracy", xaxis_title="")
st.plotly_chart(fig_acc, use_container_width=True)

st.caption(
    f"Direction accuracy = fraction of test-set days where the model correctly called whether "
    f"realised volatility over the next {config.HORIZON} days would rise or fall."
)

st.divider()
st.caption(f"Last updated: {portfolio_results.get('timestamp', 'unknown')}  ·  "
           f"as of {portfolio_results.get('as_of_date', 'unknown')}")
