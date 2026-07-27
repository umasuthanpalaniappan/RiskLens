"""RiskLens Streamlit dashboard.

Single-file, read-only dashboard. The Portfolio and Model Performance tabs
read only the three pipeline result JSONs (results/lstm_results.json,
results/gnn_results.json, results/portfolio_weights.json) as required. The
Market Overview tab additionally reads the local price/volume CSV cache
under data/raw/ (written by the training/portfolio scripts) to show a
correlation heatmap and per-stock volatility trends across the whole
universe -- it degrades gracefully with a friendly message if that cache
isn't present yet. No browser storage is used anywhere: every reload
re-reads on-disk files, so the dashboard always reflects the latest
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

from src import config, features  # noqa: E402

REGIME_STYLE = {
    "calm": {"color": "#1a7f37", "bg": "#dafbe1", "emoji": "🟢", "label": "CALM"},
    "normal": {"color": "#9a6700", "bg": "#fff8c5", "emoji": "🟡", "label": "NORMAL"},
    "crisis": {"color": "#cf222e", "bg": "#ffebe9", "emoji": "🔴", "label": "CRISIS"},
}

CUSTOM_CSS = """
<style>
.rl-kpi {
    border-radius: 12px; padding: 16px 18px; border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.06);
}
.rl-kpi .label { font-size: 0.78rem; color: gray; text-transform: uppercase; letter-spacing: 0.04em; }
.rl-kpi .value { font-size: 1.6rem; font-weight: 700; margin-top: 2px; }
.rl-badge {
    text-align:center; padding:14px; border-radius:12px; margin-top:4px;
}
.rl-badge .emoji { font-size:1.7rem; }
.rl-badge .label { font-weight:800; letter-spacing:1.5px; font-size:1.05rem; }
.rl-badge .sub { font-size:0.72rem; color:gray; margin-top:2px; }
</style>
"""


def load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_price_volume_cache():
    """Best-effort local read of the cached price/volume CSVs, or None if absent."""
    price_p, vol_p = Path(config.RAW_DATA_PATH), Path(config.RAW_VOLUME_PATH)
    if not (price_p.exists() and vol_p.exists()):
        return None, None
    prices = pd.read_csv(price_p, index_col="date", parse_dates=True)
    volumes = pd.read_csv(vol_p, index_col="date", parse_dates=True)
    return prices, volumes


st.set_page_config(page_title="RiskLens", page_icon="📈", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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
            <div class="rl-badge" style="border:2px solid {style['color']}; background:{style['bg']}22;">
                <div class="emoji">{style['emoji']}</div>
                <div class="label" style="color:{style['color']};">{style['label']}</div>
                <div class="sub">today's regime</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("No portfolio results yet")

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

# --- KPI row ---------------------------------------------------------------
weighted_vol = sum(weights[t] * predicted_vol[t] for t in tickers)
top_holding = max(weights, key=weights.get)

k1, k2, k3, k4 = st.columns(4)
for col, label, value in zip(
    (k1, k2, k3, k4),
    ("Holdings", "Weighted Avg. Predicted Vol", "Largest Holding", "As Of"),
    (
        str(len(tickers)),
        f"{weighted_vol:.1%}",
        f"{top_holding.replace('.NS', '')} ({weights[top_holding]:.0%})",
        portfolio_results.get("as_of_date", "n/a"),
    ),
):
    col.markdown(
        f'<div class="rl-kpi"><div class="label">{label}</div><div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.divider()

tab_portfolio, tab_models, tab_market = st.tabs(["📊 Portfolio", "🧠 Model Performance", "🗺️ Market Overview"])

# ============================================================================
# TAB 1: Portfolio
# ============================================================================
with tab_portfolio:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Portfolio weights")
        weights_df = pd.DataFrame(
            {
                "ticker": tickers,
                "weight": [weights[t] for t in tickers],
                "sector": [config.SECTOR_MAP.get(t, "Other") for t in tickers],
            }
        ).sort_values("weight", ascending=False)
        chart_kind = st.radio("Chart type", ["Bar", "Pie"], horizontal=True, label_visibility="collapsed")
        if chart_kind == "Bar":
            fig = px.bar(
                weights_df, x="ticker", y="weight", text_auto=".1%", color="sector",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Weight", legend_title="Sector")
        else:
            fig = px.pie(weights_df, names="ticker", values="weight", hole=0.45, color="sector",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sector allocation")
        sector_df = weights_df.groupby("sector", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
        fig_sector = px.bar(sector_df, x="sector", y="weight", text_auto=".1%", color="sector",
                             color_discrete_sequence=px.colors.qualitative.Set2)
        fig_sector.update_layout(showlegend=False, yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Weight")
        st.plotly_chart(fig_sector, use_container_width=True)

    with right:
        st.subheader("Stock-level detail")
        table_df = pd.DataFrame(
            {
                "Stock": tickers,
                "Sector": [config.SECTOR_MAP.get(t, "Other") for t in tickers],
                "Predicted Volatility": [predicted_vol[t] * 100 for t in tickers],
                "Portfolio Weight": [weights[t] * 100 for t in tickers],
                "GNN Gate Alpha": [gnn_alpha.get(t) for t in tickers],
            }
        ).sort_values("Portfolio Weight", ascending=False)
        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            height=460,
            column_config={
                "Predicted Volatility": st.column_config.NumberColumn("Predicted Volatility", format="%.1f%%"),
                "GNN Gate Alpha": st.column_config.NumberColumn("GNN Gate Alpha", format="%.2f"),
                "Portfolio Weight": st.column_config.ProgressColumn(
                    "Portfolio Weight", format="%.1f%%", min_value=0.0, max_value=float(max(weights.values()) * 100)
                ),
            },
        )
        st.caption(
            "GNN Gate Alpha is the model's learned trust in the graph-refined signal for that "
            "stock (0 = pure LSTM, 1 = pure graph-convolved prediction)."
        )

# ============================================================================
# TAB 2: Model Performance
# ============================================================================
with tab_models:
    st.subheader("Direction accuracy: LSTM vs. LSTM + GNN")
    acc_df = pd.DataFrame(
        {
            "model": ["LSTM (baseline)", "LSTM + GNN (blended)"],
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
    fig_acc.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="coin flip")
    fig_acc.update_layout(yaxis_tickformat=".0%", yaxis_title="Test-set direction accuracy", xaxis_title="")
    st.plotly_chart(fig_acc, use_container_width=True)
    st.caption(
        f"Direction accuracy = fraction of test-set days where the model correctly called whether "
        f"realised volatility over the next {config.HORIZON} days would rise or fall."
    )

    st.divider()
    st.subheader("GNN internals: blended vs. LSTM-only vs. GNN-only")
    breakdown_df = pd.DataFrame(
        {
            "component": ["LSTM-only head", "GNN-only head", "Blended (deployed)"],
            "direction_accuracy": [
                gnn_results.get("test_direction_accuracy_lstm_only"),
                gnn_results.get("test_direction_accuracy_gnn_only"),
                gnn_results.get("test_direction_accuracy_blended"),
            ],
            "mse": [
                gnn_results.get("test_mse_lstm_only"),
                gnn_results.get("test_mse_gnn_only"),
                gnn_results.get("test_mse_blended"),
            ],
        }
    )
    c1, c2 = st.columns(2)
    with c1:
        fig_b1 = px.bar(breakdown_df, x="component", y="direction_accuracy", text_auto=".1%")
        fig_b1.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Direction accuracy")
        st.plotly_chart(fig_b1, use_container_width=True)
    with c2:
        fig_b2 = px.bar(breakdown_df, x="component", y="mse", text_auto=".3f")
        fig_b2.update_layout(xaxis_title="", yaxis_title="Test MSE (vol ratio)")
        st.plotly_chart(fig_b2, use_container_width=True)

    st.divider()
    st.subheader("Per-stock GNN gate alpha")
    alpha_df = pd.DataFrame({"ticker": list(gnn_alpha.keys()), "alpha": list(gnn_alpha.values())}).sort_values(
        "alpha", ascending=False
    )
    fig_alpha = px.bar(alpha_df, x="ticker", y="alpha", color="alpha", color_continuous_scale="Purples")
    fig_alpha.add_hline(y=0.5, line_dash="dot", line_color="gray")
    fig_alpha.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="alpha (graph trust)")
    st.plotly_chart(fig_alpha, use_container_width=True)

    with st.expander("Raw training metadata"):
        m1, m2 = st.columns(2)
        m1.json({k: v for k, v in lstm_results.items() if k != "tickers"})
        m2.json({k: v for k, v in gnn_results.items() if k not in ("tickers", "alpha")})

# ============================================================================
# TAB 3: Market Overview
# ============================================================================
with tab_market:
    prices_cache, volumes_cache = load_price_volume_cache()
    if prices_cache is None:
        st.info(
            "No local price cache found at data/raw/. Run scripts/run_lstm.py, run_gnn.py or "
            "run_portfolio.py at least once to populate it, then reload this tab."
        )
    else:
        available = [t for t in tickers if t in prices_cache.columns]
        returns_cache = features.log_returns(prices_cache[available])

        st.subheader("Correlation structure (most recent 60-day window)")
        recent_window = returns_cache.iloc[-config.GRAPH_WINDOW:]
        corr = recent_window.corr()
        fig_corr = px.imshow(
            corr,
            x=[t.replace(".NS", "") for t in corr.columns],
            y=[t.replace(".NS", "") for t in corr.index],
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        fig_corr.update_layout(height=520)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption(
            "This is the same rolling correlation window that graph.py turns into the GNN's "
            "dynamic adjacency matrix and portfolio.py uses to build Sigma."
        )

        st.divider()
        st.subheader("Historical trends")
        LOOKBACK_OPTIONS = {"1Y": 252, "3Y": 756, "5Y": 1260, "10Y": 2520, "Max (~15Y)": None}
        ctrl_left, ctrl_right = st.columns([2, 1])
        with ctrl_left:
            default_selection = sorted(available, key=lambda t: -weights.get(t, 0))[:6]
            selected = st.multiselect(
                "Stocks to plot", options=available, default=default_selection,
                format_func=lambda t: t.replace(".NS", ""),
            )
        with ctrl_right:
            lookback_label = st.selectbox("Lookback", options=list(LOOKBACK_OPTIONS.keys()), index=0)
        lookback_days = LOOKBACK_OPTIONS[lookback_label]

        if selected:
            def _tail(frame: pd.DataFrame) -> pd.DataFrame:
                return frame if lookback_days is None else frame.iloc[-lookback_days:]

            price_frame = _tail(pd.DataFrame({t.replace(".NS", ""): prices_cache[t] for t in selected}))
            fig_price = px.line(price_frame, labels={"value": "Close price", "date": "Date"})
            fig_price.update_layout(legend_title="Stock", yaxis_title="Close price (INR)")
            st.plotly_chart(fig_price, use_container_width=True)

            vol_frame = _tail(
                pd.DataFrame({t.replace(".NS", ""): features.realized_vol(returns_cache[t]) for t in selected})
            )
            fig_vol = px.line(vol_frame, labels={"value": "Annualised realised vol", "date": "Date"})
            fig_vol.update_layout(yaxis_tickformat=".0%", legend_title="Stock", yaxis_title="Realised volatility")
            st.plotly_chart(fig_vol, use_container_width=True)
        st.caption(
            f"Price cache spans {prices_cache.index.min().date()} to {prices_cache.index.max().date()} "
            f"({len(prices_cache)} trading days)."
        )

        st.divider()
        st.subheader("Universe snapshot")
        snapshot_df = pd.DataFrame(
            {
                "Stock": [t.replace(".NS", "") for t in available],
                "Sector": [config.SECTOR_MAP.get(t, "Other") for t in available],
                "Last Close": [prices_cache[t].iloc[-1] for t in available],
                "Trailing Realised Vol": [features.realized_vol(returns_cache[t]).iloc[-1] for t in available],
            }
        ).sort_values("Sector")
        st.dataframe(
            snapshot_df.style.format({"Last Close": "{:.2f}", "Trailing Realised Vol": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.caption(
    f"Last updated: {portfolio_results.get('timestamp', 'unknown')}  ·  "
    f"as of {portfolio_results.get('as_of_date', 'unknown')}"
)
