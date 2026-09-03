import os
import json
import yaml
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.dashboard_data import (
    load_config_data,
    load_dataset_summary,
    load_experiment_manifest,
    load_model_metrics,
    load_final_test_metrics,
    load_threshold_analysis,
    load_spike_results
)
from src.rag_assistant import RiskPolicyRAG

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="SentinelPay | AI Risk Manager Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM FINTECH DARK SOC AESTHETIC (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    .stApp {
        background-color: #0b0f17;
        color: #c9d1d9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Styling */
    .hero-container {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid rgba(56, 139, 253, 0.25);
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
        margin-bottom: 4px;
    }
    
    .hero-subtitle {
        font-size: 15px;
        color: #8b949e;
        margin-bottom: 12px;
    }
    
    .badge-prototype {
        background: rgba(56, 139, 253, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(56, 139, 253, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-block;
    }

    .badge-ready {
        background: rgba(35, 134, 54, 0.15);
        color: #3fb950;
        border: 1px solid rgba(35, 134, 54, 0.4);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
    }

    /* KPI Cards */
    .kpi-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 15px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .kpi-card:hover {
        border-color: rgba(56, 139, 253, 0.5);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #8b949e;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #f0f6fc;
    }
    .kpi-subtext {
        font-size: 12px;
        color: #3fb950;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Value Pillars */
    .pillar-card {
        background: rgba(22, 27, 34, 0.5);
        border: 1px solid rgba(48, 54, 61, 0.6);
        border-radius: 10px;
        padding: 20px;
        height: 100%;
    }
    .pillar-num {
        font-size: 20px;
        font-weight: 800;
        color: #58a6ff;
        margin-bottom: 8px;
    }
    .pillar-title {
        font-size: 15px;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 6px;
    }
    .pillar-desc {
        font-size: 13px;
        color: #8b949e;
        line-height: 1.5;
    }

    /* Architecture Flow */
    .flow-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(13, 17, 23, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 10px;
        padding: 16px 20px;
        margin: 20px 0;
    }
    .flow-node {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 16px;
        text-align: center;
        font-size: 12px;
        font-weight: 700;
        color: #c9d1d9;
    }
    .flow-node.active {
        border-color: #58a6ff;
        color: #58a6ff;
        box-shadow: 0 0 10px rgba(88, 166, 255, 0.2);
    }
    .flow-arrow {
        color: #484f58;
        font-weight: 800;
        font-size: 16px;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background-color: rgba(210, 153, 34, 0.1);
        border-left: 3px solid #d29922;
        padding: 12px 16px;
        margin-bottom: 20px;
        border-radius: 6px;
        color: #e3b341;
        font-size: 13px;
        line-height: 1.5;
    }

    /* Badges */
    .badge-success {
        background-color: rgba(35, 134, 54, 0.2);
        color: #3fb950;
        border: 1px solid rgba(35, 134, 54, 0.4);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
    }
    .badge-danger {
        background-color: rgba(218, 54, 51, 0.2);
        color: #f85149;
        border: 1px solid rgba(218, 54, 51, 0.4);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
    }
    .badge-warning {
        background-color: rgba(210, 153, 34, 0.2);
        color: #e3b341;
        border: 1px solid rgba(210, 153, 34, 0.4);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# ARTIFACT LOADING VIA DASHBOARD DATA HELPERS
# ---------------------------------------------------------
config = load_config_data()
dataset_summary = load_dataset_summary()
manifest = load_experiment_manifest()
metrics = load_model_metrics()
test_metrics = load_final_test_metrics()
rag_engine = RiskPolicyRAG()

@st.cache_resource
def load_trained_model():
    model_path = 'models/winning_model.joblib'
    feat_path = 'models/feature_cols.joblib'
    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        return None, None
    model = joblib.load(model_path)
    feats = joblib.load(feat_path)
    return model, feats

winning_model, feature_cols = load_trained_model()

# ---------------------------------------------------------
# SIDEBAR NAVIGATION & SYSTEM STATUS
# ---------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0;">
    <div style="font-size: 24px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">🛡️ SentinelPay</div>
    <div style="font-size: 12px; color: #8b949e; font-weight: 600;">AI RISK COMMAND CENTER</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="background: rgba(35, 134, 54, 0.1); border: 1px solid rgba(35, 134, 54, 0.3); border-radius: 8px; padding: 10px; text-align: center; margin-bottom: 15px;">
    <span class="badge-ready">🟢 SYSTEM READY</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🔒 Active Experiment")
if manifest:
    st.sidebar.markdown(f"• **Dataset**: `Simulated Benchmark (60D)`")
    st.sidebar.markdown(f"• **Split**: `70 / 15 / 15 Chronological`")
    st.sidebar.markdown(f"• **Winning Model**: `{manifest['winning_model']}`")
    st.sidebar.markdown(f"• **Locked Threshold ($t^*$)**: `{manifest['locked_threshold']:.2f}`")
    st.sidebar.caption(f"Cost ratio: FP = ${manifest['fp_cost']:.2f} | FN = ${manifest['fn_cost']:.2f}")
else:
    st.sidebar.error("Experiment manifest not found. Run training pipeline.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 Command Navigation")

nav_choice = st.sidebar.radio(
    "Go to section:",
    [
        "📊 OVERVIEW",
        "📈 MODEL PERFORMANCE",
        "⚡ DECISION ENGINE",
        "🔍 LIVE SCORING",
        "🚨 THREAT MONITORING",
        "🤖 POLICY ASSISTANT"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Razorpay AI Risk Manager (Track 02) • Defense-Only Prototype")

# ---------------------------------------------------------
# SECTION 1: OVERVIEW
# ---------------------------------------------------------
if nav_choice == "📊 OVERVIEW":
    st.markdown("""
    <div class="hero-container">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div class="hero-title">SENTINELPAY AI RISK MANAGER</div>
                <div class="hero-subtitle">Real-time transaction risk scoring and fraud intelligence powered by machine learning.</div>
            </div>
            <div>
                <span class="badge-prototype">RESEARCH PROTOTYPE • DEFENSIVE AI</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <b>⚠️ SIMULATED DATASET & COMPETITION DISCLAIMER</b><br>
        SentinelPay uses the official transformed credit card transaction dataset from the Fraud Detection Handbook. 
        This prototype contains <b>zero</b> internal Razorpay payment data, merchant profiles, or proprietary rules. 
        Cost parameters ($1.0 FP / $10.0 FN) are hypothetical defensive modeling assumptions. Strictly defense-only.
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards Row
    if dataset_summary and test_metrics:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">TOTAL TRANSACTIONS</div>
                <div class="kpi-value">{dataset_summary['total_transactions']:,}</div>
                <div class="kpi-subtext">60 Benchmark Days</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">FRAUD RATE</div>
                <div class="kpi-value">{dataset_summary['fraud_percentage']:.3f}%</div>
                <div class="kpi-subtext">{dataset_summary['fraud_count']:,} Fraud Cases</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">HELD-OUT TEST PR-AUC</div>
                <div class="kpi-value">{test_metrics['pr_auc']:.4f}</div>
                <div class="kpi-subtext">Imbalance-Robust Metric</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">TOTAL TEST COST</div>
                <div class="kpi-value">${test_metrics['total_cost']:,.2f}</div>
                <div class="kpi-subtext">At Locked t* = 0.31</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Why SentinelPay?")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-num">01</div>
            <div class="pillar-title">TRANSACTION RISK</div>
            <div class="pillar-desc">Scores incoming payment transactions in real time using non-leaking behavioral and time-context signals.</div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-num">02</div>
            <div class="pillar-title">HISTORICAL INTELLIGENCE</div>
            <div class="pillar-desc">Leverages customer and terminal transaction history with mandatory 7-day reporting delays (t ≤ T - 7D) to eliminate data leakage.</div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-num">03</div>
            <div class="pillar-title">OPERATIONAL RESPONSE</div>
            <div class="pillar-desc">Translates probabilities into cost-minimized decisions, terminal fraud-spike alerts, and grounded analyst verification.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🔄 3-Layer Pipeline Architecture")
    st.markdown("""
    <div class="flow-container">
        <div class="flow-node">TRANSACTION T</div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node">RETROSPECTIVE FEATURE ENGINE</div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node active">XGBoost MODEL</div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node">RISK SCORE</div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node">LOCKED THRESHOLD (0.31)</div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node">ACTION / ALERT</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧠 15 Model Features Matrix")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown("#### Transaction Context")
        st.markdown("• `TX_AMOUNT`: Amount in dollars\n• `log_tx_amount`: Log-scaled amount\n• `hour`: Hour of day (0-23)\n• `day_of_week`: Day of week (0-6)\n• `TX_DURING_WEEKEND`: Weekend binary flag\n• `TX_DURING_NIGHT`: Night hour flag (0-5)")
    with fc2:
        st.markdown("#### Customer History (`closed='left'`)")
        st.markdown("• `CUSTOMER_ID_NB_TX_1DAY`: Past 1D tx count\n• `CUSTOMER_ID_AVG_AMOUNT_1DAY`: Past 1D mean amount\n• `CUSTOMER_ID_NB_TX_7DAY`: Past 7D tx count\n• `CUSTOMER_ID_AVG_AMOUNT_7DAY`: Past 7D mean amount\n• `CUSTOMER_ID_NB_TX_30DAY`: Past 30D tx count\n• `CUSTOMER_ID_AVG_AMOUNT_30DAY`: Past 30D mean amount")
    with fc3:
        st.markdown("#### Terminal History (7-Day Delay Offset)")
        st.markdown("• `TERMINAL_ID_NB_TX_1DAY`: Past 1D terminal tx count\n• `TERMINAL_ID_RISK_7DAY_DELAYED`: Terminal fraud rate ($t \\le T - 7\\text{D}$)\n• `TERMINAL_ID_RISK_30DAY_DELAYED`: Terminal fraud rate ($t \\le T - 7\\text{D}$)")

    st.markdown("---")
    st.markdown("### 🔒 Methodology & Model Integrity Guarantees")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.markdown("<span class=\"badge-ready\">✓ 70/15/15 Temporal Split</span>", unsafe_allow_html=True)
    m2.markdown("<span class=\"badge-ready\">✓ No Future Data Leakage</span>", unsafe_allow_html=True)
    m3.markdown("<span class=\"badge-ready\">✓ Target Independence</span>", unsafe_allow_html=True)
    m4.markdown("<span class=\"badge-ready\">✓ Validation Cost Tuning</span>", unsafe_allow_html=True)
    m5.markdown("<span class=\"badge-ready\">✓ Held-Out Test Evaluation</span>", unsafe_allow_html=True)
    m6.markdown("<span class=\"badge-ready\">✓ Seed 42 Deterministic</span>", unsafe_allow_html=True)

# ---------------------------------------------------------
# SECTION 2: MODEL PERFORMANCE
# ---------------------------------------------------------
elif nav_choice == "📈 MODEL PERFORMANCE":
    st.markdown("## 📈 Model Performance & Validation Selection")
    st.markdown("### Why did we choose XGBoost?")

    if metrics and 'validation_results' in metrics:
        val_res = metrics['validation_results']
        
        # Prepare comparison data
        comp_rows = []
        for m_name, res in val_res.items():
            m_data = res['val_metrics']
            comp_rows.append({
                'Model': m_name,
                'PR-AUC': m_data['pr_auc'],
                'Precision': m_data['precision'],
                'Recall': m_data['recall'],
                'F1 Score': m_data['f1'],
                'Validation Cost ($)': m_data['cost'],
                'Optimal Threshold': res['optimal_threshold']
            })
        comp_df = pd.DataFrame(comp_rows)

        # Plotly Comparison Chart
        c_left, c_right = st.columns([1.2, 1])
        with c_left:
            fig_comp = px.bar(
                comp_df,
                x='Model',
                y=['PR-AUC', 'Precision', 'Recall', 'F1 Score'],
                barmode='group',
                title='Validation Performance Across Model Candidates',
                color_discrete_sequence=['#58a6ff', '#3fb950', '#d29922', '#bc8cff']
            )
            fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9')
            st.plotly_chart(fig_comp, use_container_width=True)

        with c_right:
            fig_cost_bar = px.bar(
                comp_df,
                x='Model',
                y='Validation Cost ($)',
                title='Validation Business Cost Comparison (Lower is Better)',
                color='Model',
                color_discrete_map={'Logistic Regression': '#f85149', 'Random Forest': '#d29922', 'XGBoost': '#3fb950'},
                text_auto='.2f'
            )
            fig_cost_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9', showlegend=False)
            st.plotly_chart(fig_cost_bar, use_container_width=True)

        st.markdown("""
        <div style="background: rgba(35, 134, 54, 0.1); border: 1px solid rgba(35, 134, 54, 0.3); border-radius: 8px; padding: 14px; margin-bottom: 20px;">
            <b>🏆 WINNER SELECTION REASONING:</b><br>
            XGBoost achieved the lowest total business cost (<b>$2,510.00</b> vs Random Forest $2,646.00 and Logistic Regression $3,097.00) on the 15% Validation set. 
            It was therefore selected as the winning classifier and locked for final test set evaluation.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏆 Final Held-Out Test Set Results (Locked Model)")
    
    if test_metrics:
        tm = test_metrics
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.metric("Test PR-AUC", f"{tm['pr_auc']:.4f}", help="Primary imbalance-robust metric")
        with t2:
            st.metric("Test Precision", f"{tm['precision']*100:.2f}%", help="55.22% of flagged alerts are true fraud")
        with t3:
            st.metric("Test Recall", f"{tm['recall']*100:.2f}%", help="74.47% of total fraud caught")
        with t4:
            st.metric("Test F1 Score", f"{tm['f1']:.4f}", help="Harmonic mean")

        st.markdown(f"### **Total Test Business Cost**: `${tm['total_cost']:,.2f}` (At Locked Threshold $t^* = {tm['locked_threshold']:.2f}$)")

        # Confusion Matrix Heatmap
        cm_data = [[tm['tn'], tm['fp']], [tm['fn'], tm['tp']]]
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
            x=['Legitimate (0)', 'Fraud (1)'],
            y=['Legitimate (0)', 'Fraud (1)'],
            text_auto=True,
            color_continuous_scale='Blues',
            title='Held-Out Test Set Confusion Matrix'
        )
        fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9')

        col_cm1, col_cm2 = st.columns([1, 1])
        with col_cm1:
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_cm2:
            st.markdown("#### Test Confusion Matrix Breakdown")
            st.markdown(f"• **True Positives (TP)**: `{tm['tp']}` frauds detected successfully.")
            st.markdown(f"• **False Positives (FP)**: `{tm['fp']}` false alarms (Cost: `{tm['fp']} x $1.0 = ${tm['fp']*1.0:,.2f}`).")
            st.markdown(f"• **False Negatives (FN)**: `{tm['fn']}` missed frauds (Cost: `{tm['fn']} x $10.0 = ${tm['fn']*10.0:,.2f}`).")
            st.markdown(f"• **True Negatives (TN)**: `{tm['tn']:,}` legitimate transactions correctly approved.")
            st.caption("Evaluation performed strictly ONCE on the untouched chronological test set.")

# ---------------------------------------------------------
# SECTION 3: DECISION ENGINE & THRESHOLD OPTIMIZATION
# ---------------------------------------------------------
elif nav_choice == "⚡ DECISION ENGINE":
    st.markdown("## ⚡ Decision Engine & Threshold Optimization")
    st.markdown("""
    In financial fraud detection, decision thresholds must balance false-alarm operational cost vs uncaptured fraud loss:
    
    `Total Cost = (FP × FP_COST) + (FN × FN_COST)  where FP_COST = $1.0, FN_COST = $10.0`
    """)

    curve, winning_model_name, locked_thresh = load_threshold_analysis()
    if curve and manifest:
        curve_df = pd.DataFrame(curve)

        c_curve, c_pr = st.columns(2)
        with c_curve:
            fig_cost = px.line(
                curve_df, x='threshold', y='cost',
                title=f"Validation Business Cost vs Decision Threshold ({winning_model_name})",
                labels={'threshold': 'Decision Threshold (t)', 'cost': 'Total Expected Business Cost ($)'}
            )
            fig_cost.add_vline(x=locked_thresh, line_dash="dash", line_color="#f85149", annotation_text=f"Locked t* = {locked_thresh:.2f}")
            fig_cost.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9')
            st.plotly_chart(fig_cost, use_container_width=True)

        with c_pr:
            if metrics and 'cost_curves' in metrics:
                fig_pr = go.Figure()
                for m_name, c_data in metrics['cost_curves'].items():
                    pr_val = metrics['validation_results'][m_name]['val_metrics']['pr_auc']
                    recalls = [item['recall'] for item in c_data]
                    precisions = [item['precision'] for item in c_data]
                    fig_pr.add_trace(go.Scatter(
                        x=recalls, y=precisions, mode='lines',
                        name=f"{m_name} (PR-AUC: {pr_val:.4f})"
                    ))
                fig_pr.update_layout(
                    title="Validation Precision-Recall (PR) Curves",
                    xaxis_title="Recall", yaxis_title="Precision",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9'
                )
                st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎛️ Interactive Threshold Decision Simulator")
    sim_prob = st.slider("Simulated Model Fraud Probability Output", 0.0, 1.0, 0.45, 0.01)
    locked_t = manifest['locked_threshold'] if manifest else 0.31

    d1, d2, d3 = st.columns(3)
    d1.metric("Predicted Fraud Probability", f"{sim_prob*100:.1f}%")
    d2.metric("Locked Operational Threshold (t*)", f"{locked_t*100:.1f}%")
    
    if sim_prob >= locked_t:
        d3.markdown("<div style='margin-top: 15px;'><span class='badge-danger'>HIGH RISK — BLOCK / REVIEW</span></div>", unsafe_allow_html=True)
    elif sim_prob >= (locked_t * 0.7):
        d3.markdown("<div style='margin-top: 15px;'><span class='badge-warning'>MEDIUM RISK — ELEVATE MONITORING</span></div>", unsafe_allow_html=True)
    else:
        d3.markdown("<div style='margin-top: 15px;'><span class='badge-success'>LOW RISK — AUTOMATIC APPROVE</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# SECTION 4: LIVE SCORING
# ---------------------------------------------------------
elif nav_choice == "🔍 LIVE SCORING":
    st.markdown("## 🔍 Real-Time Transaction Risk Analyst Tool")
    st.markdown("Input transaction parameters to compute live risk scores via the locked trained XGBoost classifier.")

    if winning_model and feature_cols and manifest:
        l1, l2, l3 = st.columns(3)
        tx_amt = l1.number_input("Transaction Amount ($)", min_value=1.0, max_value=10000.0, value=450.0)
        tx_hour = l2.slider("Hour of Transaction (0-23)", 0, 23, 3)
        cust_1d_tx = l3.number_input("Customer Past 1-Day Transaction Volume", min_value=0, max_value=50, value=2)

        l4, l5 = st.columns(2)
        term_risk = l4.slider("Delayed Terminal 7-Day Fraud Risk Rate", 0.0, 1.0, 0.12)
        night_flag = 1 if tx_hour in [0, 1, 2, 3, 4, 5] else 0

        input_data = pd.DataFrame([{
            'TX_AMOUNT': tx_amt,
            'log_tx_amount': np.log1p(tx_amt),
            'hour': tx_hour,
            'day_of_week': 2,
            'TX_DURING_WEEKEND': 0,
            'TX_DURING_NIGHT': night_flag,
            'CUSTOMER_ID_NB_TX_1DAY': cust_1d_tx,
            'CUSTOMER_ID_AVG_AMOUNT_1DAY': tx_amt,
            'CUSTOMER_ID_NB_TX_7DAY': cust_1d_tx,
            'CUSTOMER_ID_AVG_AMOUNT_7DAY': tx_amt,
            'CUSTOMER_ID_NB_TX_30DAY': cust_1d_tx,
            'CUSTOMER_ID_AVG_AMOUNT_30DAY': tx_amt,
            'TERMINAL_ID_NB_TX_1DAY': 6,
            'TERMINAL_ID_RISK_7DAY_DELAYED': term_risk,
            'TERMINAL_ID_RISK_30DAY_DELAYED': term_risk
        }])[feature_cols]

        if st.button("Score Transaction", type="primary"):
            prob = float(winning_model.predict_proba(input_data)[0, 1])
            locked_t = manifest['locked_threshold']
            is_fraud = prob >= locked_t

            st.markdown("---")
            res1, res2, res3 = st.columns(3)
            res1.metric("Predicted Fraud Risk Score", f"{prob*100:.2f}%")
            res2.metric("Decision Threshold", f"{locked_t*100:.2f}%")
            if is_fraud:
                res3.markdown("<div style='margin-top: 15px;'><span class='badge-danger'>ACTION: BLOCK TRANSACTION / FLAG FOR REVIEW</span></div>", unsafe_allow_html=True)
            else:
                res3.markdown("<div style='margin-top: 15px;'><span class='badge-success'>ACTION: APPROVE TRANSACTION</span></div>", unsafe_allow_html=True)

            st.markdown("#### Analyst Signal Interpretation")
            st.markdown(f"• **Night Time Transaction**: `{'Yes (Night Hour ' + str(tx_hour) + ')' if night_flag else 'No'}`\n• **Historical Terminal Fraud Rate**: `{term_risk*100:.1f}%` (Delayed 7-day reporting offset enforced)\n• **Customer 1-Day Activity**: `{cust_1d_tx}` transactions")
    else:
        st.error("Model artifacts not available. Train models first.")

# ---------------------------------------------------------
# SECTION 5: THREAT MONITORING
# ---------------------------------------------------------
elif nav_choice == "🚨 THREAT MONITORING":
    st.markdown("## 🚨 Merchant & Terminal Fraud-Spike Alert Center")
    st.markdown("Monitors terminal baseline fraud rates (30D) vs recent 7-day monitoring window rates to identify compromised merchant terminals.")

    spikes = load_spike_results()
    if spikes is not None:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("High Spike Alerts (≥3.0x)", len(spikes[spikes['alert_status'] == 'ALERT_HIGH']))
        s2.metric("Elevated Alerts (≥1.5x)", len(spikes[spikes['alert_status'] == 'ALERT_ELEVATED']))
        s3.metric("Normal Terminals", len(spikes[spikes['alert_status'] == 'NORMAL']))
        s4.metric("Filtered Low Volume", len(spikes[spikes['alert_status'] == 'INSUFFICIENT_VOLUME']))

        st.markdown("---")
        st.markdown("### ⚠ Top Active Terminal Fraud Spike Alerts")
        active_spikes = spikes[spikes['alert_status'].isin(['ALERT_HIGH', 'ALERT_ELEVATED'])].head(10)
        
        if len(active_spikes) > 0:
            for idx, row in active_spikes.iterrows():
                badge_class = "badge-danger" if row['alert_status'] == 'ALERT_HIGH' else "badge-warning"
                st.markdown(f"""
                <div style="background: rgba(22, 27, 34, 0.8); border: 1px solid rgba(48, 54, 61, 0.8); border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <b style="font-size: 16px; color: #f0f6fc;">Terminal ID: {row['TERMINAL_ID']}</b><br>
                            <span style="font-size: 13px; color: #8b949e;">Recent Fraud Rate: {row['monitoring_fraud_rate']*100:.2f}% | Baseline Fraud Rate: {row['baseline_fraud_rate']*100:.2f}%</span>
                        </div>
                        <div>
                            <span style="font-size: 18px; font-weight: 800; color: #58a6ff; margin-right: 15px;">Spike Ratio: {row['spike_ratio']:.2f}x</span>
                            <span class="{badge_class}">{row['alert_status']}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active high-risk fraud spikes detected.")
    else:
        st.error("Spike detection dataset or pipeline not available.")

# ---------------------------------------------------------
# SECTION 6: POLICY ASSISTANT
# ---------------------------------------------------------
elif nav_choice == "🤖 POLICY ASSISTANT":
    st.markdown("## 🤖 Grounded Risk Policy Verification Assistant")
    st.markdown("Grounds model outputs and analyst decisions against internal defensive risk documentation (`documents/risk_policies.json`). *RAG is strictly downstream and cannot alter ML predictions or thresholds.*")

    query_input = st.text_input("Enter policy query or operational incident question:", value="How to handle merchant terminal fraud spike?")

    if st.button("Query Policy Knowledge Base", type="primary"):
        res = rag_engine.retrieve(query_input, top_k=2)
        st.markdown("---")
        st.markdown("### 📚 Retrieved Grounded Policy Documents")
        for item in res:
            with st.expander(f"[{item['policy_id']}] {item['title']} ({item['category']})", expanded=True):
                st.markdown(f"**Policy Content**: {item['content']}")
                st.markdown(f"**Mandatory Analyst Action**: {item['action']}")
                if 'relevance_score' in item:
                    st.caption(f"Semantic Similarity Match Score: {item['relevance_score']:.4f}")
