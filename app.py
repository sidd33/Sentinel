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

# Page Configuration
st.set_page_config(
    page_title="SentinelPay | AI Risk Manager",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Palette)
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stMetric {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.8);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(22,27,34,0.9) 0%, rgba(13,17,23,0.9) 100%);
        border: 1px solid rgba(56, 139, 253, 0.3);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .badge-success {
        background-color: #238636;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-danger {
        background-color: #da3633;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #d29922;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .disclaimer-box {
        background-color: rgba(210, 153, 34, 0.15);
        border-left: 4px solid #d29922;
        padding: 12px 16px;
        margin-bottom: 20px;
        border-radius: 4px;
        color: #e3b341;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Load Artifacts via Dashboard Data Helpers
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

# Sidebar Header
st.sidebar.image("https://img.icons8.com/isometric-line/100/security-checked.png", width=70)
st.sidebar.title("SentinelPay Risk Engine")
st.sidebar.caption("Track 02 — Defensive AI Risk Manager")

st.sidebar.markdown("---")
st.sidebar.subheader("🔒 Experiment Settings")
if manifest:
    st.sidebar.markdown(f"**Dataset**: `{manifest['dataset_type'].upper()}`")
    st.sidebar.markdown(f"**Split Strategy**: `{manifest['split']}`")
    st.sidebar.markdown(f"**Winning Model**: `{manifest['winning_model']}`")
    st.sidebar.markdown(f"**Locked Threshold**: `{manifest['locked_threshold']:.2f}`")
    st.sidebar.caption(f"Prototype relative cost assumptions: FP = ${manifest['fp_cost']:.2f} | FN = ${manifest['fn_cost']:.2f}")
else:
    st.sidebar.error("Required evaluation artifact (experiment_manifest.json) not found. Run the evaluation pipeline before viewing this section.")

# Main Header
st.title("🛡️ SentinelPay — AI Risk Manager")
st.markdown("""
<div class="disclaimer-box">
    <b>SIMULATED DATASET DISCLAIMER:</b> SentinelPay uses the simulated transaction dataset from the Fraud Detection Handbook. 
    This system does NOT use or represent internal Razorpay data. Strictly defense-only.
</div>
""", unsafe_allow_html=True)

# Navigation Tabs (10 Required Sections)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "1. Dataset Overview",
    "2. Class Imbalance",
    "3. Model Comparison",
    "4. Held-Out Test",
    "5. Confusion Matrix",
    "6. PR Curves",
    "7. Threshold & Cost",
    "8. Interactive Scoring",
    "9. Merchant Spike Monitor",
    "10. RAG Risk Verification"
])

# ---------------------------------------------------------
# TAB 1: DATASET OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.header("1. Dataset Overview & Data Leakage Prevention")
    st.markdown("SentinelPay is built on the **Fraud Detection Handbook** benchmark dataset (*Baisholan et al., 2025*).")

    if dataset_summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{dataset_summary['total_transactions']:,}")
        col2.metric("Fraudulent Transactions", f"{dataset_summary['fraud_count']:,}", delta=f"{dataset_summary['fraud_percentage']:.3f}%")
        col3.metric("Unique Customers", f"{dataset_summary['unique_customers']:,}")
        col4.metric("Unique Terminals", f"{dataset_summary['unique_terminals']:,}")

        st.info(f"**Dataset Time Range**: `{dataset_summary['min_timestamp']}` to `{dataset_summary['max_timestamp']}`")
    else:
        st.error("Required dataset summary not found. Run the dataset pipeline before viewing this section.")

    st.subheader("Data Leakage Prevention Methodology")
    st.markdown("""
    - **Chronological Split Only**: Earliest 70% used for training, middle 15% for validation cost-tuning, latest 15% locked for final evaluation. No future transaction lookahead.
    - **Retrospective Customer Features**: `CUSTOMER_ID_NB_TX_1DAY` and `CUSTOMER_ID_AVG_AMOUNT_1DAY` use strictly `closed='left'` rolling windows ending *before* transaction time $T$.
    - **Delayed Terminal Fraud Risk**: Fraud label reporting incorporates a mandatory 7-day delay ($T - 7$ days offset) to account for chargeback notice latency.
    - **Zero Synthetic Fallback**: Final metrics are computed exclusively on official transformed dataset benchmark files.
    """)

# ---------------------------------------------------------
# TAB 2: CLASS IMBALANCE
# ---------------------------------------------------------
with tab2:
    st.header("2. Class Imbalance Analysis")
    st.markdown("Original class distribution preserved in validation and held-out test sets without synthetic oversampling on test data.")

    if dataset_summary:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### Transaction Class Distribution")
            labels = ['Legitimate (TX_FRAUD=0)', 'Fraudulent (TX_FRAUD=1)']
            values = [dataset_summary['legitimate_count'], dataset_summary['fraud_count']]
            fig_pie = px.pie(
                names=labels, values=values, 
                color_discrete_sequence=['#238636', '#da3633'],
                hole=0.4, title=f"Original Class Imbalance ({dataset_summary['fraud_percentage']:.3f}% Fraud Rate)"
            )
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9')
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("#### Imbalance Mitigation Strategy")
            st.markdown("""
            - **Class Weighting / scale_pos_weight**: Cost-aware loss weighting applied during Random Forest (`class_weight='balanced'`) and XGBoost (`scale_pos_weight=10.0`) training.
            - **Precision-Recall Evaluation**: Primary selection evaluated via **PR-AUC** rather than misleading overall Accuracy.
            - **Validation Threshold Optimization**: Decision threshold optimized specifically to minimize total business cost on Validation set.
            """)
    else:
        st.error("Required dataset summary not found. Run dataset pipeline first.")

# ---------------------------------------------------------
# TAB 3: MODEL COMPARISON (VALIDATION SET)
# ---------------------------------------------------------
with tab3:
    st.header("3. Validation Set Model Comparison")
    st.markdown("Models evaluated on the 15% Validation set prior to threshold locking.")

    if metrics and 'validation_results' in metrics:
        val_res = metrics['validation_results']
        records = []
        for m_name, res in val_res.items():
            m_data = res['val_metrics']
            records.append({
                'Model': m_name,
                'Optimal Threshold': res['optimal_threshold'],
                'Validation Cost ($)': f"${m_data['cost']:,.2f}",
                'PR-AUC': round(m_data['pr_auc'], 4),
                'Precision': round(m_data['precision'], 4),
                'Recall': round(m_data['recall'], 4),
                'F1 Score': round(m_data['f1'], 4),
                'False Positives': m_data['fp'],
                'False Negatives': m_data['fn']
            })
        st.dataframe(pd.DataFrame(records), use_container_width=True)
    else:
        st.error("Required evaluation artifact (metrics.json) not found. Run the evaluation pipeline before viewing this section.")

# ---------------------------------------------------------
# TAB 4: HELD-OUT TEST PERFORMANCE
# ---------------------------------------------------------
with tab4:
    st.header("4. Held-Out Test Set Final Performance (Locked Model)")
    st.markdown("Single evaluation on untouched 15% Test set using locked model & locked threshold.")

    if test_metrics:
        tm = test_metrics
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Winning Model", tm['model_name'])
        mc2.metric("Locked Threshold", f"{tm['locked_threshold']:.2f}")
        mc3.metric("Test PR-AUC", f"{tm['pr_auc']:.4f}")
        mc4.metric("Test F1 Score", f"{tm['f1']:.4f}")
        mc5.metric("Total Test Cost", f"${tm['total_cost']:,.2f}")

        st.markdown("---")
        c_p, c_r = st.columns(2)
        c_p.metric("Test Precision", f"{tm['precision']:.4f}", help="TP / (TP + FP)")
        c_r.metric("Test Recall", f"{tm['recall']:.4f}", help="TP / (TP + FN)")
    else:
        st.error("Required evaluation artifact (metrics.json -> test_metrics) not found. Run the evaluation pipeline before viewing this section.")

# ---------------------------------------------------------
# TAB 5: CONFUSION MATRIX
# ---------------------------------------------------------
with tab5:
    st.header("5. Test Set Confusion Matrix")
    st.markdown("Breakdown of predictions on held-out test data at locked threshold.")

    if test_metrics and manifest:
        tm = test_metrics
        cm_data = [[tm['tn'], tm['fp']], [tm['fn'], tm['tp']]]
        
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
            x=['Legitimate (0)', 'Fraud (1)'],
            y=['Legitimate (0)', 'Fraud (1)'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9')
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(fig_cm, use_container_width=True)
        with c2:
            st.markdown("#### Metric Interpretation")
            st.markdown(f"- **True Negatives (TN)**: `{tm['tn']:,}` legitimate transactions correctly approved.")
            st.markdown(f"- **False Positives (FP)**: `{tm['fp']:,}` false alerts (Cost: `{tm['fp']} x ${manifest['fp_cost']} = ${tm['fp']*manifest['fp_cost']:,.2f}`).")
            st.markdown(f"- **False Negatives (FN)**: `{tm['fn']:,}` missed frauds (Cost: `{tm['fn']} x ${manifest['fn_cost']} = ${tm['fn']*manifest['fn_cost']:,.2f}`).")
            st.markdown(f"- **True Positives (TP)**: `{tm['tp']:,}` frauds detected successfully.")
    else:
        st.error("Required evaluation artifact not found. Run the evaluation pipeline before viewing this section.")

# ---------------------------------------------------------
# TAB 6: PR CURVES
# ---------------------------------------------------------
with tab6:
    st.header("6. Precision-Recall Curves (PR-AUC)")
    st.markdown("PR curves generated directly from evaluated threshold points on Validation data.")
    
    if metrics and 'cost_curves' in metrics and 'validation_results' in metrics:
        fig_pr = go.Figure()
        for m_name, curve in metrics['cost_curves'].items():
            pr_val = metrics['validation_results'][m_name]['val_metrics']['pr_auc']
            recalls = [item['recall'] for item in curve]
            precisions = [item['precision'] for item in curve]
            fig_pr.add_trace(go.Scatter(
                x=recalls, y=precisions, mode='lines+markers',
                name=f"{m_name} (PR-AUC: {pr_val:.4f})"
            ))
        
        fig_pr.update_layout(
            title="Validation PR Curves",
            xaxis_title="Recall", yaxis_title="Precision",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9'
        )
        st.plotly_chart(fig_pr, use_container_width=True)
    else:
        st.error("Required evaluation artifact (metrics.json -> cost_curves) not found. Run the evaluation pipeline before viewing this section.")

# ---------------------------------------------------------
# TAB 7: THRESHOLD & COST TRADE-OFF
# ---------------------------------------------------------
with tab7:
    st.header("7. Threshold & Cost Minimization Trade-off Curve")
    
    curve, winning_model_name, locked_thresh = load_threshold_analysis()
    if curve and winning_model_name and manifest:
        st.markdown(f"Primary Objective: Minimize expected business cost on Validation set ($FP \\times \\${manifest['fp_cost']:.2f} + FN \\times \\${manifest['fn_cost']:.2f}$).")
        curve_df = pd.DataFrame(curve)
        
        fig_cost = px.line(
            curve_df, x='threshold', y='cost',
            title=f"Validation Cost Minimization Curve ({winning_model_name})",
            labels={'threshold': 'Decision Threshold (t)', 'cost': 'Total Business Cost ($)'}
        )
        
        fig_cost.add_vline(x=locked_thresh, line_dash="dash", line_color="#da3633", annotation_text=f"Locked t* = {locked_thresh:.2f}")
        fig_cost.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9')
        
        st.plotly_chart(fig_cost, use_container_width=True)
    else:
        st.error("Required evaluation artifact (metrics.json -> cost_curves) not found. Run the evaluation pipeline before viewing this section.")

# ---------------------------------------------------------
# TAB 8: INTERACTIVE TRANSACTION RISK SCORING
# ---------------------------------------------------------
with tab8:
    st.header("8. Interactive Transaction Risk Scoring")
    st.markdown("Score custom transaction parameters using the locked trained model.")

    if winning_model and feature_cols and manifest:
        ic1, ic2, ic3 = st.columns(3)
        tx_amt = ic1.number_input("Transaction Amount ($)", min_value=1.0, max_value=5000.0, value=350.0)
        tx_hour = ic2.slider("Hour of Day (0-23)", 0, 23, 2)
        cust_1d_tx = ic3.number_input("Customer 1-Day Tx Count", min_value=0, max_value=50, value=1)

        ic4, ic5 = st.columns(2)
        term_risk = ic4.slider("Delayed Terminal 7-Day Risk Rate", 0.0, 1.0, 0.05)
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
            'TERMINAL_ID_NB_TX_1DAY': 5,
            'TERMINAL_ID_RISK_7DAY_DELAYED': term_risk,
            'TERMINAL_ID_RISK_30DAY_DELAYED': term_risk
        }])[feature_cols]

        if st.button("Calculate Fraud Risk Score", type="primary"):
            prob = float(winning_model.predict_proba(input_data)[0, 1])
            locked_t = manifest['locked_threshold']
            is_fraud = prob >= locked_t

            st.markdown("---")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Predicted Fraud Probability", f"{prob*100:.2f}%")
            sc2.metric("Locked Threshold", f"{locked_t*100:.2f}%")
            sc3.markdown(f"**Decision**: {'<span class=\"badge-danger\">REJECT / FRAUD ALERT</span>' if is_fraud else '<span class=\"badge-success\">APPROVE / LOW RISK</span>'}", unsafe_allow_html=True)
    else:
        st.error("Required model artifacts (winning_model.joblib) not found. Run model training before using interactive scoring.")

# ---------------------------------------------------------
# TAB 9: MERCHANT SPIKE MONITOR
# ---------------------------------------------------------
with tab9:
    st.header("9. Merchant / Terminal Fraud-Spike Alert Center")

    min_vol = config.get('spike_detector', {}).get('min_transactions', 10) if config else 10
    st.markdown(f"Monitors terminal baseline fraud rate vs. recent 7-day monitoring window rate (`MIN_TX_COUNT` volume guard = `{min_vol}`).")

    spikes = load_spike_results()
    if spikes is not None:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("High Spike Terminals", len(spikes[spikes['alert_status'] == 'ALERT_HIGH']))
        sc2.metric("Elevated Terminals", len(spikes[spikes['alert_status'] == 'ALERT_ELEVATED']))
        sc3.metric("Normal Terminals", len(spikes[spikes['alert_status'] == 'NORMAL']))
        sc4.metric("Low Volume Filtered", len(spikes[spikes['alert_status'] == 'INSUFFICIENT_VOLUME']))

        st.subheader("Top Merchant Fraud Spike Alerts")
        st.dataframe(spikes[spikes['alert_status'].isin(['ALERT_HIGH', 'ALERT_ELEVATED'])], use_container_width=True)
    else:
        st.error("Required dataset or spike detection pipeline not available.")

# ---------------------------------------------------------
# TAB 10: RAG RISK VERIFICATION
# ---------------------------------------------------------
with tab10:
    st.header("10. RAG Risk Verification Assistant (Defense-Only)")
    st.markdown("Grounds model outputs against internal risk policies. *RAG is strictly downstream and does not alter ML predictions.*")

    query_input = st.text_input("Enter policy verification query or incident context:", value="How to handle merchant terminal fraud spike?")
    
    if st.button("Run Grounded RAG Query"):
        res = rag_engine.retrieve(query_input, top_k=2)
        st.subheader("Retrieved Policy Documents")
        for item in res:
            with st.expander(f"[{item['policy_id']}] {item['title']} ({item['category']})"):
                st.markdown(f"**Content**: {item['content']}")
                st.markdown(f"**Required Action**: {item['action']}")
                if 'relevance_score' in item:
                    st.caption(f"Vector Similarity Score: {item['relevance_score']:.4f}")
