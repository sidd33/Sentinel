# SentinelPay — AI Risk Manager

[![License](https://img.shields.io/badge/License-Defense--Only-green.svg)]()
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)]()

> **DEFENSE-ONLY STATEMENT**: SentinelPay is strictly a defensive fraud detection and merchant risk management research prototype built for Razorpay AI Risk Manager (Track 02). It provides transaction risk scoring, merchant spike alerts, and policy compliance verification. It contains zero offensive capabilities, evasion mechanisms, or bypass features.

---

## ⚠️ Important Dataset Disclaimer

> **SIMULATED DATASET NOTICE**: SentinelPay uses the official simulated credit card transaction dataset from the **Fraud Detection Handbook** (*Le Borgne et al.*).
> 
> **Experimental Subset Scope**: Final experiments use a **60-day chronological subset** of the official Fraud Detection Handbook simulated transaction dataset, covering the period from **2018-04-01 through 2018-05-30** (575,755 transactions). To keep the prototype computationally reproducible within the internship submission timeframe, experiments use a fixed 60-day subset of the official simulated dataset.
> 
> **This project DOES NOT use, access, or represent internal Razorpay merchant or payment data.** All reported metrics, transaction amounts, customer IDs, terminal IDs, and cost parameters are derived strictly from the public simulated benchmark dataset or explicitly declared hypothetical prototype assumptions.

---

## 🔬 Research Foundation

Primary academic baseline:

> **Baisholan et al. (2025)**  
> *"A Systematic Review of Machine Learning in Credit Card Fraud Detection Under Original Class Imbalance"*  
> **Computers**, 14(10), 437. [DOI: 10.3390/computers14100437]

Key methodological principles adopted from the paper:
1. **Original Class Imbalance**: Preserving natural fraud distribution (~0.74%) without applying artificial oversampling (e.g. SMOTE) to test evaluations.
2. **Imbalance-Robust Evaluation**: Utilizing **PR-AUC**, Precision, Recall, and F1-score as primary metrics rather than misleading overall Accuracy.
3. **Cost-Sensitive Optimization**: Selecting operating decision thresholds based on explicit business cost trade-offs ($\text{FP\_COST} = \$1.0$, $\text{FN\_COST} = \$10.0$).
4. **Data Leakage Prevention**: Enforcing strict temporal splitting and retrospective rolling feature windows with realistic label reporting delays.

---

## 🏗️ 3-Layer Defensive Architecture

```
                                  Incoming Transaction T
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │      Layer 1: Transaction ML Detector        │
                    │   (XGBoost / Random Forest / LogReg)        │
                    │   - Non-leaking retrospective features       │
                    │   - Delayed terminal risk (7-day offset)     │
                    │   - Locked threshold t* (Val Cost Minimized) │
                    └───────────────────────┬──────────────────────┘
                                            │
                                    Probability Score
                                            │
                    ┌───────────────────────┴──────────────────────┐
                    │                                              │
                    ▼                                              ▼
┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
│  Layer 2: Merchant Spike Detector    │        │ Layer 3: Grounded RAG Assistant      │
│  - Baseline vs Monitoring Rate       │        │  - TF-IDF / Semantic Policy Match    │
│  - Spike Ratio (High >= 3.0x)        │        │  - Grounded Analyst Explanations     │
│  - Configurable MIN_TX_COUNT Guard   │        │  - ZERO score modification           │
└──────────────────────────────────────┘        └──────────────────────────────────────┘
```

---

## 🔒 Data Leakage Prevention Methodology

To guarantee operational validity and zero data leakage:
- **Chronological Temporal Split**: 70% Train (earliest), 15% Validation (middle), 15% Final Test (latest). Future transactions NEVER influence past model fitting.
- **Retrospective Customer Features**: `CUSTOMER_ID_NB_TX_1DAY` and `CUSTOMER_ID_AVG_AMOUNT_1DAY` use `closed='left'` rolling windows strictly excluding the current row index $i$.
- **Delayed Terminal Fraud Labeling**: Terminal risk rate (`TERMINAL_ID_RISK_7DAY_DELAYED`) incorporates a mandatory 7-day delay offset to reflect real-world chargeback reporting latency.
- **Zero Synthetic Fallback**: If official pickle dataset download fails, the loader halts with explicit error messages rather than silently substituting generated synthetic data.

---

## 📊 Cost Model & Threshold Optimization

Because fraud loss heavily outweighs manual investigation effort, decision thresholds are optimized on the **Validation set** to minimize total business cost:

`Total Cost = (FP × FP_COST) + (FN × FN_COST)`

- **Hypothetical Cost Assumptions**:
  - `FP_COST` = `$1.0` (Manual review / analyst verification effort)
  - `FN_COST` = `$10.0` (Uncaptured fraud loss ratio)
- **Locked Threshold Protocol**: Once threshold $t^*$ is selected on Validation cost minimization, it is permanently locked in `artifacts/experiment_manifest.json` before performing a single evaluation on the untouched held-out Test set.

### 🏆 Official Final Experiment Results (Held-Out Test Set)

| Metric / Parameter | Value | Details |
| :--- | :---: | :--- |
| **Winning Model** | **XGBoost** | Selected via Validation set cost minimization ($2,510.00) |
| **Locked Threshold ($t^*$)** | **0.31** (0.306969...) | Locked on Validation cost curve minimization |
| **Test PR-AUC** | **0.7101** | Primary imbalance-robust evaluation metric |
| **Test Precision** | **0.5522** | 55.22% of flagged alerts are true fraud |
| **Test Recall** | **0.7447** | 74.47% of all fraud transactions caught |
| **Test F1 Score** | **0.6342** | Harmonic mean of precision and recall |
| **Test Total Cost** | **$2,399.00** | FP = 459 × $1.0 + FN = 194 × $10.0 |
| **Test Confusion Matrix** | TP = 566, FP = 459, FN = 194, TN = 85,145 | Evaluated on 86,364 held-out test transactions |

---

## ⚡ Merchant Fraud-Spike Detector

Monitors aggregate merchant terminal risk over time:
- **Baseline Window**: 30 days
- **Monitoring Window**: 7 days
- **Spike Ratio**: $\frac{\text{Monitoring Fraud Rate} + 0.001}{\text{Baseline Fraud Rate} + 0.001}$
- **Volume Filter**: `MIN_TX_COUNT = 10` (Configurable prototype volume guard to prevent small-sample false alarms).

---

## 🤖 Grounded RAG Risk Verification Assistant

- **Purpose**: Retrieve defensive risk policies (`documents/risk_policies.json`) and present grounded analyst verification steps.
- **Strict Downstream Rule**: Operating downstream of the ML pipeline; RAG CANNOT override ML scores, modify decision thresholds, or invent evidence.
- **Guardrails**: Rejects queries requesting evasion techniques or fraud generation.

---

## 🚀 Reproducibility & Execution Instructions

### 1. Environment Setup
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate

pip install pandas numpy scikit-learn xgboost matplotlib streamlit joblib pyyaml
```

### 2. Download Official Dataset & Train Models
```bash
# Step A: Download official Fraud Detection Handbook transformed dataset (60 days)
python src/dataset_loader.py

# Step B: Execute leakage audit, train models, optimize validation threshold, evaluate held-out test
python src/model_trainer.py
```

### 3. Launch Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
SentinelPay/
├── configs/
│   └── config.yaml                     # Central configuration (costs, splits, seeds, thresholds)
├── src/
│   ├── dataset_loader.py               # Official dataset acquisition & integrity validation
│   ├── feature_engineering.py          # Leakage-free temporal feature pipeline & audit
│   ├── model_trainer.py                # Chronological split, training, cost tuning & test eval
│   ├── spike_detector.py               # Merchant/terminal fraud-spike monitor
│   └── rag_assistant.py                # Grounded defensive RAG policy engine
├── documents/
│   └── risk_policies.json              # Defensive risk policy documentation base
├── models/
│   ├── winning_model.joblib            # Locked trained ML classifier
│   └── feature_cols.joblib             # Feature column names
├── artifacts/
│   ├── metrics.json                    # Output metrics & cost curves
│   └── experiment_manifest.json        # Reproducibility manifest
├── app.py                              # Streamlit 10-tab dashboard
└── README.md                           # Documentation & research foundation
```

---

## 📄 References

- **Baisholan, A., et al. (2025).** *A Systematic Review of Machine Learning in Credit Card Fraud Detection Under Original Class Imbalance*. Computers, 14(10), 437.
- **Le Borgne, Y. A., et al. (2022).** *Reproducible Machine Learning for Credit Card Fraud Detection*. Fraud Detection Handbook. GitHub repository: https://github.com/Fraud-Detection-Handbook/simulated-data-transformed
