# Research Methodology Claim Audit Report

## Executive Summary

This document performs a read-only audit of SentinelPay's research claims, paper citations, dataset provenance, feature engineering origins, and Razorpay competition disclaimers.

---

## 1. Primary Research References

1. **Academic Systematic Review Baseline**:
   - **Title**: *"A Systematic Review of Machine Learning in Credit Card Fraud Detection Under Original Class Imbalance"*
   - **Authors**: Baisholan, A., et al. (2025)
   - **Journal**: *Computers*, 14(10), 437. DOI: `10.3390/computers14100437`
   - **Adoption**: Methodological guidelines (preserving natural imbalance ~0.74%, chronological temporal split, PR-AUC evaluation, validation cost curve threshold optimization).

2. **Benchmark Dataset & Educational Handbook**:
   - **Title**: *"Reproducible Machine Learning for Credit Card Fraud Detection"* (Fraud Detection Handbook)
   - **Authors**: Yann-Aël Le Borgne, Wissam Siblini, Lasserre, & Gianluca Bontempi (2022)
   - **Repository**: [Fraud-Detection-Handbook/simulated-data-transformed](https://github.com/Fraud-Detection-Handbook/simulated-data-transformed)
   - **Adoption**: Source of the 60-day transformed simulated benchmark transaction dataset (`2018-04-01` to `2018-05-30`).

---

## 2. Feature Provenance Comparison Matrix (15 Features)

| Our Feature | Exists in Paper/Handbook? | Same Definition? | SentinelPay Definition | Citation / Source Evidence | Classification |
| :--- | :---: | :---: | :--- | :--- | :--- |
| `TX_AMOUNT` | Yes | Yes | Raw transaction amount in dollars | Standard Handbook column | **DIRECTLY FROM RESEARCH** |
| `log_tx_amount` | Yes | Inspired | $\log(1 + \text{TX\_AMOUNT})$ | Handbook Chapter 3 transformation | **INSPIRED BY RESEARCH** |
| `hour` | Yes | Yes | `TX_DATETIME.dt.hour` (0-23) | Handbook derived time feature | **DIRECTLY FROM RESEARCH** |
| `day_of_week` | Yes | Yes | `TX_DATETIME.dt.dayofweek` (0-6) | Handbook derived time feature | **DIRECTLY FROM RESEARCH** |
| `TX_DURING_WEEKEND` | Yes | Yes | Binary indicator if day in $[5, 6]$ | Handbook time flag | **DIRECTLY FROM RESEARCH** |
| `TX_DURING_NIGHT` | Yes | Inspired | Binary indicator if hour in $[0, 5]$ | Handbook Chapter 3 night indicator | **INSPIRED BY RESEARCH** |
| `CUSTOMER_ID_NB_TX_1DAY` | Yes | Yes | Customer tx count in past 1D (`closed='left'`) | Handbook RF/XGBoost customer feature | **DIRECTLY FROM RESEARCH** |
| `CUSTOMER_ID_AVG_AMOUNT_1DAY` | Yes | Yes | Customer mean amount in past 1D (`closed='left'`) | Handbook RF/XGBoost customer feature | **DIRECTLY FROM RESEARCH** |
| `CUSTOMER_ID_NB_TX_7DAY` | Yes | Yes | Customer tx count in past 7D (`closed='left'`) | Handbook RF/XGBoost customer feature | **DIRECTLY FROM RESEARCH** |
| `CUSTOMER_ID_AVG_AMOUNT_7DAY` | Yes | Yes | Customer mean amount in past 7D (`closed='left'`) | Handbook RF/XGBoost customer feature | **DIRECTLY FROM RESEARCH** |
| `CUSTOMER_ID_NB_TX_30DAY` | Yes | Yes | Customer tx count in past 30D (`closed='left'`) | Handbook RF/XGBoost customer feature | **DIRECTLY FROM RESEARCH** |
| `CUSTOMER_ID_AVG_AMOUNT_30DAY` | Yes | Yes | Customer mean amount in past 30D (`closed='left'`) | Handbook RF/XGBoost customer feature | **DIRECTLY FROM RESEARCH** |
| `TERMINAL_ID_NB_TX_1DAY` | Yes | Yes | Terminal tx count in past 1D (`closed='left'`) | Handbook terminal volume feature | **DIRECTLY FROM RESEARCH** |
| `TERMINAL_ID_RISK_7DAY_DELAYED` | No | No | Terminal fraud rate in $[T-14\text{D}, T-7\text{D}]$ via backward `merge_asof` | SentinelPay implementation of 7-day chargeback label delay | **SENTINELPAY-DESIGNED** |
| `TERMINAL_ID_RISK_30DAY_DELAYED` | No | No | Terminal fraud rate in $[T-37\text{D}, T-7\text{D}]$ via backward `merge_asof` | SentinelPay implementation of 7-day chargeback label delay | **SENTINELPAY-DESIGNED** |

---

## 3. Provenance of Delayed Terminal Features

`TERMINAL_ID_RISK_7DAY_DELAYED` and `TERMINAL_ID_RISK_30DAY_DELAYED` are **SentinelPay-designed features**. While the Fraud Detection Handbook discusses delayed feedback conceptually in credit card systems, the explicit implementation using `pd.merge_asof(direction='backward')` to match historical snapshots at $t \le T - 7\text{ days}$ is an original SentinelPay design component.

---

## 4. Dataset Provenance & Razorpay Disclaimers

- **Dataset Provenance**: Official transformed daily pickle files from `Fraud-Detection-Handbook/simulated-data-transformed` (60-day chronological subset: `2018-04-01` to `2018-05-30`, $575,755$ transactions).
- **Razorpay Competition Disclaimer**: SentinelPay is built for **Razorpay AI Risk Manager (Track 02)** research competition. It contains **ZERO** internal Razorpay payment data, merchant profiles, or proprietary risk rules. Cost parameters ($\text{FP\_COST}=\$1.0, \text{FN\_COST}=\$10.0$) and merchant spike thresholds are prototype modeling assumptions.

---

## 5. Final Project Component Classification (A/B/C/D/E)

| Component | Classification Category | Description / Provenance |
| :--- | :--- | :--- |
| **Simulated Dataset** | **A. DIRECTLY FROM RESEARCH** | Official Fraud Detection Handbook benchmark dataset |
| **Chronological Temporal Split** | **A. DIRECTLY FROM RESEARCH** | 70/15/15 chronological split methodology (Baisholan et al., 2025) |
| **60-Day Dataset Subset** | **E. PRESENTATION/ENGINEERING CHOICE**| Chosen for computational reproducibility within submission timeframe |
| **Payload & Time Features** | **A. DIRECTLY FROM RESEARCH** | Standard benchmark features (`TX_AMOUNT`, `hour`, `day_of_week`) |
| **Customer Rolling Features** | **A. DIRECTLY FROM RESEARCH** | Retrospective customer 1D/7D/30D rolling count and mean amount |
| **Delayed Terminal Fraud Risk**| **C. SENTINELPAY-DESIGNED** | Enforces 7-day label reporting delay via `pd.merge_asof` backward lookup |
| **ML Model Candidates** | **B. INSPIRED BY RESEARCH** | LR, RF, XGBoost classifiers recommended in fraud detection reviews |
| **Validation Cost Minimization** | **B. INSPIRED BY RESEARCH** | Business cost-sensitive decision threshold optimization |
| **Hypothetical Cost Values** | **D. ASSUMPTION** | $\text{FP\_COST}=\$1.0, \text{FN\_COST}=\$10.0$ prototype modeling assumptions |
| **Merchant Fraud-Spike Detector**| **C. SENTINELPAY-DESIGNED** | Baseline vs monitoring window fraud rate ratio with `MIN_TX_COUNT=10` guard |
| **Grounded RAG Policy Assistant**| **C. SENTINELPAY-DESIGNED** | Downstream TF-IDF policy retrieval over `risk_policies.json` |
| **Streamlit Interactive UI** | **E. PRESENTATION/ENGINEERING CHOICE**| 10-tab glassmorphism research prototype dashboard |

---

## 6. Final Question & Verdict

**Can we honestly say that SentinelPay reproduces the cited research?**

### **PARTIALLY**

**Precise Explanation**:
1. **What is reproduced**: SentinelPay faithfully reproduces the **methodological principles** of Baisholan et al. (2025) and the Fraud Detection Handbook by preserving natural class imbalance (~0.74%), using chronological temporal splitting, avoiding synthetic test oversampling, evaluating on PR-AUC, and selecting thresholds via validation cost curves.
2. **What is SentinelPay-designed**: The explicit 7-day delayed terminal fraud risk implementation via `pd.merge_asof`, the Merchant Fraud-Spike Detector, the Grounded RAG Risk Policy Assistant, and the Streamlit 10-tab dashboard are original SentinelPay defensive engineering designs, not direct reproductions of paper algorithms.
