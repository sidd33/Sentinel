# Final Clean Experiment Report — SentinelPay

## Executive Summary

This document presents the official, fully reproducible final experiment report for **SentinelPay — AI Risk Manager (Razorpay Track 02)**. All methodological audits (dataset scope alignment, temporal split verification, feature availability audit, research claim audit, and training pipeline integrity audit) have passed with zero leakage and 100% deterministic reproducibility.

---

## 1. Dataset Provenance & Scope

- **Dataset**: Official transformed simulated credit card transaction dataset from the **Fraud Detection Handbook** (*Le Borgne et al., 2022*).
- **Dataset Scope**: 60 chronological benchmark days (`2018-04-01 00:00:31` to `2018-05-30 23:59:45`).
- **Total Transactions**: **575,755**
- **Fraud Transactions**: **4,263** (Fraud Rate: **0.740%**)
- **Legitimate Transactions**: **571,492** (Legitimate Rate: **99.260%**)

---

## 2. Feature Matrix (15 Features)

1. `TX_AMOUNT`: Raw transaction amount
2. `log_tx_amount`: Log-transformed amount $\log(1 + \text{TX\_AMOUNT})$
3. `hour`: Hour of transaction (0–23)
4. `day_of_week`: Day of week (0–6)
5. `TX_DURING_WEEKEND`: Weekend binary indicator
6. `TX_DURING_NIGHT`: Night hour (0–5) binary indicator
7. `CUSTOMER_ID_NB_TX_1DAY`: Customer transaction count in past 1 day (`closed='left'`)
8. `CUSTOMER_ID_AVG_AMOUNT_1DAY`: Customer average amount in past 1 day (`closed='left'`)
9. `CUSTOMER_ID_NB_TX_7DAY`: Customer transaction count in past 7 days (`closed='left'`)
10. `CUSTOMER_ID_AVG_AMOUNT_7DAY`: Customer average amount in past 7 days (`closed='left'`)
11. `CUSTOMER_ID_NB_TX_30DAY`: Customer transaction count in past 30 days (`closed='left'`)
12. `CUSTOMER_ID_AVG_AMOUNT_30DAY`: Customer average amount in past 30 days (`closed='left'`)
13. `TERMINAL_ID_NB_TX_1DAY`: Terminal transaction count in past 1 day (`closed='left'`)
14. `TERMINAL_ID_RISK_7DAY_DELAYED`: Delayed 7-day terminal fraud risk rate ($t \le T - 7\text{D}$)
15. `TERMINAL_ID_RISK_30DAY_DELAYED`: Delayed 30-day terminal fraud risk rate ($t \le T - 7\text{D}$)

---

## 3. Chronological Temporal Split Breakdown

Strict chronological split without random sampling or label rebalancing:

| Partition | Proportion | Transaction Count | Fraud Count | Fraud Rate | Start Timestamp | End Timestamp |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Train** | 70% | 403,028 | 2,748 | 0.682% | `2018-04-01 00:00:31` | `2018-05-12 22:25:55` |
| **Validation** | 15% | 86,363 | 755 | 0.874% | `2018-05-12 22:26:11` | `2018-05-21 20:13:53` |
| **Held-Out Test** | 15% | 86,364 | 760 | 0.880% | `2018-05-21 20:13:57` | `2018-05-30 23:59:45` |

---

## 4. Validation Results & Threshold Selection

Model candidates were trained strictly on `X_train, y_train` (Random Seed = 42). Operating decision thresholds were optimized strictly on the **Validation set** by minimizing expected business cost:

$$\text{Total Cost} = \text{FP} \times \$1.0 + \text{FN} \times \$10.0$$

| Model Candidate | Class Imbalance Strategy | Optimal Threshold | Validation PR-AUC | Validation Precision | Validation Recall | Validation F1 | Validation Cost |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `class_weight='balanced'` | $0.84$ | $0.3966$ | $0.4161$ | $0.6861$ | $0.5180$ | $\$3,097.00$ |
| **Random Forest** | `class_weight='balanced'` | $0.42$ | $0.5836$ | $0.5558$ | $0.7060$ | $0.6219$ | $\$2,646.00$ |
| **XGBoost (WINNER)** | `scale_pos_weight=10.0` | **$0.31$** | **$0.6885$** | **$0.5652$** | **$0.7232$** | **$0.6345$** | **$\$2,510.00$** |

- **Winning Model Selected**: **XGBoost**
- **Locked Decision Threshold ($t^*$)**: **$0.31$**
- **Validation Minimum Cost**: **$\$2,510.00$**

---

## 5. Final Held-Out Test Set Evaluation

The held-out test set (`X_test, y_test`) was evaluated **strictly ONCE** under the locked XGBoost model and locked threshold $t^* = 0.31$:

- **Test PR-AUC**: **0.7101**
- **Test Precision**: **0.5522**
- **Test Recall**: **0.7447**
- **Test F1 Score**: **0.6342**
- **Test Total Cost**: **$2,399.00**
- **Test Confusion Matrix**:
  - True Positives (TP): `566`
  - False Positives (FP): `459` (Cost: $\$459.00$)
  - False Negatives (FN): `194` (Cost: $\$1,940.00$)
  - True Negatives (TN): `85,145`

---

## 6. Verification & Reproducibility Audits

- **Independent Metric Verification**: **PASS** (100% exact match between `artifacts/metrics.json` and independent recomputation on test partition).
- **Deterministic Reproducibility**: **PASS** (Repeated executions produce identical model parameters, predictions, threshold $t^* = 0.31$, and test cost $\$2,399.00$ under seed 42).
- **Unit Test Suite**: **PASS** (16/16 unit tests passed in 4.79s).

---

## 7. Software Environment (`artifacts/run_manifest.json`)

- **Python**: `3.13.7`
- **pandas**: `2.2.3`
- **numpy**: `2.1.1`
- **scikit-learn**: `1.5.2`
- **xgboost**: `2.1.1`
- **joblib**: `1.4.2`
- **Random Seed**: `42`

---

## 8. Disclaimer & Scope

> **SIMULATED DATASET & COMPETITION NOTICE**: SentinelPay is built for **Razorpay AI Risk Manager (Track 02)**. It uses the public Fraud Detection Handbook simulated dataset ($575,755$ transactions, $60$ days). It contains **zero** internal Razorpay payment data, merchant records, or proprietary infrastructure code. Cost parameters ($\text{FP\_COST}=\$1.0, \text{FN\_COST}=\$10.0$) and threshold objectives are defensive prototype modeling assumptions.
