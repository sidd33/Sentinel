# Model Training & Evaluation Integrity Audit Report

## Executive Summary

This document performs an independent, read-only audit of SentinelPay's machine learning model training, hyperparameter configuration, threshold selection, cost minimization objective, PR-AUC calculation, and test set evaluation pipeline.

---

## 1. Feature Matrix & Target Verification

- **Exact Feature Columns (15 Features)**:
  `TX_AMOUNT`, `log_tx_amount`, `hour`, `day_of_week`, `TX_DURING_WEEKEND`, `TX_DURING_NIGHT`, `CUSTOMER_ID_NB_TX_1DAY`, `CUSTOMER_ID_AVG_AMOUNT_1DAY`, `CUSTOMER_ID_NB_TX_7DAY`, `CUSTOMER_ID_AVG_AMOUNT_7DAY`, `CUSTOMER_ID_NB_TX_30DAY`, `CUSTOMER_ID_AVG_AMOUNT_30DAY`, `TERMINAL_ID_NB_TX_1DAY`, `TERMINAL_ID_RISK_7DAY_DELAYED`, `TERMINAL_ID_RISK_30DAY_DELAYED`.
- **Target Vector**: Derived strictly from `TX_FRAUD` ($\{0, 1\}$ binary target).
- **Excluded Non-Feature Columns**: `TRANSACTION_ID`, `CUSTOMER_ID`, `TERMINAL_ID`, `TX_DATETIME`, `TX_FRAUD_SCENARIO` are strictly excluded from feature matrix $X$.
- **Partition Matrix Shapes**:
  - `X_train` shape: `(403028, 15)` | `y_train` fraud count: $2,748$
  - `X_val` shape: `(86363, 15)` | `y_val` fraud count: $755$
  - `X_test` shape: `(86364, 15)` | `y_test` fraud count: $760$

---

## 2. Feature Order & Preprocessing Consistency

- **Feature Order**: `X_train`, `X_val`, `X_test` use identical column order. Saved to `models/feature_cols.joblib`.
- **Inference Order**: `app.py` loads `models/feature_cols.joblib` and re-indexes incoming input DataFrames to match the exact training feature order.
- **Preprocessing Isolation**: Input features are filled with static `fillna(0.0)`. No fitted scalers (`StandardScaler`, `MinMaxScaler`) or target encoders are fit on validation or test sets.

---

## 3. Model Candidates & Class Imbalance Strategy

| Model Candidate | Class Imbalance Strategy | Hyperparameters | Training Partition |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | `class_weight='balanced'` | $C=1.0, \text{max\_iter}=1000, \text{seed}=42$ | `X_train, y_train` strictly |
| **Random Forest** | `class_weight='balanced'` | $n\_est=100, \text{max\_depth}=12, \text{seed}=42$ | `X_train, y_train` strictly |
| **XGBoost** | `scale_pos_weight=10.0` | $n\_est=100, \text{max\_depth}=6, \eta=0.1, \text{seed}=42$ | `X_train, y_train` strictly |

- **Validation Test Set Isolation**: Zero validation or test set data was supplied to model `.fit()`. No `eval_set` containing test data was passed to XGBoost.

---

## 4. Validation Model Selection & Cost Minimization Threshold

- **Selection Objective**: Minimize total business cost on the 15% Validation set:
  $$\text{Total Cost} = \text{FP} \times \text{FP\_COST} + \text{FN} \times \text{FN\_COST} \quad (\text{where } \text{FP\_COST}=\$1.0, \text{FN\_COST}=\$10.0)$$
- **Threshold Candidate Sweep**: 100 thresholds evaluated $t \in [0.01, 0.99]$.
- **Validation Evaluation Results**:
  - **Logistic Regression**: Val Cost = $\$1,765.00$ | PR-AUC = $0.7274$ | Opt Threshold = $0.89$
  - **Random Forest**: Val Cost = $\$1,362.00$ | PR-AUC = $0.8406$ | Opt Threshold = $0.41$
  - **XGBoost (WINNER)**: Val Cost = **$\$1,109.00$** | PR-AUC = **$0.8818$** | Locked Threshold $t^* = \mathbf{0.44}$

---

## 5. Metric Calculation & PR-AUC Integrity

- **PR-AUC Calculation**: Computed via `precision_recall_curve(y_true, y_probs)` and `auc(recall_vals, precision_vals)` using continuous predicted probabilities (NOT binary class predictions).
- **Confusion Matrix Alignment**: Formatted as `[[TN, FP], [FN, TP]]`.
- **Zero Denominator Handling**: Enforced using `zero_division=0` in scikit-learn metrics.

---

## 6. Held-Out Test Set Evaluation Results (`artifacts/metrics.json`)

Evaluated strictly **ONCE** on untouched held-out Test set (`X_test, y_test`):

- **Winning Model**: `XGBoost`
- **Locked Threshold**: `0.44` ($0.435656...$)
- **Test PR-AUC**: **0.8877**
- **Test Precision**: **0.7241**
- **Test Recall**: **0.8908**
- **Test F1 Score**: **0.7988**
- **Test Total Cost**: **$1,088.00**
- **Test Confusion Matrix**:
  - True Positives (TP): `677`
  - False Positives (FP): `258` (Cost: $\$258.00$)
  - False Negatives (FN): `83` (Cost: $\$830.00$)
  - True Negatives (TN): `85,346`

---

## 7. Model Integrity Verdict Table

| Audit Area | Verdict | Summary |
| :--- | :---: | :--- |
| **FEATURE MATRIX** | **PASS** | 15 features, zero target or ID leakage, correct shapes |
| **TARGET** | **PASS** | $\{0, 1\}$ binary target derived strictly from `TX_FRAUD` |
| **CLASS IMBALANCE** | **PASS** | Train-only `class_weight='balanced'` and `scale_pos_weight=10.0` |
| **LOGISTIC REGRESSION** | **PASS** | Trained strictly on `X_train`, evaluated passively |
| **RANDOM FOREST** | **PASS** | Trained strictly on `X_train`, evaluated passively |
| **XGBOOST** | **PASS** | Trained strictly on `X_train`, no test `eval_set` contamination |
| **RANDOMNESS** | **PASS** | `random_state=42` initialized across all models |
| **MODEL SELECTION** | **PASS** | Winner selected based on Validation set minimum cost |
| **THRESHOLD SELECTION** | **PASS** | Threshold locked on Validation cost curve minimization |
| **COST FUNCTION** | **PASS** | $\text{Cost} = \text{FP} \times 1.0 + \text{FN} \times 10.0$ prototype assumption |
| **PR-AUC** | **PASS** | Calculated on continuous probabilities via `precision_recall_curve` |
| **CONFUSION MATRIX** | **PASS** | Correctly ordered `[[TN, FP], [FN, TP]]` |
| **FINAL TEST EVALUATION** | **PASS** | Evaluated strictly ONCE under locked model & threshold |
| **MODEL ARTIFACTS** | **PASS** | Saved `winning_model.joblib` matches manifest & evaluation |
| **STORED METRICS** | **PASS** | 100% consistent with independent metric recomputation |

---

## 8. Summary Answers

1. **Exact Final Feature Count**: 15 features.
2. **Exact Model Candidates**: Logistic Regression, Random Forest, XGBoost.
3. **Class Imbalance Strategy**: Train-only `class_weight='balanced'` (LR, RF) and `scale_pos_weight=10.0` (XGBoost).
4. **Validation Selection Criterion**: Minimum total expected business cost on Validation set.
5. **Locked Threshold**: $t^* = 0.44$ ($0.435656...$).
6. **Cost Equation**: $\text{Total Cost} = \text{FP} \times \$1.0 + \text{FN} \times \$10.0$.
7. **Metric Implementation**: Standard scikit-learn metrics (`precision_recall_curve`, `auc`, `confusion_matrix`, `zero_division=0`).
8. **Independent Metric Comparison**: 100% match between stored artifact metrics and independent recomputation.
9. **Test-Set Usage Audit**: Test set accessed strictly ONCE after model + threshold locking. Zero contamination.
10. **Artifact Consistency Audit**: `winning_model.joblib`, `feature_cols.joblib`, `metrics.json`, and `experiment_manifest.json` are 100% consistent.
11. **Warnings**: 0 warnings.
12. **Bugs**: 0 bugs.
13. **Can existing metrics still be trusted?**: **YES. 100% trustworthy and procedurally sound.**
