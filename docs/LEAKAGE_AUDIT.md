# Data Leakage & Temporal Split Audit Report

## Executive Summary

SentinelPay implements a **zero-leakage, strictly chronological temporal pipeline** for defensive transaction fraud detection.

---

## 1. Chronological Split Boundaries (`artifacts/split_summary.json`)

The dataset is partitioned strictly by transaction timestamp `TX_DATETIME`:

$$\max(\text{train\_timestamp}) < \min(\text{validation\_timestamp}) < \max(\text{validation\_timestamp}) < \min(\text{test\_timestamp})$$

| Partition | Proportion | Transaction Count | Fraud Count | Fraud Rate | Start Timestamp | End Timestamp |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Train** | 70% | 403,028 | 2,748 | 0.682% | `2018-04-01 00:00:31` | `2018-05-12 22:25:55` |
| **Validation** | 15% | 86,363 | 755 | 0.874% | `2018-05-12 22:26:11` | `2018-05-21 20:13:53` |
| **Held-Out Test** | 15% | 86,364 | 760 | 0.880% | `2018-05-21 20:13:57` | `2018-05-30 23:59:45` |
| **Total** | 100% | 575,755 | 4,263 | 0.740% | `2018-04-01 00:00:31` | `2018-05-30 23:59:45` |

- **Overlap Verification**: Zero transaction overlap across partitions ($\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$).
- **No Random Sampling**: Split is 100% temporal. No random shuffle or stratified splitting.

---

## 2. Engineered Feature Audit Table

| Feature Name | Source Information | Historical Window | Can Use Future Rows? | Can Use Current Row? | Uses Target (`TX_FRAUD`)? | Label Delay | Operational Safety |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `TX_AMOUNT` | Transaction amount | Static $T$ | ❌ No | ✅ Row $T$ property | ❌ No | None | ✅ Safe |
| `log_tx_amount` | $\log(1 + \text{TX\_AMOUNT})$ | Static $T$ | ❌ No | ✅ Row $T$ property | ❌ No | None | ✅ Safe |
| `hour` | Hour of timestamp $T$ | Static $T$ | ❌ No | ✅ Row $T$ property | ❌ No | None | ✅ Safe |
| `day_of_week` | Day of week at $T$ | Static $T$ | ❌ No | ✅ Row $T$ property | ❌ No | None | ✅ Safe |
| `TX_DURING_WEEKEND` | Weekend indicator | Static $T$ | ❌ No | ✅ Row $T$ property | ❌ No | None | ✅ Safe |
| `TX_DURING_NIGHT` | Night hour indicator | Static $T$ | ❌ No | ✅ Row $T$ property | ❌ No | None | ✅ Safe |
| `CUSTOMER_ID_NB_TX_1DAY` | Customer past 1-day volume | $[T - 1\text{D}, T)$ | ❌ No | ❌ Excluded (`closed='left'`) | ❌ No | N/A | ✅ Safe |
| `CUSTOMER_ID_AVG_AMOUNT_1DAY` | Customer past 1-day avg amount | $[T - 1\text{D}, T)$ | ❌ No | ❌ Excluded (`closed='left'`) | ❌ No | N/A | ✅ Safe |
| `CUSTOMER_ID_NB_TX_7DAY` | Customer past 7-day volume | $[T - 7\text{D}, T)$ | ❌ No | ❌ Excluded (`closed='left'`) | ❌ No | N/A | ✅ Safe |
| `CUSTOMER_ID_AVG_AMOUNT_7DAY` | Customer past 7-day avg amount | $[T - 7\text{D}, T)$ | ❌ No | ❌ Excluded (`closed='left'`) | ❌ No | N/A | ✅ Safe |
| `CUSTOMER_ID_NB_TX_30DAY` | Customer past 30-day volume | $[T - 30\text{D}, T)$ | ❌ No | ❌ Excluded (`closed='left'`) | ❌ No | N/A | ✅ Safe |
| `CUSTOMER_ID_AVG_AMOUNT_30DAY` | Customer past 30-day avg amount | $[T - 30\text{D}, T)$ | ❌ No | ❌ Excluded (`closed='left'`) | ❌ No | N/A | ✅ Safe |
| `TERMINAL_ID_NB_TX_1DAY` | Terminal past 1-day volume | $[T - 1\text{D}, T)$ | ❌ No | ❌ Excluded (`closed='left'`) | ❌ No | N/A | ✅ Safe |
| `TERMINAL_ID_RISK_7DAY_DELAYED` | Terminal delayed 7-day fraud risk | $[T - 14\text{D}, T - 7\text{D}]$ | ❌ No | ❌ Excluded ($t \le T - 7\text{D}$) | ✅ Yes (Historical) | 7 Days | ✅ Safe |
| `TERMINAL_ID_RISK_30DAY_DELAYED` | Terminal delayed 30-day fraud risk | $[T - 37\text{D}, T - 7\text{D}]$ | ❌ No | ❌ Excluded ($t \le T - 7\text{D}$) | ✅ Yes (Historical) | 7 Days | ✅ Safe |

---

## 3. Target-Derived Feature Rules & Reporting Delay

The only features derived from `TX_FRAUD` are `TERMINAL_ID_RISK_7DAY_DELAYED` and `TERMINAL_ID_RISK_30DAY_DELAYED`.

- **Operational Justification**: Merchants monitor terminal-level historical chargeback rates to detect compromise patterns.
- **Reporting Delay Control**: Ground-truth fraud labels require time to be reported via chargebacks. SentinelPay enforces a mandatory **7-day delay offset** ($T - 7\text{D}$).
- **Lookup Engine**: Implemented via `pd.merge_asof(direction='backward')`, matching each transaction at $T$ against historical terminal statistics snapshot computed at $t \le T - 7\text{D}$.
- **Independence Guarantees**:
  - Current row target (`TX_FRAUD` at $T$) has **0 effect**.
  - Future targets ($t > T$) have **0 effect**.
  - Target labels inside the delay window ($T - 7\text{D} < t \le T$) have **0 effect**.

---

## 4. Train / Validation / Test Boundary Mechanics

Historical features are calculated over the continuous chronological transaction stream:

- A transaction in the Validation set (e.g., May 15) legitimately uses customer/terminal transaction history accumulated prior to May 15 (and terminal fraud labels prior to May 8).
- A transaction in the Held-Out Test set (e.g., May 25) legitimately uses history accumulated prior to May 25.
- **Strict Prohibition**: A transaction at time $T$ **NEVER** accesses information from $t > T$.

---

## 5. Preprocessing & Model Selection Isolation

- **Preprocessing**: Input features are standard numeric features filled with static default value `0.0`. No fitted scalers (`StandardScaler`, `MinMaxScaler`) or target encoders are applied.
- **Model Selection & Tuning**: Model training occurs strictly on `X_train, y_train`. Decision threshold selection ($t^* = 0.44$) and model candidate selection occur strictly on `X_val, y_val` via business cost curve minimization.
- **Held-Out Test Set Locking**: The test set (`X_test, y_test`) is evaluated **strictly ONCE** under locked model parameters and locked threshold $t^* = 0.44$.
