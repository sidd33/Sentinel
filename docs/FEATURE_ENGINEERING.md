# Feature Engineering Documentation: Delayed Terminal Fraud-Risk Feature

## Overview & Definition

**Feature Name**: `TERMINAL_ID_RISK_7DAY_DELAYED`

This feature calculates the historical fraud rate of a merchant terminal (`TERMINAL_ID`) over a rolling historical window (default: 7 days) while strictly enforcing a **7-day fraud label reporting delay**.

---

## 1. Why a Fraud Label Reporting Delay is Necessary

In real-world credit card transaction processing:
- When a transaction occurs at time $T$, its ground-truth fraud label (`TX_FRAUD`) is **NOT** immediately known.
- Fraud is typically reported via cardholder chargebacks or bank fraud reports, which take **7 to 14 days** (or up to 60 days) to be processed and reported back to the merchant/acquiring bank.
- Therefore, assuming that past transactions from yesterday or earlier today already have confirmed fraud labels introduces **severe data leakage**.

SentinelPay enforces a modeling assumption of a **7-day reporting delay** to simulate real-world chargeback notice latency.

---

## 2. Implementation Methodology

For a target transaction occurring at timestamp $T$ on terminal $M$:

1. **Target Lookup Timestamp Calculation**:
   $$\text{LOOKUP\_DATETIME} = T - \text{delay\_days} \quad (\text{where } \text{delay\_days} = 7)$$

2. **Eligible Transactions**:
   Only historical transactions from terminal $M$ with timestamp $t_{\text{hist}}$ satisfying:
   $$t_{\text{hist}} \le T - \text{delay\_days}$$
   are eligible for feature computation.

3. **Prohibited Information**:
   - **Current-row label**: The target transaction's own `TX_FRAUD` label at time $T$ is strictly excluded.
   - **Future labels**: Any transaction occurring after time $T$ ($t > T$) is strictly excluded.
   - **In-delay labels**: Any transaction occurring within the 7-day delay period ($T - \text{delay\_days} < t \le T$) is strictly excluded.

4. **Lookup Engine (`pd.merge_asof`)**:
   Historical rolling fraud sum and total count over the 7-day window $[t_{\text{hist}} - 7\text{D}, t_{\text{hist}}]$ are pre-calculated for each terminal. Using pandas `merge_asof` with `direction='backward'`, each target transaction matches the latest available historical statistics snapshot at or before its `LOOKUP_DATETIME`.

---

## 3. Edge Cases & Boundary Handling

| Edge Case | Handled Behavior |
| :--- | :--- |
| **No Historical Transactions** | Returns `0.0` default risk rate |
| **No Eligible Transactions ($\le T - 7\text{D}$)** | Returns `0.0` default risk rate |
| **Terminal with Only Recent Transactions ($> T - 7\text{D}$)** | Returns `0.0` default risk rate |
| **Multiple Transactions at Exact Same Timestamp** | `merge_asof` joins tie-broken historical snapshot state |
| **Exact 7-Day Boundary ($t = T - 7\text{D}$)** | Transaction at $T - 7\text{D}$ is included (eligible) |

---

## 4. Verification & Unit Tests (`tests/test_feature_engineering.py`)

The behavior is verified via 5 deterministic unit tests:

1. `test_delayed_terminal_risk_semantics`: Confirms that for a transaction on Day 10 with a 7-day delay, only labels from Day 3 or earlier are eligible, while labels from Day 5 (inside delay window) and Day 10 (current row) are excluded.
2. `test_current_row_label_independence`: Proves that mutating the target transaction's own `TX_FRAUD` label from 0 to 1 produces **zero change** in its feature value.
3. `test_future_transaction_independence`: Proves that adding a future transaction on Day 12 with `TX_FRAUD = 1` produces **zero change** in an earlier transaction's feature value.
4. `test_delay_window_label_exclusion`: Proves that mutating a fraud label occurring on Day 5 (inside the 7-day delay window) produces **zero change** in the feature calculated on Day 10.
5. `test_edge_cases_no_eligible_data_and_exact_boundary`: Verifies default `0.0` behavior for new/recent terminals and exact 7-day boundary matching.

---

## ⚠️ Disclaimer

> This feature represents a modeling assumption designed for this research prototype based on synthetic benchmark data methodology. It does NOT claim to represent an internal Razorpay production system rule.
