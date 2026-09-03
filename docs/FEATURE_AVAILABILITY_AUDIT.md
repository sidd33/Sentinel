# Independent Feature Availability Audit Report

## 1. Exact Final Model Feature List (15 Features)

1. `TX_AMOUNT`
2. `log_tx_amount`
3. `hour`
4. `day_of_week`
5. `TX_DURING_WEEKEND`
6. `TX_DURING_NIGHT`
7. `CUSTOMER_ID_NB_TX_1DAY`
8. `CUSTOMER_ID_AVG_AMOUNT_1DAY`
9. `CUSTOMER_ID_NB_TX_7DAY`
10. `CUSTOMER_ID_AVG_AMOUNT_7DAY`
11. `CUSTOMER_ID_NB_TX_30DAY`
12. `CUSTOMER_ID_AVG_AMOUNT_30DAY`
13. `TERMINAL_ID_NB_TX_1DAY`
14. `TERMINAL_ID_RISK_7DAY_DELAYED`
15. `TERMINAL_ID_RISK_30DAY_DELAYED`

---

## 2. Feature Availability Table

| Feature Name | Source | Historical Window | Current Row? | Future Rows? | Uses `TX_FRAUD`? | Label Delay | Decision-Time Available? | Final Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `TX_AMOUNT` | Transaction payload | Static $T$ | Included | Excluded | No | None | Yes | **SAFE** |
| `log_tx_amount` | $\log(1+\text{TX\_AMOUNT})$ | Static $T$ | Included | Excluded | No | None | Yes | **SAFE** |
| `hour` | `TX_DATETIME` | Static $T$ | Included | Excluded | No | None | Yes | **SAFE** |
| `day_of_week` | `TX_DATETIME` | Static $T$ | Included | Excluded | No | None | Yes | **SAFE** |
| `TX_DURING_WEEKEND` | `TX_DATETIME` | Static $T$ | Included | Excluded | No | None | Yes | **SAFE** |
| `TX_DURING_NIGHT` | `TX_DATETIME` | Static $T$ | Included | Excluded | No | None | Yes | **SAFE** |
| `CUSTOMER_ID_NB_TX_1DAY` | Customer History | $[T-1\text{D}, T)$ | Excluded | Excluded | No | N/A | Yes | **SAFE** |
| `CUSTOMER_ID_AVG_AMOUNT_1DAY` | Customer History | $[T-1\text{D}, T)$ | Excluded | Excluded | No | N/A | Yes | **SAFE** |
| `CUSTOMER_ID_NB_TX_7DAY` | Customer History | $[T-7\text{D}, T)$ | Excluded | Excluded | No | N/A | Yes | **SAFE** |
| `CUSTOMER_ID_AVG_AMOUNT_7DAY` | Customer History | $[T-7\text{D}, T)$ | Excluded | Excluded | No | N/A | Yes | **SAFE** |
| `CUSTOMER_ID_NB_TX_30DAY` | Customer History | $[T-30\text{D}, T)$ | Excluded | Excluded | No | N/A | Yes | **SAFE** |
| `CUSTOMER_ID_AVG_AMOUNT_30DAY` | Customer History | $[T-30\text{D}, T)$ | Excluded | Excluded | No | N/A | Yes | **SAFE** |
| `TERMINAL_ID_NB_TX_1DAY` | Terminal History | $[T-1\text{D}, T)$ | Excluded | Excluded | No | N/A | Yes | **SAFE** |
| `TERMINAL_ID_RISK_7DAY_DELAYED` | Delayed Fraud Labels | $[T-14\text{D}, T-7\text{D}]$ | Excluded | Excluded | Yes (Hist) | 7 Days | Yes | **SAFE WITH ASSUMPTION** |
| `TERMINAL_ID_RISK_30DAY_DELAYED` | Delayed Fraud Labels | $[T-37\text{D}, T-7\text{D}]$ | Excluded | Excluded | Yes (Hist) | 7 Days | Yes | **SAFE WITH ASSUMPTION** |

---

## 3. Exact Historical Window Semantics

- **Customer Features**: Calculated using `rolling(f"{w}D", closed='left')` on transactions sorted by `CUSTOMER_ID` and `TX_DATETIME`. For transaction at $T$, window is strictly $[T - W, T)$. Current row $T$ is **EXCLUDED**. Future rows ($> T$) are **EXCLUDED**.
- **Terminal Volume**: Calculated using `rolling('1D', closed='left')` on transactions sorted by `TERMINAL_ID` and `TX_DATETIME`. For transaction at $T$, window is strictly $[T - 1\text{D}, T)$. Current row $T$ is **EXCLUDED**.
- **Target-Derived Terminal Risk**: Calculated using `pd.merge_asof` with `LOOKUP_DATETIME = T - 7\text{ days}` and `direction='backward'`. Matches historical rolling fraud statistics computed at $t_{\text{hist}} \le T - 7\text{ days}$. Labels inside $[T - 7\text{D}, T]$ are **EXCLUDED**.

---

## 4. Operational & Test Verification Results

1. **Future-Information Independence**: Adding a future transaction at $T + 1\text{s}$ with arbitrary amount or fraud label produces **zero change** in transaction $T$'s feature values.
2. **Current-Target Independence**: Changing transaction $T$'s own `TX_FRAUD` label from 0 to 1 produces **zero change** in transaction $T$'s feature values.
3. **Post-Decision Label Delay**: For $T = \text{Day 10 10:00}$ ($\text{LOOKUP\_DATETIME} = \text{Day 3 10:00}$), Day 5 fraud label is **EXCLUDED** (inside 7-day delay), while Day 3 fraud label is **INCLUDED** (7+ days old).
4. **Missing Values**: Missing historical counts/rates default to `0.0` via `.fillna(0.0)`. No forward-fill or backward-fill imputation is used.
5. **Pipeline Order**: Raw data $\to$ Chronological sorting $\to$ Retrospective feature generation $\to$ Target separation $\to$ Chronological 70/15/15 split $\to$ Static fillna(0) $\to$ Model fitting.

---

## 5. Final Answer & Conclusion

**Can every feature used by the model be known at the exact transaction decision time without using future information?**

**YES WITH ASSUMPTIONS.**

*Reason*:
- All 6 raw payload/timestamp features are static properties of transaction $T$.
- All 7 customer/terminal historical volume features use strictly retrospective windows ($t < T$) via `closed='left'`.
- Both target-derived terminal risk features enforce a strict 7-day label reporting delay ($t \le T - 7\text{ days}$) via backward `merge_asof`, under the prototype modeling assumption that fraud chargebacks require 7 days to become available to the risk engine.
