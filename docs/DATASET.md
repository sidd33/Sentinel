# Dataset Provenance & Experimental Scope Documentation

## Overview

**Official Dataset Source**:  
[Fraud Detection Handbook Simulated Dataset](https://github.com/Fraud-Detection-Handbook/simulated-data-transformed)  
*Le Borgne et al. (2022)*

**Legal & Institutional Disclaimer**:  
> SentinelPay is built exclusively on the open benchmark simulated transaction dataset from the Fraud Detection Handbook.  
> **This project DOES NOT use, access, or represent internal Razorpay payment data.**

---

## Experimental Dataset Scope

Final experiments use a **60-day chronological subset** of the official Fraud Detection Handbook simulated transaction dataset, covering the period from **2018-04-01 through 2018-05-30**.

- **Rationale**: To keep the prototype computationally reproducible within the internship submission timeframe, experiments use a fixed 60-day subset of the official simulated dataset.
- **Dataset Properties**:
  - **Start Date**: `2018-04-01 00:00:31`
  - **End Date**: `2018-05-30 23:59:45`
  - **Total Transactions**: `575,755`
  - **Legitimate Transactions (TX_FRAUD=0)**: `571,492`
  - **Fraudulent Transactions (TX_FRAUD=1)**: `4,263`
  - **Overall Fraud Rate**: `0.740%`
  - **Unique Customers**: `4,979`
  - **Unique Terminals**: `10,000`

---

## Chronological Train / Validation / Test Split

The dataset is partitioned strictly in chronological order:

| Split Partition | Percentage | Rows Count | Date Range | Fraud Rate |
| :--- | :---: | :---: | :--- | :---: |
| **Training Set** | 70% | 403,028 | 2018-04-01 00:00:31 to 2018-05-12 22:25:55 | 0.682% |
| **Validation Set** | 15% | 86,363 | 2018-05-12 22:26:11 to 2018-05-21 20:13:53 | 0.874% |
| **Held-Out Test Set** | 15% | 86,364 | 2018-05-21 20:13:57 to 2018-05-30 23:59:45 | 0.880% |

---

## Reproducibility Protocol

To reproduce dataset acquisition and validation:

```bash
# Download exact 60-day daily pickle files (2018-04-01 to 2018-05-30)
python src/dataset_loader.py
```
