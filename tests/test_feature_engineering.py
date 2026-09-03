import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add src to python path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from feature_engineering import add_delayed_terminal_features, build_feature_pipeline

def create_deterministic_sample():
    """
    Creates a deterministic test dataset:
    Terminal T1:
      Day 1 (2018-05-01 10:00): TX_A, Fraud = 1
      Day 3 (2018-05-03 10:00): TX_B, Fraud = 1
      Day 5 (2018-05-05 10:00): TX_C, Fraud = 0
      Day 10 (2018-05-10 10:00): TX_D, Fraud = 0
    """
    df = pd.DataFrame({
        'TRANSACTION_ID': [1, 2, 3, 4],
        'TX_DATETIME': [
            pd.Timestamp('2018-05-01 10:00:00'), # Day 1
            pd.Timestamp('2018-05-03 10:00:00'), # Day 3
            pd.Timestamp('2018-05-05 10:00:00'), # Day 5
            pd.Timestamp('2018-05-10 10:00:00'), # Day 10
        ],
        'CUSTOMER_ID': [100, 101, 102, 103],
        'TERMINAL_ID': ['T1', 'T1', 'T1', 'T1'],
        'TX_AMOUNT': [100.0, 150.0, 200.0, 250.0],
        'TX_FRAUD': [1, 1, 0, 0],
        'TX_FRAUD_SCENARIO': [1, 1, 0, 0]
    })
    return df

def test_delayed_terminal_risk_semantics():
    """
    Verifies that for TX_D on Day 10 with a 7-day reporting delay:
    - Target cutoff timestamp = Day 10 10:00 - 7 days = Day 3 10:00.
    - Eligible transactions: TX_A (Day 1) and TX_B (Day 3).
    - Excluded transactions: TX_C (Day 5, inside 7-day delay) and TX_D (Day 10, current row).
    - Calculated risk rate = (1 + 1) / 2 = 1.0.
    """
    df = create_deterministic_sample()
    df_processed = add_delayed_terminal_features(df, delay_days=7, risk_windows=[7])
    
    tx_d = df_processed[df_processed['TRANSACTION_ID'] == 4].iloc[0]
    
    # Expected: 2 fraud / 2 tx = 1.0
    assert tx_d['TERMINAL_ID_RISK_7DAY_DELAYED'] == pytest.approx(1.0), \
        f"Expected 1.0, got {tx_d['TERMINAL_ID_RISK_7DAY_DELAYED']}"

def test_current_row_label_independence():
    """
    Proves that changing current transaction's TX_FRAUD label from 0 to 1
    does NOT change its own terminal-risk feature.
    """
    df1 = create_deterministic_sample()
    res1 = add_delayed_terminal_features(df1, delay_days=7, risk_windows=[7])
    val1 = res1.loc[res1['TRANSACTION_ID'] == 4, 'TERMINAL_ID_RISK_7DAY_DELAYED'].values[0]

    # Mutate current transaction TX_D fraud label to 1
    df2 = create_deterministic_sample()
    df2.loc[df2['TRANSACTION_ID'] == 4, 'TX_FRAUD'] = 1
    res2 = add_delayed_terminal_features(df2, delay_days=7, risk_windows=[7])
    val2 = res2.loc[res2['TRANSACTION_ID'] == 4, 'TERMINAL_ID_RISK_7DAY_DELAYED'].values[0]

    assert val1 == val2, f"Feature changed when target mutated! {val1} != {val2}"

def test_future_transaction_independence():
    """
    Proves that adding a future transaction TX_E on Day 12 with TX_FRAUD = 1
    does NOT change an earlier transaction's (TX_D on Day 10) terminal-risk feature.
    """
    df = create_deterministic_sample()
    res1 = add_delayed_terminal_features(df, delay_days=7, risk_windows=[7])
    val_before = res1.loc[res1['TRANSACTION_ID'] == 4, 'TERMINAL_ID_RISK_7DAY_DELAYED'].values[0]

    # Append future transaction on Day 12
    df_future = pd.concat([df, pd.DataFrame([{
        'TRANSACTION_ID': 5,
        'TX_DATETIME': pd.Timestamp('2018-05-12 10:00:00'),
        'CUSTOMER_ID': 104,
        'TERMINAL_ID': 'T1',
        'TX_AMOUNT': 300.0,
        'TX_FRAUD': 1,
        'TX_FRAUD_SCENARIO': 1
    }])], ignore_index=True)

    res2 = add_delayed_terminal_features(df_future, delay_days=7, risk_windows=[7])
    val_after = res2.loc[res2['TRANSACTION_ID'] == 4, 'TERMINAL_ID_RISK_7DAY_DELAYED'].values[0]

    assert val_before == val_after, f"Future transaction leaked into past feature! {val_before} != {val_after}"

def test_delay_window_label_exclusion():
    """
    Proves that a fraud label occurring inside the 7-day reporting delay window
    (Day 5: 2018-05-05) CANNOT affect the feature calculated on Day 10 (2018-05-10).
    """
    df1 = create_deterministic_sample() # TX_C on Day 5 has TX_FRAUD = 0
    res1 = add_delayed_terminal_features(df1, delay_days=7, risk_windows=[7])
    val1 = res1.loc[res1['TRANSACTION_ID'] == 4, 'TERMINAL_ID_RISK_7DAY_DELAYED'].values[0]

    # Mutate Day 5 transaction (TX_C) fraud label to 1 (inside 7-day delay period)
    df2 = create_deterministic_sample()
    df2.loc[df2['TRANSACTION_ID'] == 3, 'TX_FRAUD'] = 1
    res2 = add_delayed_terminal_features(df2, delay_days=7, risk_windows=[7])
    val2 = res2.loc[res2['TRANSACTION_ID'] == 4, 'TERMINAL_ID_RISK_7DAY_DELAYED'].values[0]

    assert val1 == val2, f"Fraud label inside delay window leaked into feature! {val1} != {val2}"

def test_edge_cases_no_eligible_data_and_exact_boundary():
    """
    Edge Cases:
    1. Day 1 transaction: No historical transactions -> Risk = 0.0
    2. Day 5 transaction: Has TX_A (Day 1) and TX_B (Day 3), but lookup is Day 5 - 7D = Day -2 (No eligible tx) -> Risk = 0.0
    3. Exact boundary: Day 8 transaction (2018-05-08 10:00). Lookup = Day 1 10:00. Eligible: TX_A (Day 1 10:00). Risk = 1/1 = 1.0.
    """
    df = create_deterministic_sample()
    # Add exact boundary tx on Day 8
    df = pd.concat([df, pd.DataFrame([{
        'TRANSACTION_ID': 5,
        'TX_DATETIME': pd.Timestamp('2018-05-08 10:00:00'), # Day 8
        'CUSTOMER_ID': 105,
        'TERMINAL_ID': 'T1',
        'TX_AMOUNT': 120.0,
        'TX_FRAUD': 0,
        'TX_FRAUD_SCENARIO': 0
    }])], ignore_index=True)

    res = add_delayed_terminal_features(df, delay_days=7, risk_windows=[7])

    # Day 1 (TX_A): No eligible data -> 0.0
    val_day1 = res.loc[res['TRANSACTION_ID'] == 1, 'TERMINAL_ID_RISK_7DAY_DELAYED'].values[0]
    assert val_day1 == 0.0, f"Day 1 expected 0.0, got {val_day1}"

    # Day 5 (TX_C): Lookup Day 5 - 7D = April 28 (No eligible tx) -> 0.0
    val_day5 = res.loc[res['TRANSACTION_ID'] == 3, 'TERMINAL_ID_RISK_7DAY_DELAYED'].values[0]
    assert val_day5 == 0.0, f"Day 5 expected 0.0, got {val_day5}"

    # Day 8 (TX 5): Lookup Day 8 - 7D = Day 1 10:00. Matches TX_A (Day 1). Risk = 1.0
    val_day8 = res.loc[res['TRANSACTION_ID'] == 5, 'TERMINAL_ID_RISK_7DAY_DELAYED'].values[0]
    assert val_day8 == pytest.approx(1.0), f"Day 8 expected 1.0, got {val_day8}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
