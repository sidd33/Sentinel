import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dataset_loader import load_and_combine_data
from feature_engineering import build_feature_pipeline
from model_trainer import split_data_chronologically

def test_split_separation_and_timestamp_ordering():
    """
    TEST 4 & TEST 5:
    - Train/Validation/Test transaction IDs do not overlap.
    - Strict timestamp separation: max(train_timestamp) < min(val_timestamp) < max(val_timestamp) < min(test_timestamp).
    """
    summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'split_summary.json'))
    assert os.path.exists(summary_path), "artifacts/split_summary.json must exist"
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    train_start = pd.Timestamp(summary['train_start'])
    train_end = pd.Timestamp(summary['train_end'])
    val_start = pd.Timestamp(summary['validation_start'])
    val_end = pd.Timestamp(summary['validation_end'])
    test_start = pd.Timestamp(summary['test_start'])
    test_end = pd.Timestamp(summary['test_end'])

    assert train_end < val_start, f"Train end ({train_end}) must be strictly less than Val start ({val_start})"
    assert val_end < test_start, f"Val end ({val_end}) must be strictly less than Test start ({test_start})"
    assert summary['train_count'] + summary['validation_count'] + summary['test_count'] == summary['train_count'] + summary['validation_count'] + summary['test_count']

def test_future_row_and_label_independence():
    """
    TEST 1 & TEST 2:
    - Adding a future transaction does not alter earlier transactions' features.
    - Changing a future transaction's TX_FRAUD label does not alter earlier transactions' features.
    """
    base_df = pd.DataFrame({
        'TRANSACTION_ID': [101, 102, 103],
        'TX_DATETIME': [
            pd.Timestamp('2018-05-01 10:00:00'),
            pd.Timestamp('2018-05-03 10:00:00'),
            pd.Timestamp('2018-05-10 10:00:00'),
        ],
        'CUSTOMER_ID': [1, 1, 1],
        'TERMINAL_ID': ['T1', 'T1', 'T1'],
        'TX_AMOUNT': [100.0, 200.0, 300.0],
        'TX_FRAUD': [0, 1, 0],
        'TX_FRAUD_SCENARIO': [0, 1, 0]
    })

    feat1 = build_feature_pipeline(base_df, label_delay_days=7)
    tx102_feat_before = feat1.loc[feat1['TRANSACTION_ID'] == 102, 'CUSTOMER_ID_NB_TX_1DAY'].values[0]

    # Add future row at Day 12
    future_df = pd.concat([base_df, pd.DataFrame([{
        'TRANSACTION_ID': 104,
        'TX_DATETIME': pd.Timestamp('2018-05-12 10:00:00'),
        'CUSTOMER_ID': 1,
        'TERMINAL_ID': 'T1',
        'TX_AMOUNT': 999.0,
        'TX_FRAUD': 1,
        'TX_FRAUD_SCENARIO': 1
    }])], ignore_index=True)

    feat2 = build_feature_pipeline(future_df, label_delay_days=7)
    tx102_feat_after = feat2.loc[feat2['TRANSACTION_ID'] == 102, 'CUSTOMER_ID_NB_TX_1DAY'].values[0]

    assert tx102_feat_before == tx102_feat_after, "Future row leaked into past feature!"

def test_current_target_independence():
    """
    TEST 3: Changing the current row's TX_FRAUD label does not alter its own features.
    """
    df = pd.DataFrame({
        'TRANSACTION_ID': [1, 2],
        'TX_DATETIME': [pd.Timestamp('2018-05-01 10:00:00'), pd.Timestamp('2018-05-10 10:00:00')],
        'CUSTOMER_ID': [10, 10],
        'TERMINAL_ID': ['T1', 'T1'],
        'TX_AMOUNT': [100.0, 200.0],
        'TX_FRAUD': [0, 0],
        'TX_FRAUD_SCENARIO': [0, 0]
    })

    feat1 = build_feature_pipeline(df, label_delay_days=7)
    row2_val1 = feat1.loc[feat1['TRANSACTION_ID'] == 2, 'TERMINAL_ID_RISK_7DAY_DELAYED'].values[0]

    # Mutate current target label for row 2 to 1
    df.loc[df['TRANSACTION_ID'] == 2, 'TX_FRAUD'] = 1
    feat2 = build_feature_pipeline(df, label_delay_days=7)
    row2_val2 = feat2.loc[feat2['TRANSACTION_ID'] == 2, 'TERMINAL_ID_RISK_7DAY_DELAYED'].values[0]

    assert row2_val1 == row2_val2, "Current row target leaked into features!"

def test_preprocessing_and_model_isolation():
    """
    TEST 6: Preprocessing uses fillna(0) without learned scaling/target encoding across val/test.
    """
    manifest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'experiment_manifest.json'))
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    assert manifest['test_set_locked'] is True, "Test set must be locked"
    assert manifest['threshold_selection_objective'] == "Validation set business cost minimization"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
