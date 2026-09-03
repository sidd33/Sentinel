import os
import yaml
import numpy as np
import pandas as pd

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def detect_merchant_spikes(df, config=None):
    """
    Computes historical baseline fraud rate vs. current monitoring window fraud rate per TERMINAL_ID.
    Calculates spike ratio and assigns alert status subject to configurable min_tx_count guard.
    """
    if config is None:
        config = load_config()

    min_tx = config['spike_detector']['min_transactions']
    high_thresh = config['spike_detector']['high_spike_threshold']
    elevated_thresh = config['spike_detector']['elevated_spike_threshold']
    baseline_days = config['spike_detector']['baseline_window_days']
    monitoring_days = config['spike_detector']['monitoring_window_days']

    max_date = df['TX_DATETIME'].max()
    monitoring_start = max_date - pd.Timedelta(days=monitoring_days)
    baseline_start = max_date - pd.Timedelta(days=baseline_days)

    # 1. Monitoring window data (most recent 7 days)
    monitoring_df = df[df['TX_DATETIME'] >= monitoring_start]
    # 2. Baseline window data (previous 30 days up to monitoring_start)
    baseline_df = df[(df['TX_DATETIME'] >= baseline_start) & (df['TX_DATETIME'] < monitoring_start)]

    # Group by terminal ID
    mon_stats = monitoring_df.groupby('TERMINAL_ID').agg(
        mon_tx_count=('TX_FRAUD', 'count'),
        mon_fraud_count=('TX_FRAUD', 'sum'),
        mon_fraud_rate=('TX_FRAUD', 'mean')
    ).reset_index()

    base_stats = baseline_df.groupby('TERMINAL_ID').agg(
        base_tx_count=('TX_FRAUD', 'count'),
        base_fraud_count=('TX_FRAUD', 'sum'),
        base_fraud_rate=('TX_FRAUD', 'mean')
    ).reset_index()

    # Merge
    merged = pd.merge(mon_stats, base_stats, on='TERMINAL_ID', how='outer').fillna(0)

    # Spike ratio with smoothing epsilon = 0.001 to prevent division by zero
    eps = 0.001
    merged['spike_ratio'] = (merged['mon_fraud_rate'] + eps) / (merged['base_fraud_rate'] + eps)

    # Apply volume guard (min_tx_count)
    def determine_status(row):
        if row['mon_tx_count'] < min_tx:
            return 'INSUFFICIENT_VOLUME'
        elif row['spike_ratio'] >= high_thresh:
            return 'ALERT_HIGH'
        elif row['spike_ratio'] >= elevated_thresh:
            return 'ALERT_ELEVATED'
        else:
            return 'NORMAL'

    merged['alert_status'] = merged.apply(determine_status, axis=1)
    
    # Sort by spike ratio descending
    result = merged.sort_values(by='spike_ratio', ascending=False).reset_index(drop=True)
    return result

if __name__ == "__main__":
    from dataset_loader import load_and_combine_data
    df = load_and_combine_data()
    spikes = detect_merchant_spikes(df)
    print("\nMerchant Spike Detector Summary:")
    print(spikes['alert_status'].value_counts())
    print("\nTop 5 High Spike Terminals:")
    print(spikes[spikes['alert_status'] == 'ALERT_HIGH'].head())
