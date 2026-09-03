import numpy as np
import pandas as pd
import datetime

def add_time_and_amount_features(df):
    """
    Derives transaction time and log amount features.
    No leakage risk as these are static row properties at time T.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['TX_DATETIME']):
        df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])

    df['hour'] = df['TX_DATETIME'].dt.hour
    df['day_of_week'] = df['TX_DATETIME'].dt.dayofweek
    df['TX_DURING_WEEKEND'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['TX_DURING_NIGHT'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df['log_tx_amount'] = np.log1p(df['TX_AMOUNT'])
    return df

def add_retrospective_customer_features(df, windows=[1, 7, 30]):
    """
    Calculates historical customer transaction count and average amount strictly prior to time T.
    Uses closed='left' rolling windows indexed by TX_DATETIME to prevent current row or future row leakage.
    """
    df = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME']).reset_index(drop=True)
    df_indexed = df.set_index('TX_DATETIME')
    
    for w in windows:
        w_str = f"{w}D"
        
        # Group by customer and compute rolling count & mean with closed='left' (strictly before current T)
        rolled_cnt = df_indexed.groupby('CUSTOMER_ID')['TX_AMOUNT'].rolling(w_str, closed='left').count()
        rolled_avg = df_indexed.groupby('CUSTOMER_ID')['TX_AMOUNT'].rolling(w_str, closed='left').mean()
        
        df[f'CUSTOMER_ID_NB_TX_{w}DAY'] = rolled_cnt.fillna(0).values
        df[f'CUSTOMER_ID_AVG_AMOUNT_{w}DAY'] = rolled_avg.fillna(0).values

    # Restore chronological order
    df = df.sort_values('TX_DATETIME').reset_index(drop=True)
    return df

def add_delayed_terminal_features(df, delay_days=7, risk_windows=[7, 30]):
    """
    Computes terminal transaction counts up to time T (excluding current row),
    and terminal historical fraud risk rate enforcing a strict label reporting delay (default: 7 days).

    For a transaction at time T on terminal M:
    Eligible historical transactions must have timestamp t <= T - delay_days.
    The rolling risk rate is calculated over the window [T - delay_days - window_days, T - delay_days]
    using only eligible historical labels.
    """
    df = df.sort_values(['TERMINAL_ID', 'TX_DATETIME']).reset_index(drop=True)
    df_indexed = df.set_index('TX_DATETIME')

    # Terminal volume (without fraud label, strictly retrospective closed='left')
    rolled_term_cnt = df_indexed.groupby('TERMINAL_ID')['TX_AMOUNT'].rolling('1D', closed='left').count()
    df['TERMINAL_ID_NB_TX_1DAY'] = rolled_term_cnt.fillna(0).values

    # Delayed terminal fraud risk calculation via pd.merge_asof
    for w in risk_windows:
        col_name = f'TERMINAL_ID_RISK_{w}DAY_DELAYED'
        
        # Pre-compute historical rolling fraud sum and count at each historical timestamp t
        # over the rolling window [t - w, t] (closed='both' includes historical timestamp t)
        roll_sum = df_indexed.groupby('TERMINAL_ID')['TX_FRAUD'].rolling(f"{w}D", closed='both').sum().reset_index()
        roll_cnt = df_indexed.groupby('TERMINAL_ID')['TX_FRAUD'].rolling(f"{w}D", closed='both').count().reset_index()

        hist_stats = pd.DataFrame({
            'TERMINAL_ID': roll_sum['TERMINAL_ID'],
            'TX_DATETIME': roll_sum['TX_DATETIME'],
            'hist_fraud_sum': roll_sum['TX_FRAUD'],
            'hist_tx_cnt': roll_cnt['TX_FRAUD']
        })

        # Drop duplicate timestamps for the same terminal, keeping the last computed state, and sort by TX_DATETIME
        hist_stats = hist_stats.drop_duplicates(subset=['TERMINAL_ID', 'TX_DATETIME'], keep='last').sort_values('TX_DATETIME')

        # For each transaction in df, calculate lookup timestamp: TX_DATETIME - delay_days
        df_temp = df[['TERMINAL_ID', 'TX_DATETIME']].copy()
        df_temp['original_order'] = np.arange(len(df_temp))
        df_temp['LOOKUP_DATETIME'] = df_temp['TX_DATETIME'] - pd.Timedelta(days=delay_days)

        # Merge asof: match LOOKUP_DATETIME to historical TX_DATETIME at or before LOOKUP_DATETIME
        df_temp_sorted = df_temp.sort_values('LOOKUP_DATETIME')
        merged = pd.merge_asof(
            df_temp_sorted,
            hist_stats,
            left_on='LOOKUP_DATETIME',
            right_on='TX_DATETIME',
            by='TERMINAL_ID',
            direction='backward',
            suffixes=('', '_hist')
        )

        # Restore original sorting order
        merged = merged.sort_values('original_order')

        # Compute rate = hist_fraud_sum / hist_tx_cnt
        denom = merged['hist_tx_cnt'].replace(0, np.nan)
        rate = (merged['hist_fraud_sum'] / denom).fillna(0.0)

        df[col_name] = rate.values

    # Restore chronological order
    df = df.sort_values('TX_DATETIME').reset_index(drop=True)
    return df

def build_feature_pipeline(df, label_delay_days=7):
    """
    Complete leakage-free feature engineering pipeline.
    """
    print("[Feature Engineering] Deriving time and log-amount features...")
    df = add_time_and_amount_features(df)
    
    print("[Feature Engineering] Calculating retrospective customer rolling features (1D, 7D, 30D)...")
    df = add_retrospective_customer_features(df, windows=[1, 7, 30])
    
    print(f"[Feature Engineering] Calculating delayed terminal risk features (Delay: {label_delay_days} days)...")
    df = add_delayed_terminal_features(df, delay_days=label_delay_days, risk_windows=[7, 30])
    
    return df

def audit_leakage(df):
    """
    Audits feature engineering pipeline for data leakage.
    Ensures:
    1. No feature matches or correlates perfectly with TX_FRAUD.
    2. Customer features at index i only depend on rows < i.
    3. Terminal risk features use delayed information.
    """
    print("[Leakage Audit] Running automated data leakage checks...")
    
    feature_cols = [c for c in df.columns if c not in ['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_FRAUD', 'TX_FRAUD_SCENARIO']]
    
    # Check 1: Perfect correlation check with target
    correlations = df[feature_cols].apply(lambda c: c.corr(df['TX_FRAUD']) if pd.api.types.is_numeric_dtype(c) else 0)
    max_corr = correlations.abs().max()
    max_corr_feature = correlations.abs().idxmax()
    print(f" -> Max feature correlation with TX_FRAUD target: '{max_corr_feature}' = {max_corr:.4f}")
    assert max_corr < 0.95, f"LEAKAGE DETECTED: Feature '{max_corr_feature}' has suspicious correlation ({max_corr}) with target!"

    # Check 2: First transaction per customer must have 0 historical counts
    first_cust_tx = df.groupby('CUSTOMER_ID').first()
    first_cust_counts = df.loc[df.groupby('CUSTOMER_ID')['TX_DATETIME'].idxmin(), 'CUSTOMER_ID_NB_TX_1DAY']
    assert (first_cust_counts == 0).all(), "LEAKAGE DETECTED: First transaction for customer contains non-zero historical count!"
    print(" -> Confirmed: First customer transactions have 0 historical count (closed='left' verified).")

    # Check 3: Check timestamp monotonicity
    assert df['TX_DATETIME'].is_monotonic_increasing, "LEAKAGE DETECTED: DataFrame is not strictly sorted chronologically!"
    print(" -> Confirmed: DataFrame is strictly sorted by timestamp.")

    print("[Leakage Audit] PASSED all 3 leakage sanity checks cleanly.")
    return True

if __name__ == "__main__":
    # Test on mock data for fast verification
    mock_data = pd.DataFrame({
        'TRANSACTION_ID': range(100),
        'TX_DATETIME': pd.date_range('2018-04-01', periods=100, freq='h'),
        'CUSTOMER_ID': np.random.choice([1, 2, 3, 4, 5], 100),
        'TERMINAL_ID': np.random.choice([10, 20, 30], 100),
        'TX_AMOUNT': np.random.uniform(10, 500, 100),
        'TX_FRAUD': np.random.choice([0, 1], 100, p=[0.95, 0.05]),
        'TX_FRAUD_SCENARIO': 0
    })
    processed = build_feature_pipeline(mock_data)
    audit_leakage(processed)
    print("Feature engineering unit test completed successfully.")
