import os
import json
import yaml
import pandas as pd
import numpy as np

def load_config_data(config_path="configs/config.yaml"):
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_dataset_summary(data_dir="data"):
    """
    Computes or loads dynamic dataset summary metrics directly from actual data.
    Returns dict containing total, legitimate, fraud counts, percentages, unique entities, and date range.
    """
    summary_path = "artifacts/dataset_summary.json"
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    from src.dataset_loader import load_and_combine_data
    try:
        df = load_and_combine_data(data_dir=data_dir)
        total = len(df)
        fraud = int(df['TX_FRAUD'].sum())
        legit = int(total - fraud)
        pct = float((fraud / total) * 100) if total > 0 else 0.0
        n_cust = int(df['CUSTOMER_ID'].nunique())
        n_term = int(df['TERMINAL_ID'].nunique())
        min_ts = str(df['TX_DATETIME'].min())
        max_ts = str(df['TX_DATETIME'].max())

        summary = {
            "total_transactions": total,
            "fraud_count": fraud,
            "legitimate_count": legit,
            "fraud_percentage": pct,
            "unique_customers": n_cust,
            "unique_terminals": n_term,
            "min_timestamp": min_ts,
            "max_timestamp": max_ts
        }
        
        os.makedirs("artifacts", exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        return summary
    except Exception as e:
        return None

def load_experiment_manifest(manifest_path="artifacts/experiment_manifest.json"):
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r") as f:
        return json.load(f)

def load_model_metrics(metrics_path="artifacts/metrics.json"):
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)

def load_final_test_metrics(metrics_path="artifacts/metrics.json"):
    metrics = load_model_metrics(metrics_path)
    if metrics and 'test_metrics' in metrics:
        return metrics['test_metrics']
    return None

def load_threshold_analysis(metrics_path="artifacts/metrics.json"):
    metrics = load_model_metrics(metrics_path)
    manifest = load_experiment_manifest()
    if metrics and manifest and 'cost_curves' in metrics:
        winning_model = manifest.get('winning_model')
        curve = metrics['cost_curves'].get(winning_model, [])
        return curve, winning_model, manifest.get('locked_threshold')
    return None, None, None

def load_spike_results():
    from src.dataset_loader import load_and_combine_data
    from src.spike_detector import detect_merchant_spikes
    try:
        df = load_and_combine_data()
        return detect_merchant_spikes(df)
    except Exception:
        return None
