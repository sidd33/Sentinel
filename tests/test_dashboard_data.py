import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dashboard_data import (
    load_dataset_summary,
    load_experiment_manifest,
    load_model_metrics,
    load_final_test_metrics,
    load_threshold_analysis
)

def test_dataset_summary_loading():
    summary = load_dataset_summary()
    assert summary is not None, "Dataset summary should not be None"
    assert "total_transactions" in summary
    assert "fraud_count" in summary
    assert "legitimate_count" in summary
    assert "fraud_percentage" in summary
    assert summary["total_transactions"] == summary["fraud_count"] + summary["legitimate_count"]

def test_model_metrics_loading():
    metrics = load_model_metrics()
    assert metrics is not None, "Model metrics should not be None"
    assert "validation_results" in metrics
    assert "cost_curves" in metrics
    assert "test_metrics" in metrics

def test_final_test_metrics_loading():
    test_metrics = load_final_test_metrics()
    assert test_metrics is not None, "Final test metrics should not be None"
    assert "winning_model" in test_metrics or "model_name" in test_metrics
    assert "pr_auc" in test_metrics
    assert "precision" in test_metrics
    assert "recall" in test_metrics
    assert "f1" in test_metrics
    assert "total_cost" in test_metrics

def test_missing_artifact_graceful_handling(tmp_path):
    # Test loading from a non-existent path
    fake_path = str(tmp_path / "non_existent_metrics.json")
    metrics = load_model_metrics(metrics_path=fake_path)
    assert metrics is None, "Missing metrics should return None rather than raising an unhandled exception or returning dummy values"
    
    test_metrics = load_final_test_metrics(metrics_path=fake_path)
    assert test_metrics is None, "Missing test metrics should return None"

def test_no_hardcoded_experimental_counts_in_app():
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app.py'))
    with open(app_path, 'r', encoding='utf-8') as f:
        app_code = f.read()

    # Search for previously identified hardcoded values
    hardcoded_snippets = [
        "values = [571492, 4263]",
        "571492",
        "4263",
        "0.74%",
        "575755"
    ]
    for snippet in hardcoded_snippets:
        assert snippet not in app_code, f"Hardcoded snippet '{snippet}' found in app.py!"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
