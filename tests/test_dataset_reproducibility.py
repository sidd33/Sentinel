import os
import sys
import yaml
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dashboard_data import load_config_data, load_dataset_summary, load_experiment_manifest

def test_dataset_scope_consistency():
    config = load_config_data()
    summary = load_dataset_summary()
    manifest = load_experiment_manifest()

    assert config is not None, "config.yaml must exist"
    assert summary is not None, "dataset_summary.json must exist"
    assert manifest is not None, "experiment_manifest.json must exist"

    config_subset_days = config['dataset']['subset_days']
    summary_subset_days = summary['subset_days']
    manifest_subset_days = manifest['subset_days']

    # 1. Verify subset_days consistency
    assert config_subset_days == 60, f"config.yaml subset_days must be 60, got {config_subset_days}"
    assert summary_subset_days == 60, f"dataset_summary.json subset_days must be 60, got {summary_subset_days}"
    assert manifest_subset_days == 60, f"experiment_manifest.json subset_days must be 60, got {manifest_subset_days}"

    # 2. Verify total transaction count consistency
    actual_total = summary['total_transactions']
    manifest_total = manifest['total_rows_processed']
    assert actual_total == 575755, f"Expected 575,755 transactions, got {actual_total}"
    assert manifest_total == 575755, f"Manifest expected 575,755 transactions, got {manifest_total}"

    # 3. Verify date range consistency
    assert summary['min_timestamp'].startswith("2018-04-01"), f"Start date mismatch: {summary['min_timestamp']}"
    assert summary['max_timestamp'].startswith("2018-05-30"), f"End date mismatch: {summary['max_timestamp']}"
    assert manifest['start_date'].startswith("2018-04-01"), f"Manifest start date mismatch: {manifest['start_date']}"
    assert manifest['end_date'].startswith("2018-05-30"), f"Manifest end date mismatch: {manifest['end_date']}"

def test_actual_data_files_count():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        assert len(files) == 60, f"Expected 60 daily pickle files in data/, found {len(files)}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
