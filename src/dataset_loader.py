import os
import sys
import yaml
import datetime
import urllib.request
import pandas as pd
import numpy as np

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_dataset(config=None, output_dir="data"):
    """
    Downloads official daily pickle files from the Fraud Detection Handbook repository.
    Uses the configured subset_days parameter (default: 60 days).
    Strictly uses official benchmark data. Does NOT generate synthetic replacement data.
    """
    if config is None:
        config = load_config()

    os.makedirs(output_dir, exist_ok=True)
    base_url = config["dataset"]["repo_url"]
    start_date_str = config["dataset"]["start_date"]
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    subset_days = config["dataset"].get("subset_days", 60)

    print(f"[Dataset Loader] Fetching {subset_days} days of official Fraud Detection Handbook transformed data ({start_date_str} onwards)...")
    
    downloaded_files = []
    failed_dates = []

    for day_offset in range(subset_days):
        current_date = start_date + datetime.timedelta(days=day_offset)
        date_str = current_date.strftime("%Y-%m-%d")
        file_name = f"{date_str}.pkl"
        file_path = os.path.join(output_dir, file_name)

        if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
            downloaded_files.append(file_path)
            continue

        file_url = f"{base_url}{file_name}"
        try:
            print(f" Downloading {file_name} from official repo...", end="\r")
            urllib.request.urlretrieve(file_url, file_path)
            if os.path.getsize(file_path) > 1000:
                downloaded_files.append(file_path)
            else:
                os.remove(file_path)
                failed_dates.append(date_str)
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            failed_dates.append(date_str)

    print()

    if failed_dates and len(downloaded_files) == 0:
        error_msg = (
            "\n" + "="*80 + "\n"
            "CRITICAL DATASET DOWNLOAD FAILURE:\n"
            f"Failed to download official dataset files for dates starting from {failed_dates[0]}.\n"
            "OFFICIAL DOWNLOAD INSTRUCTIONS:\n"
            "1. Visit the repository: https://github.com/Fraud-Detection-Handbook/simulated-data-transformed\n"
            "2. Download daily .pkl files into the local 'data/' directory.\n"
            "3. Re-run this script.\n"
            "NOTE: SentinelPay strictly enforces zero synthetic data substitution for final reported metrics.\n"
            + "="*80 + "\n"
        )
        raise RuntimeError(error_msg)

    print(f"[Dataset Loader] Successfully validated {len(downloaded_files)} official daily data files.")
    return downloaded_files

def load_and_combine_data(data_dir="data"):
    """
    Loads all daily .pkl files in chronological order and concatenates into a single DataFrame.
    """
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pkl")])
    if not files:
        raise FileNotFoundError(f"No .pkl files found in '{data_dir}'. Run dataset_loader.py download step first.")

    print(f"[Dataset Loader] Loading {len(files)} daily files into DataFrame...")
    dfs = [pd.read_pickle(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    df = df.sort_values('TX_DATETIME').reset_index(drop=True)
    
    print(f"[Dataset Loader] Total Dataset Loaded: {len(df):,} transactions from {df['TX_DATETIME'].min()} to {df['TX_DATETIME'].max()}")
    print(f"[Dataset Loader] Target Distribution (TX_FRAUD): {df['TX_FRAUD'].value_counts().to_dict()} (Fraud rate: {df['TX_FRAUD'].mean()*100:.3f}%)")
    return df

if __name__ == "__main__":
    download_dataset()
    df = load_and_combine_data()
    print("Dataset loading test completed successfully.")
