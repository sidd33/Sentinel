import os
import sys
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, precision_score, recall_score, f1_score

try:
    from src.dataset_loader import load_and_combine_data, load_config
    from src.feature_engineering import build_feature_pipeline, audit_leakage
except ImportError:
    from dataset_loader import load_and_combine_data, load_config
    from feature_engineering import build_feature_pipeline, audit_leakage

def split_data_chronologically(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Splits DataFrame strictly by time.
    Earliest 70% = train, next 15% = val, latest 15% = test.
    """
    df = df.sort_values('TX_DATETIME').reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    split_summary = {
        'train_start': str(train_df['TX_DATETIME'].min()),
        'train_end': str(train_df['TX_DATETIME'].max()),
        'validation_start': str(val_df['TX_DATETIME'].min()),
        'validation_end': str(val_df['TX_DATETIME'].max()),
        'test_start': str(test_df['TX_DATETIME'].min()),
        'test_end': str(test_df['TX_DATETIME'].max()),
        'train_count': len(train_df),
        'validation_count': len(val_df),
        'test_count': len(test_df),
        'train_fraud_count': int(train_df['TX_FRAUD'].sum()),
        'validation_fraud_count': int(val_df['TX_FRAUD'].sum()),
        'test_fraud_count': int(test_df['TX_FRAUD'].sum()),
        'train_fraud_rate': float(train_df['TX_FRAUD'].mean()),
        'validation_fraud_rate': float(val_df['TX_FRAUD'].mean()),
        'test_fraud_rate': float(test_df['TX_FRAUD'].mean())
    }

    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/split_summary.json', 'w') as f:
        json.dump(split_summary, f, indent=2)

    print(f"[Data Split] Chronological temporal split executed:")
    print(f" -> Train:      {len(train_df):,} rows ({train_df['TX_DATETIME'].min()} to {train_df['TX_DATETIME'].max()}) | Fraud rate: {train_df['TX_FRAUD'].mean()*100:.3f}%")
    print(f" -> Validation: {len(val_df):,} rows ({val_df['TX_DATETIME'].min()} to {val_df['TX_DATETIME'].max()}) | Fraud rate: {val_df['TX_FRAUD'].mean()*100:.3f}%")
    print(f" -> Held Test:  {len(test_df):,} rows ({test_df['TX_DATETIME'].min()} to {test_df['TX_DATETIME'].max()}) | Fraud rate: {test_df['TX_FRAUD'].mean()*100:.3f}%")

    return train_df, val_df, test_df

def compute_cost_and_metrics(y_true, y_probs, fp_cost=1.0, fn_cost=10.0, num_thresholds=100):
    """
    Evaluates cost curve across decision thresholds t in [0.01, 0.99].
    Primary objective: Find threshold t* that minimizes total expected cost.
    """
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_cost = float('inf')
    best_thresh = 0.5
    best_metrics = {}
    cost_curve = []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        total_cost = (fp * fp_cost) + (fn * fn_cost)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        cost_curve.append({
            'threshold': float(t),
            'cost': float(total_cost),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'tn': int(tn),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        })

        if total_cost < best_cost:
            best_cost = total_cost
            best_thresh = float(t)
            best_metrics = cost_curve[-1]

    # Compute PR-AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = float(auc(recall_vals, precision_vals))
    best_metrics['pr_auc'] = pr_auc

    return best_thresh, best_metrics, cost_curve

def train_and_evaluate_all():
    config = load_config()
    fp_cost = config['cost']['false_positive']
    fn_cost = config['cost']['false_negative']
    seed = config['random_seed']

    # 1. Load dataset
    df = load_and_combine_data()

    # 2. Build feature engineering pipeline & leakage audit
    df_feat = build_feature_pipeline(df, label_delay_days=7)
    audit_leakage(df_feat)

    feature_cols = [
        'TX_AMOUNT', 'log_tx_amount', 'hour', 'day_of_week', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT',
        'CUSTOMER_ID_NB_TX_1DAY', 'CUSTOMER_ID_AVG_AMOUNT_1DAY',
        'CUSTOMER_ID_NB_TX_7DAY', 'CUSTOMER_ID_AVG_AMOUNT_7DAY',
        'CUSTOMER_ID_NB_TX_30DAY', 'CUSTOMER_ID_AVG_AMOUNT_30DAY',
        'TERMINAL_ID_NB_TX_1DAY', 'TERMINAL_ID_RISK_7DAY_DELAYED', 'TERMINAL_ID_RISK_30DAY_DELAYED'
    ]
    target_col = 'TX_FRAUD'

    # 3. Chronological split
    train_df, val_df, test_df = split_data_chronologically(
        df_feat, 
        train_ratio=config['split']['train'],
        val_ratio=config['split']['validation'],
        test_ratio=config['split']['test']
    )

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # Fill NaNs if any
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)

    # 4. Model Candidates Definition
    models = {
        "Logistic Regression": LogisticRegression(
            C=config['models']['logistic_regression']['C'],
            max_iter=config['models']['logistic_regression']['max_iter'],
            class_weight=config['models']['logistic_regression']['class_weight'],
            random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=config['models']['random_forest']['n_estimators'],
            max_depth=config['models']['random_forest']['max_depth'],
            min_samples_split=config['models']['random_forest']['min_samples_split'],
            class_weight=config['models']['random_forest']['class_weight'],
            random_state=seed,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=config['models']['xgboost']['n_estimators'],
            max_depth=config['models']['xgboost']['max_depth'],
            learning_rate=config['models']['xgboost']['learning_rate'],
            scale_pos_weight=config['models']['xgboost']['scale_pos_weight'],
            random_state=seed,
            n_jobs=-1
        )
    }

    val_results = {}
    cost_curves = {}
    best_model_name = None
    lowest_val_cost = float('inf')
    best_locked_threshold = 0.5
    fitted_models = {}

    print("\n" + "="*80)
    print("TRAINING MODELS & VALIDATION COST MINIMIZATION THRESHOLD SELECTION")
    print("="*80)

    for name, model in models.items():
        print(f"\n[Trainer] Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Validation prediction probabilities
        val_probs = model.predict_proba(X_val)[:, 1]

        # Cost-based threshold selection on Validation data
        opt_thresh, best_val_metric, curve = compute_cost_and_metrics(y_val, val_probs, fp_cost=fp_cost, fn_cost=fn_cost)
        
        val_results[name] = {
            'optimal_threshold': opt_thresh,
            'val_metrics': best_val_metric
        }
        cost_curves[name] = curve

        print(f" -> {name} Validation Results at Optimal Threshold ({opt_thresh:.2f}):")
        print(f"    Total Expected Cost: ${best_val_metric['cost']:,.2f}")
        print(f"    PR-AUC:    {best_val_metric['pr_auc']:.4f}")
        print(f"    Precision: {best_val_metric['precision']:.4f}")
        print(f"    Recall:    {best_val_metric['recall']:.4f}")
        print(f"    F1 Score:  {best_val_metric['f1']:.4f}")
        print(f"    Confusion Matrix: TP={best_val_metric['tp']}, FP={best_val_metric['fp']}, FN={best_val_metric['fn']}, TN={best_val_metric['tn']}")

        if best_val_metric['cost'] < lowest_val_cost:
            lowest_val_cost = best_val_metric['cost']
            best_model_name = name
            best_locked_threshold = opt_thresh

    print("\n" + "="*80)
    print(f"WINNING MODEL SELECTED: {best_model_name}")
    print(f"LOCKED OPTIMAL THRESHOLD: {best_locked_threshold:.2f} (Validation Cost: ${lowest_val_cost:,.2f})")
    print("="*80)

    # 5. Final Evaluation strictly ONCE on untouched Test Set
    winning_model = fitted_models[best_model_name]
    test_probs = winning_model.predict_proba(X_test)[:, 1]

    test_preds = (test_probs >= best_locked_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    test_cost = (fp * fp_cost) + (fn * fn_cost)
    test_prec = float(precision_score(y_test, test_preds, zero_division=0))
    test_rec = float(recall_score(y_test, test_preds, zero_division=0))
    test_f1 = float(f1_score(y_test, test_preds, zero_division=0))

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, test_probs)
    test_pr_auc = float(auc(recall_vals, precision_vals))

    test_metrics = {
        'model_name': best_model_name,
        'locked_threshold': best_locked_threshold,
        'total_cost': float(test_cost),
        'pr_auc': test_pr_auc,
        'precision': test_prec,
        'recall': test_rec,
        'f1': test_f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'test_size': len(test_df),
        'fraud_count': int(y_test.sum())
    }

    print("\n" + "="*80)
    print("HELD-OUT TEST SET FINAL UNTOUCHED EVALUATION RESULTS")
    print("="*80)
    print(f" Model Name:        {test_metrics['model_name']}")
    print(f" Locked Threshold:  {test_metrics['locked_threshold']:.2f}")
    print(f" Test Total Cost:   ${test_metrics['total_cost']:,.2f}")
    print(f" Test PR-AUC:       {test_metrics['pr_auc']:.4f}")
    print(f" Test Precision:    {test_metrics['precision']:.4f}")
    print(f" Test Recall:       {test_metrics['recall']:.4f}")
    print(f" Test F1 Score:     {test_metrics['f1']:.4f}")
    print(f" Test Confusion:    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print("="*80 + "\n")

    # Verify dataset counts before training
    total_tx = len(df_feat)
    fraud_tx = int(df_feat[target_col].sum())
    legit_tx = total_tx - fraud_tx

    print(f"[Trainer Check] Total Transactions: {total_tx:,} | Fraud: {fraud_tx:,} | Legitimate: {legit_tx:,}")
    if total_tx != 575755 or fraud_tx != 4263 or legit_tx != 571492:
        raise ValueError(f"CRITICAL DISCREPANCY: Dataset counts mismatch! Expected 575,755 / 4,263 / 571,492. Got {total_tx} / {fraud_tx} / {legit_tx}.")

    # Generate Run Manifest
    import platform
    import sklearn
    import xgboost as xgb
    import datetime

    run_manifest = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgb.__version__,
        "joblib_version": joblib.__version__,
        "random_seed": seed,
        "dataset_scope": "60 chronological days (2018-04-01 through 2018-05-30)",
        "total_transactions": total_tx,
        "fraud_count": fraud_tx,
        "legitimate_count": legit_tx,
        "feature_list": feature_cols,
        "model_candidates": list(models.keys()),
        "fp_cost": fp_cost,
        "fn_cost": fn_cost
    }

    # 6. Save Artifacts & Models
    os.makedirs('models', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)

    joblib.dump(winning_model, 'models/winning_model.joblib')
    joblib.dump(feature_cols, 'models/feature_cols.joblib')

    output_metrics = {
        'validation_results': val_results,
        'cost_curves': cost_curves,
        'test_metrics': test_metrics
    }
    with open('artifacts/metrics.json', 'w') as f:
        json.dump(output_metrics, f, indent=2)

    with open('artifacts/run_manifest.json', 'w') as f:
        json.dump(run_manifest, f, indent=2)

    # 7. Generate Experiment Manifest
    manifest = {
        "dataset": "Official Fraud Detection Handbook simulated dataset (60-day chronological subset)",
        "dataset_source": "Official Fraud Detection Handbook simulated transaction dataset",
        "dataset_type": "simulated",
        "subset_days": 60,
        "start_date": "2018-04-01 00:00:31",
        "end_date": "2018-05-30 23:59:45",
        "total_rows_processed": len(df_feat),
        "split": "70/15/15 chronological temporal split",
        "random_seed": seed,
        "features": feature_cols,
        "models_evaluated": list(models.keys()),
        "threshold_selection_objective": "Validation set business cost minimization",
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
        "winning_model": best_model_name,
        "locked_threshold": best_locked_threshold,
        "test_pr_auc": test_metrics['pr_auc'],
        "test_precision": test_metrics['precision'],
        "test_recall": test_metrics['recall'],
        "test_f1": test_metrics['f1'],
        "test_total_cost": test_metrics['total_cost'],
        "test_set_locked": True
    }
    with open('artifacts/experiment_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print("[Trainer] Model artifacts, metrics.json, run_manifest.json, and experiment_manifest.json generated successfully.")

if __name__ == "__main__":
    train_and_evaluate_all()
