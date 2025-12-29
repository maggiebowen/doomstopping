import numpy as np
import pandas as pd

def normalize_subjects(df_train, df_test, feature_cols, label_col='label', baseline_label=1, subject_col='subject'):
    """
    Normalizes features using subject-specific baseline statistics (Robust).
    
    LOGIC:
    - Train: (X - mu_sub_train_base) / sigma_sub_train_base
      (Fallback: mu_sub_train_all)
    - Test:  (X - mu_sub_test_base) / sigma_sub_test_base
      (Fallback: mu_GLOBAL_train_base)
      
    Safety:
    - std = std + eps (never zero)
    
    Args:
        df_train (pd.DataFrame): Training data
        df_test (pd.DataFrame): Test data
        feature_cols (list): Features to normalize
        
    Returns:
        X_train_norm (np.array), X_test_norm (np.array)
    """
    eps = 1e-9
    
    # 1. Calculate Global Training Baseline Stats (for fallback)
    train_baseline_mask = df_train[label_col] == baseline_label
    
    if train_baseline_mask.sum() > 0:
        global_mu = df_train.loc[train_baseline_mask, feature_cols].mean()
        global_sigma = df_train.loc[train_baseline_mask, feature_cols].std()
    else:
        # Ultimate fallback: Global Train Overall
        global_mu = df_train[feature_cols].mean()
        global_sigma = df_train[feature_cols].std()
        
    global_sigma = global_sigma + eps
    
    # Prepare Output DataFrames
    df_train_norm = df_train.copy()
    df_test_norm = df_test.copy()
    
    # Ensure float
    for col in feature_cols:
        df_train_norm[col] = df_train_norm[col].astype('float64')
        df_test_norm[col] = df_test_norm[col].astype('float64')

    # --- Normalize Train (Subject-wise) ---
    unique_train_subs = df_train[subject_col].unique()
    
    for sub in unique_train_subs:
        sub_mask = df_train[subject_col] == sub
        sub_df = df_train[sub_mask]
        
        # 1. Try Subject Baseline (Train)
        base_mask = sub_df[label_col] == baseline_label
        
        if base_mask.sum() > 1:
            mu = sub_df.loc[base_mask, feature_cols].mean()
            sigma = sub_df.loc[base_mask, feature_cols].std()
        else:
            # 2. Fallback: Subject Overall (Train)
            mu = sub_df[feature_cols].mean()
            sigma = sub_df[feature_cols].std()
            
        sigma = sigma + eps
        
        # Apply
        df_train_norm.loc[sub_mask, feature_cols] = (sub_df[feature_cols] - mu) / sigma

    # --- Normalize Test (Subject-wise Calibration) ---
    # We allow using the subject's *own* baseline from the test set (calibration phase)
    # If that's missing, we must NOT use Test Overall (leakage). Use Global Train Baseline.
    
    unique_test_subs = df_test[subject_col].unique()
    
    for sub in unique_test_subs:
        sub_mask = df_test[subject_col] == sub
        sub_df = df_test[sub_mask]
        
        # 1. Try Subject Baseline (Test - Calibration)
        base_mask = sub_df[label_col] == baseline_label
        
        if base_mask.sum() > 1:
            mu = sub_df.loc[base_mask, feature_cols].mean()
            sigma = sub_df.loc[base_mask, feature_cols].std()
            sigma = sigma + eps
        else:
            # 2. Fallback: Global Train Baseline
            mu = global_mu
            sigma = global_sigma
            
        # Apply
        df_test_norm.loc[sub_mask, feature_cols] = (sub_df[feature_cols] - mu) / sigma

    # Extract clean arrays
    X_train_norm = df_train_norm[feature_cols].values
    X_test_norm = df_test_norm[feature_cols].values
    
    # Final Safety Check: fill leftover NaNs with 0.0 (though unlikely with eps)
    if np.isnan(X_train_norm).any():
        X_train_norm = np.nan_to_num(X_train_norm, nan=0.0)
    if np.isnan(X_test_norm).any():
        X_test_norm = np.nan_to_num(X_test_norm, nan=0.0)
        
    return X_train_norm, X_test_norm
