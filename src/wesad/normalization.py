import numpy as np
import pandas as pd

def normalize_subjects(df_train, df_test, feature_cols, label_col='label', baseline_label=1, subject_col='subject'):
    """
    Normalizes features using subject-specific baseline statistics.
    
    For each subject, calculates mean and std from rows where label == baseline_label.
    Applies (X - mu) / sigma to all rows for that subject.
    
    Fallback strategy:
    - If a subject in df_train has no baseline rows, use the specific subject's overall stats (or global baseline stats if preferred, here we try global baseline first).
    - Actually, provided plan says: 
      1. Train subjects: use subject baseline. Fallback: subject overall stats.
      2. Test subjects: use subject baseline. Fallback: global train baseline stats.
      
    Let's refine the fallback to match the approved plan roughly but be robust.
    
    Args:
        df_train (pd.DataFrame): Training data containing features, labels, and subject IDs.
        df_test (pd.DataFrame): Test data containing features, labels, and subject IDs.
        feature_cols (list): List of feature column names to normalize.
        label_col (str): Name of the label column.
        baseline_label (int/str): Value indicating baseline condition.
        subject_col (str): Name of the subject column.
        
    Returns:
        X_train_norm (np.array): Normalized training features.
        X_test_norm (np.array): Normalized test features.
    """
    
    # 1. Calculate Global Training Baseline Stats (for fallback)
    # We use ALL train baseline rows.
    train_baseline_mask = df_train[label_col] == baseline_label
    if train_baseline_mask.sum() > 0:
        global_mu = df_train.loc[train_baseline_mask, feature_cols].mean()
        global_sigma = df_train.loc[train_baseline_mask, feature_cols].std()
    else:
        # If absolutely no baseline in train, fallback to global train overall
        print("Warning: No baseline samples found in entire training set! Using global training mean/std.")
        global_mu = df_train[feature_cols].mean()
        global_sigma = df_train[feature_cols].std()
    
    # Ensure no zero division
    global_sigma = global_sigma.replace(0, 1.0)
    
    # helper to normalize a single subject's df
    def get_subject_stats(sub_df, is_train=True):
        # Identify baseline rows
        base_mask = sub_df[label_col] == baseline_label
        
        if base_mask.sum() > 0:
            mu = sub_df.loc[base_mask, feature_cols].mean()
            sigma = sub_df.loc[base_mask, feature_cols].std()
        else:
            # Fallback Logic
            if is_train:
                # Plan says: fallback to subject's overall mean/std on df_train
                # But typically global baseline is safer if we assume baselines are similar.
                # However, plan explicitly said: "fallback to subject's overall mean/std on df_train rows"
                print(f"Warning: Subject {sub_df[subject_col].iloc[0]} (Train) has no baseline. Using subject overall stats.")
                mu = sub_df[feature_cols].mean()
                sigma = sub_df[feature_cols].std()
            else:
                # Test subjects: fallback to global training baseline stats
                print(f"Warning: Subject {sub_df[subject_col].iloc[0]} (Test) has no baseline. Using Global Train Baseline stats.")
                mu = global_mu
                sigma = global_sigma
        
        # Handle zero std
        sigma = sigma.replace(0, 1.0)
        return mu, sigma

    # Apply to Train
    # We can group by subject and apply transform
    # Better to iterate to print warnings cleanly and control fallback
    
    
    # Create copies and ensure feature columns are float to avoid FutureWarning/SettingWithCopy
    # when assigning normalized values (floats) to integer columns (e.g. peaks count).
    df_train_norm = df_train.copy()
    for col in feature_cols:
        if df_train_norm[col].dtype != 'float64':
            df_train_norm[col] = df_train_norm[col].astype('float64')
            
    unique_train_subs = df_train[subject_col].unique()
    
    for sub in unique_train_subs:
        sub_mask = df_train[subject_col] == sub
        sub_df = df_train[sub_mask]
        
        mu, sigma = get_subject_stats(sub_df, is_train=True)
        
        # Apply normalization
        # (X - mu) / sigma
        # We need to ensure alignment. mu/sigma are Series with index=feature_cols.
        # sub_df[feature_cols] is DataFrame.
        
        # Update values in the main dataframe
        # Note: Doing this in a loop on the main df can be slow but fine for 15 subjects.
        df_train_norm.loc[sub_mask, feature_cols] = (sub_df[feature_cols] - mu) / sigma

    # Apply to Test
    df_test_norm = df_test.copy()
    for col in feature_cols:
        if df_test_norm[col].dtype != 'float64':
            df_test_norm[col] = df_test_norm[col].astype('float64')
            
    unique_test_subs = df_test[subject_col].unique()
    
    for sub in unique_test_subs:
        sub_mask = df_test[subject_col] == sub
        sub_df = df_test[sub_mask]
        
        mu, sigma = get_subject_stats(sub_df, is_train=False)
        
        df_test_norm.loc[sub_mask, feature_cols] = (sub_df[feature_cols] - mu) / sigma

    # Extract X matrices
    X_train_norm = df_train_norm[feature_cols].values
    X_test_norm = df_test_norm[feature_cols].values
    
    return X_train_norm, X_test_norm
