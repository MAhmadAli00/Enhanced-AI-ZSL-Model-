import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    """
    Loads data from CSV and handles potential NaNs.
    """
    print(f"Loading data from {filepath}...")
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please check the path.")

    # Handle NaNs in 'Class'
    if data['Class'].isnull().any():
        print(f"Warning: Found {data['Class'].isnull().sum()} NaNs in 'Class' column. Dropping rows with missing labels.")
        data = data.dropna(subset=['Class'])
    
    # Handle NaNs in features (impute with 0 or mean? For now, 0 or drop)
    # Given the nature of system calls (frequencies), 0 is often appropriate for missing values if sparse.
    # But let's check count first.
    if data.isnull().any().any():
         print("Warning: NaNs found in features. Filling with 0.")
         data = data.fillna(0)

    X = data.drop('Class', axis=1)
    y = data['Class']
    
    return X, y

def get_zsl_split(X, y, unseen_classes, random_state=42):
    """
    Splits data into SEEN (Training) and UNSEEN (Testing) sets based on CLASS LABELS.
    This ensures proper Zero-Shot Learning evaluation.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        unseen_classes (list): List of class labels to be held out.
        random_state (int): Random seed.
        
    Returns:
        X_seen (np.array): Scaled features for seen classes
        y_seen (pd.Series): Labels for seen classes
        X_unseen (np.array): Scaled features for unseen classes
        y_unseen (pd.Series): Labels for unseen classes
        scaler (StandardScaler): Fitted scaler
    """
    print(f"\n--- performing ZSL Split ---")
    print(f"Designated Unseen Classes: {unseen_classes}")
    
    # Create mask
    mask_unseen = y.isin(unseen_classes)
    
    X_unseen = X[mask_unseen]
    y_unseen = y[mask_unseen]
    
    X_seen = X[~mask_unseen]
    y_seen = y[~mask_unseen]
    
    # Verify split
    seen_labels = np.unique(y_seen)
    unseen_labels_present = np.unique(y_unseen)
    
    print(f"Seen Classes in Split: {seen_labels}")
    print(f"Unseen Classes in Split: {unseen_labels_present}")
    
    # Critical Check
    intersect = set(seen_labels).intersection(set(unseen_labels_present))
    if intersect:
        raise ValueError(f"CRITICAL ERROR: Data Leakage! Classes {intersect} are present in bot SEEN and UNSEEN sets.")
    
    if len(X_unseen) == 0:
        print("WARNING: No data found for the specified unseen classes. ZSL evaluation will not be possible.")
        
    # Standardize
    # Fit ONLY on SEEN data to emulate real-world scenario
    scaler = StandardScaler()
    X_seen_scaled = scaler.fit_transform(X_seen)
    
    if len(X_unseen) > 0:
        X_unseen_scaled = scaler.transform(X_unseen)
    else:
        X_unseen_scaled = np.array([])
        
    return X_seen_scaled, y_seen, X_unseen_scaled, y_unseen, scaler

def prepare_training_data(X_seen, y_seen, validation_split=0.2, random_state=42):
    """
    Splits SEEN data into Train and Validation sets and applies SMOTE.
    """
    print("\n--- Preparing Training Data (Seen Classes) ---")
    
    # Stratified Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_seen, y_seen, 
        test_size=validation_split, 
        stratify=y_seen, 
        random_state=random_state
    )
    
    print(f"Training shapes before SMOTE: X={X_train.shape}, y={y_train.shape}")
    
    # SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"Training shapes after SMOTE: X={X_train_res.shape}, y={y_train_res.shape}")
    print(f"Validation shapes: X={X_val.shape}, y={y_val.shape}")
    
    return X_train_res, y_train_res, X_val, y_val
