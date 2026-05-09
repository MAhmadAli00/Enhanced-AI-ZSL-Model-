import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_data, get_zsl_split
from src.models import HybridClusterModel
from src.utils import evaluate_metrics

def run_supervised_baseline(X, y, model_type='rf'):
    """
    Runs a standard supervised benchmark (Upper Bound).
    Trains on ALL classes (random split).
    """
    print(f"\n--- Running Supervised Baseline ({model_type.upper()}) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'xgb':
        # XGBoost requires classes 0..N-1
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Supervised Performance (Upper Bound):")
    evaluate_metrics(y_test, y_pred, label_prefix=f"Supervised_{model_type}")

def run_raw_feature_zsl(X, y, unseen_classes):
    """
    Runs ZSL using RAW features (no Triplet Embedding).
    This validates if the Neural Network is actually useful.
    """
    print(f"\n--- Running Raw Feature ZSL Baseline ---")
    
    # 1. Split (same as main experiment)
    X_seen, y_seen, X_unseen, y_unseen, scaler = get_zsl_split(X, y, unseen_classes)
    
    # We can reuse the HybridClusterModel but pass an Identity "model" that just returns inputs
    class IdentityModel:
        def predict(self, X, verbose=0):
            return X
            
    # Mock model
    dummy_model = IdentityModel()
    
    # 2. Few-Shot Protocol (same as main)
    # We'll put this logic here briefly or import if reusable. 
    # For ablation, we'll do a quick 5-shot like evaluate.py
    
    # ... (Copying slice data logic for brevity or we can refactor evaluate to generic) ...
    # Let's import evaluate_on_unseen and pass the dummy model!
    from src.evaluate import evaluate_on_unseen
    
    print("Evaluating Raw Features (No Embedding learning)...")
    evaluate_on_unseen(dummy_model, X_unseen, y_unseen, n_support=5, n_clusters=1, method='kmeans')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='feature_vectors_syscallsbinders_frequency_5_Cat.csv')
    parser.add_argument('--unseen_classes', type=int, nargs='+', default=[4, 5])
    args = parser.parse_args()
    
    X, y = load_data(args.data_path)
    
    # 1. Supervised Upper Bound
    run_supervised_baseline(X, y, model_type='rf')
    run_supervised_baseline(X, y, model_type='xgb')
    
    # 2. Raw Feature ZSL (Lower Bound)
    run_raw_feature_zsl(X, y, args.unseen_classes)
