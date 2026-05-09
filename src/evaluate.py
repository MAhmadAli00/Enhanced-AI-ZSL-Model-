from src.models import HybridClusterModel
from src.utils import evaluate_metrics, plot_confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

def evaluate_on_unseen(model, X_unseen, y_unseen, n_support=5, n_clusters=1, method='kmeans', seed=None):
    """
    Evaluates the model on UNSEEN classes using a Few-Shot protocol.

    Args:
        seed: If provided, shuffles each class's samples with this RNG seed before
              slicing into support/query sets. Pass different seeds to measure variance
              across independent support-set draws.
    """
    print(f"\n--- Evaluating on UNSEEN Classes (Few-Shot Protocol: {n_support}-shot, seed={seed}) ---")
    
    unique_classes = np.unique(y_unseen)
    if len(unique_classes) == 0:
        print("No unseen data to evaluate.")
        return
        
    # Split into Support and Query sets
    X_support_list = []
    y_support_list = []
    X_query_list = []
    y_query_list = []
    
    rng = np.random.default_rng(seed)

    for cls in unique_classes:
        # Get data for this class
        mask = (y_unseen == cls)
        X_cls = X_unseen[mask]
        y_cls = y_unseen[mask]

        # Shuffle with seed so each call with a different seed draws a different support set
        idx = rng.permutation(len(X_cls))
        X_cls = X_cls[idx]
        y_cls = np.array(y_cls)[idx]

        if len(X_cls) <= n_support:
            print(f"Warning: Class {cls} has only {len(X_cls)} samples. Using 1 for support.")
            split_idx = 1
        else:
            split_idx = n_support

        X_support_list.append(X_cls[:split_idx])
        y_support_list.append(y_cls[:split_idx])
        
        X_query_list.append(X_cls[split_idx:])
        y_query_list.append(y_cls[split_idx:])
        
    X_support = np.concatenate(X_support_list)
    y_support = np.concatenate(y_support_list)
    X_query = np.concatenate(X_query_list)
    y_query = np.concatenate(y_query_list)
    
    print(f"Support Set size: {X_support.shape[0]} samples")
    print(f"Query Set size: {X_query.shape[0]} samples")
    
    # Initialize Hybrid Model
    hybrid_model = HybridClusterModel(model, method=method, n_clusters=n_clusters)
    
    # 1. Fit Prototypes using SUPPORT Set
    hybrid_model.fit_prototypes(X_support, y_support)
    
    # 2. Predict on QUERY Set
    if len(unique_classes) == 1:
        print("\n[WARNING] Only 1 unseen class present. Classification accuracy will be trivially 100% if only that prototype exists.")
        print("To verify discrimination capability, consider holding out multiple classes (e.g., --unseen_classes 4 5)")
        print("or implementing Generalized ZSL (mixing seen and unseen in query).")
    
    y_pred = hybrid_model.predict(X_query)
    
    # 3. Metrics
    metrics = evaluate_metrics(y_query, y_pred, label_prefix="Unseen/ZSL")
    
    # Plot
    # Plot
    import os
    os.makedirs("plots", exist_ok=True)
    seed_tag = f"_seed{seed}" if seed is not None else ""
    plot_confusion_matrix(y_query, y_pred, filename=f"plots/confusion_matrix_unseen{seed_tag}.png")
    
    return metrics
