import os
# Set environment variables BEFORE importing other massive libraries for maximum stability
os.environ['PYTHONHASHSEED'] = '42'

import argparse
import numpy as np
import tensorflow as tf

# NUCLEAR OPTION for Reproducibility
# 1. Set global distinct seeds
tf.keras.utils.set_random_seed(42)

# 2. Enable Op Determinism (Guarantees bit-exact results on CPU/GPU, slight pert hit)
try:
    tf.config.experimental.enable_op_determinism()
    print("TensorFlow Op Determinism Enabled")
except AttributeError:
    print("Warning: tf.config.experimental.enable_op_determinism() not found.")

from src.data_loader import load_data, get_zsl_split, prepare_training_data
from src.train import train_embedding_model
from src.evaluate import evaluate_on_unseen

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot/Few-Shot Malware Detection")
    parser.add_argument('--data_path', type=str, default='feature_vectors_syscallsbinders_frequency_5_Cat.csv', help='Path to dataset CSV')
    parser.add_argument('--unseen_classes', type=int, nargs='+', default=[4, 5], help='List of class IDs to hold out as Unseen (e.g., 4 5)')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--method', type=str, default='kmeans', choices=['kmeans', 'gmm', 'agglomerative'], help='Clustering method')
    parser.add_argument('--n_clusters', type=int, default=1, help='Prototypes per class')
    parser.add_argument('--eda', action='store_true', help='Run Exploratory Data Analysis (EDA) and save plots')
    
    args = parser.parse_args()
    
    # 0. Run EDA if requested
    if args.eda:
        print("\n--- Running Exploratory Data Analysis ---")
        from src.visualization import plot_class_distribution, plot_feature_importance, plot_correlation_heatmap, plot_tsne_2d, plot_pca_3d
        
        # Load raw data for EDA
        X_raw, y_raw = load_data(args.data_path)
        os.makedirs("plots", exist_ok=True)
        
        plot_class_distribution(y_raw, save_path="plots/class_distribution.png")
        
        # Feature Importance
        top_features = plot_feature_importance(X_raw, y_raw, top_n=10, save_path="plots/feature_importance.png")
        print(f"Top Features identified: {top_features}")
        
        # Correlation on top features
        plot_correlation_heatmap(X_raw, features=top_features, save_path="plots/correlation_heatmap.png")
        
        # t-SNE
        plot_tsne_2d(X_raw, y_raw, save_path="plots/tsne_2d.png")
        
        # PCA 3D
        plot_pca_3d(X_raw, y_raw, save_path="plots/pca_3d.png")
        
        print("EDA Plotting complete. Check 'plots/' directory.\n")
        
    # 1. Load Data
    X, y = load_data(args.data_path)
    
    # 2. ZSL Split (Disjoint)
    # Note: 'Class' column likely has integers 1, 2, 3, 4, 5.
    X_seen, y_seen, X_unseen, y_unseen, scaler = get_zsl_split(X, y, unseen_classes=args.unseen_classes)
    
    # 3. Prepare Training Data (from Seen)
    X_train, y_train, X_val, y_val = prepare_training_data(X_seen, y_seen)
    
    # 4. Train Embedding Model
    input_shape = (X_train.shape[1],)
    embedding_model, history = train_embedding_model(X_train, y_train, X_val, y_val, input_shape, epochs=args.epochs)
    
    # 4a. Plot Training History
    from src.visualization import plot_training_history
    plot_training_history(history, save_dir="plots")
    
    # Save model
    embedding_model.save("malware_embedding_model.h5")
    print("Model saved to malware_embedding_model.h5")
    
    # 5. Evaluation (ZSL/Few-Shot on Unseen)
    evaluate_on_unseen(embedding_model, X_unseen, y_unseen, n_support=5, n_clusters=args.n_clusters, method=args.method)
    
if __name__ == "__main__":
    main()
