import matplotlib
matplotlib.use('Agg') # Fix for MacOS segfaults
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def plot_class_distribution(y, title="Class Distribution", save_path=None):
    """Plots the count of each class."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()

def plot_correlation_heatmap(X, features=None, save_path=None):
    """Plots correlation heatmap for specified features (or all)."""
    if features:
        data = X[features]
    else:
        # Limit to top 20 if too many
        data = X.iloc[:, :20] 
        
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_feature_importance(X, y, top_n=10, save_path=None):
    """Trains a quick RF to find and plot feature importance."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), importances[indices], align='center')
    plt.xticks(range(top_n), [X.columns[i] for i in indices], rotation=45, ha='right')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return [X.columns[i] for i in indices]

def plot_tsne_2d(X, y, save_path=None):
    """Plots 2D t-SNE of the data."""
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    # Downsample if too large for viz
    if len(X) > 2000:
        idx = np.random.choice(len(X), 2000, replace=False)
        X_sub = X.iloc[idx]
        y_sub = y.iloc[idx]
    else:
        X_sub, y_sub = X, y
        
    X_tsne = tsne.fit_transform(X_sub)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sub, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.title('2D t-SNE Scatter Plot')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_pca_3d(X, y, save_path=None):
    """Plots 3D PCA."""
    from mpl_toolkits.mplot3d import Axes3D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis')
    plt.colorbar(sc, label='Class')
    plt.title('3D PCA Visualization')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_training_history(history, save_dir="plots"):
    """Plots training and validation loss/accuracy."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    print(f"Saved loss plot to {save_dir}/training_loss.png")
    plt.close() # Close to prevent memory issues
    
    # Plot Accuracy (if available - Triplet loss usually just measures loss, but if we added metrics...)
    if 'accuracy' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy over Epochs')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "training_accuracy.png"))
        print(f"Saved accuracy plot to {save_dir}/training_accuracy.png")
        plt.close()
