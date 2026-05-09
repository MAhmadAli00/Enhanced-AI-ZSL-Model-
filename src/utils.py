import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def evaluate_metrics(y_true, y_pred, label_prefix=""):
    """
    Calculates and prints classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\n[{label_prefix}] Accuracy : {acc:.4f}")
    print(f"[{label_prefix}] Precision: {prec:.4f}")
    print(f"[{label_prefix}] Recall   : {rec:.4f}")
    print(f"[{label_prefix}] F1 Score : {f1:.4f}")
    
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    """
    Plots and saves confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Confusion matrix saved to {filename}")
    plt.close()
