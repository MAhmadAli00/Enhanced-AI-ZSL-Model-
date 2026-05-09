import time
import psutil
import os
import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_data, get_zsl_split, prepare_training_data
from src.models import HybridClusterModel
from src.train import train_embedding_model

def measure_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def run_benchmark(data_path, unseen_classes=[5]):
    print("\n--- Computational Cost Analysis ---")
    
    # 1. Load & Split
    X, y = load_data(data_path)
    X_seen, y_seen, X_unseen, y_unseen, scaler = get_zsl_split(X, y, unseen_classes)
    X_train, y_train, X_val, y_val = prepare_training_data(X_seen, y_seen)
    
    input_shape = (X_train.shape[1],)
    
    # --- Training Benchmark ---
    print("\n[Benchmarking Training]")
    start_time = time.time()
    start_mem = measure_memory()
    
    embedding_model, _ = train_embedding_model(X_train, y_train, X_val, y_val, input_shape, epochs=5, batch_size=32)
    
    end_time = time.time()
    end_mem = measure_memory()
    
    train_time_total = end_time - start_time
    train_time_per_epoch = train_time_total / 5
    mem_peak = end_mem - start_mem # Rough estimate
    
    print(f"Total Training Time (5 epochs): {train_time_total:.4f} s")
    print(f"Avg Time per Epoch: {train_time_per_epoch:.4f} s")
    print(f"Memory Usage Increase: {mem_peak:.2f} MB")
    
    # --- Inference Benchmark ---
    print("\n[Benchmarking Inference]")
    hybrid_model = HybridClusterModel(embedding_model)
    # Fit prototypes once
    hybrid_model.fit_prototypes(X_train[:100], y_train[:100]) # Small support set
    
    # Measure latency on strictly inference (embedding prediction + distance calc)
    n_samples = 1000
    if len(X_unseen) < n_samples:
        X_bench = np.tile(X_unseen, (n_samples // len(X_unseen) + 1, 1))[:n_samples]
    else:
        X_bench = X_unseen[:n_samples]
        
    start_time = time.time()
    _ = hybrid_model.predict(X_bench)
    end_time = time.time()
    
    total_inf_time = end_time - start_time
    avg_latency = total_inf_time / n_samples * 1000 # ms
    
    print(f"Total Inference Time ({n_samples} samples): {total_inf_time:.4f} s")
    print(f"Avg Latency per Sample: {avg_latency:.4f} ms")
    print(f"Throughout: {n_samples/total_inf_time:.2f} samples/sec")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='feature_vectors_syscallsbinders_frequency_5_Cat.csv')
    args = parser.parse_args()
    
    run_benchmark(args.data_path)
