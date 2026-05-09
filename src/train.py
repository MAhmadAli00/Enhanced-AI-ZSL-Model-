import tensorflow as tf
from src.models import get_triplet_model
import os

def train_embedding_model(X_train, y_train, X_val, y_val, input_shape, embedding_dim=128, epochs=50, batch_size=32):
    """
    Trains the backbone network using Triplet Semi-Hard Loss.
    """
    print("\n--- Training Embedding Model ---")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs Detected: {gpus}")
    else:
        print("No GPU detected. Training on CPU.")
        
    model = get_triplet_model(input_shape, embedding_dim)
    
    # Compile
    # TripletSemiHardLoss expects (y_true, y_pred) where y_pred are L2-normalized embeddings
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    from src.losses import triplet_semihard_loss
    
    # Simple wrapper to match expected signature (y_true, y_pred)
    def loss_wrapper(y_true, y_pred):
        return triplet_semihard_loss(y_true, y_pred, margin=1.0)
        
    loss_fn = loss_wrapper
    
    model.compile(optimizer=optimizer, loss=loss_fn)
    
    # Callback
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    
    # Train
    # Note: X_train labels (y_train) must be integers for standard TripletSemiHardLoss usage in TF
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return model, history
