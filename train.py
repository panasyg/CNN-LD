import numpy as np
import tensorflow as tf
from utils.preprocessing import load_processed_data
from models.hopf_autoencoder import build_hopf_autoencoder

def main():
    # Load and prepare data
    train_data, _ = load_processed_data('data/processed')
    
    # Add channel dimension if missing and ensure float32
    if train_data.ndim == 3:
        train_data = np.expand_dims(train_data, axis=-1)
    train_data = train_data.astype(np.float32)
    
    # Build model
    model = build_hopf_autoencoder(input_shape=train_data.shape[1:])
    model.compile(optimizer='adam', loss='mse')
    
    # Train with reduced batch size
    model.fit(
        train_data, train_data,
        batch_size=1,  # Small batch due to memory constraints
        epochs=50,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                'model.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]
    )

if __name__ == "__main__":
    main()