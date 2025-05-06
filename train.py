import numpy as np
import tensorflow as tf
from utils.preprocessing import load_processed_data
from models.hopf_autoencoder import build_hopf_autoencoder

def main():
    # Load processed data (already in 4D: d,h,w,c)
    train_data, _ = load_processed_data('data/processed')
    
    # Build model
    model = build_hopf_autoencoder(input_shape=train_data.shape[1:])
    model.compile(optimizer='adam', loss='mse')
    
    # Train with batch_size=1
    model.fit(
        train_data, train_data,
        batch_size=1,
        epochs=50,
        validation_split=0.0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)
        ]
    )

if __name__ == "__main__":
    main()