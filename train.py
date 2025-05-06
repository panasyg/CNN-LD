import tensorflow as tf
from utils.preprocessing import load_processed_data
from models.hopf_autoencoder import build_hopf_autoencoder
import yaml
import os

def load_config():
    with open('configs/params.yaml') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Завантаження оброблених даних
    train_data, train_labels = load_processed_data(os.path.join('data', 'processed', 'train'))
    
    # Побудова моделі
    model = build_hopf_autoencoder(
        input_shape=config['model']['input_shape'],
        hopf_units=config['model']['hopf_units']
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config['model']['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]
    
    # Навчання
    history = model.fit(
        train_data, train_data,  # Автоенкодер
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs'],
        validation_split=0.2,
        callbacks=callbacks
    )

if name == "main":
    main()
