import matplotlib.pyplot as plt
import numpy as np

def plot_slices(volume, num_slices=3):
    """Візуалізація зрізів 3D об'єму"""
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))
    for i in range(num_slices):
        idx = i * (volume.shape[0] // num_slices)
        axes[i].imshow(volume[idx], cmap='seismic')
        axes[i].set_title(f"Зріз {idx}")
    plt.tight_layout()
    plt.show()

def plot_learning_curve(history):
    """Візуалізація кривих навчання"""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
