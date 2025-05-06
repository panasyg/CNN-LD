from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Reshape
from .hopf_oscillator import HopfOscillator3D

def build_hopf_autoencoder(input_shape, hopf_units=64):
    """Автоенкодер з 3D осциляторами Хопфа для сейсмічних даних"""
    inputs = Input(shape=input_shape)
    
    # Енкодер
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling3D((2, 2, 2))(x)
    
    # Осциляторний шар
    x_shape = x.shape
    x = Reshape((-1, x_shape[-1]))(x)
    x = HopfOscillator3D(hopf_units)(x)
    x = Reshape((x_shape[1], x_shape[2], x_shape[3], hopf_units))(x)
    
    # Декодер
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((2, 2, 2))(x)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)
    
    return Model(inputs, decoded, name='hopf_autoencoder')