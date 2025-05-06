from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Reshape
from .hopf_oscillator import HopfOscillator3D
import tensorflow as tf

def build_hopf_autoencoder(input_shape=(66, 170, 440, 1)):
    """Builds a 3D autoencoder that properly handles your data dimensions"""
    inputs = Input(shape=input_shape)
    
    # Reshape to 5D: (batch, depth, height, width, channels)
    x = inputs
    
    # Encoder
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((1, 2, 2))(x)  # Only pool spatial dimensions
    
    # Hopf Layer
    x_shape = tf.shape(x)
    x = Reshape((-1, x_shape[-1]))(x)
    x = HopfOscillator3D(units=64)(x)
    x = Reshape((x_shape[1], x_shape[2], x_shape[3], 64))(x)
    
    # Decoder
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((1, 2, 2))(x)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)
    
    return Model(inputs, decoded)