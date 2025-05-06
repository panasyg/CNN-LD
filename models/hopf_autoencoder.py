from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Reshape
from .hopf_oscillator import HopfOscillator3D

def build_hopf_autoencoder(input_shape=(66, 170, 440, 1)):
    inputs = Input(shape=input_shape)
    
    # Proper 5D handling (batch,d,h,w,c)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling3D((1, 2, 2))(x)  # Only pool spatial dims
    
    # Hopf layer
    x_shape = x.shape
    x = Reshape((-1, x_shape[-1]))(x)
    x = HopfOscillator3D(units=64)(x)
    x = Reshape((x_shape[1], x_shape[2], x_shape[3], 64))(x)
    
    # Decoder
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((1, 2, 2))(x)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)
    
    return Model(inputs, decoded)