import tensorflow as tf
from tensorflow.keras.layers import Layer

class HopfOscillator3D(Layer):
    def __init__(self, units=64, **kwargs):
        self.units = units
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.phase = self.add_weight(
            name='phase',
            shape=(self.units,),
            initializer='random_uniform',
            trainable=False
        )
        
    def call(self, inputs):
        # Flatten spatial dimensions
        x = tf.reshape(inputs, [-1, inputs.shape[-1]])
        x = tf.matmul(x, self.kernel)
        
        # Restore original dimensions
        output_shape = tf.concat([tf.shape(inputs)[:-1], [self.units]], axis=0)
        x = tf.reshape(x, output_shape)
        
        # Hopf dynamics
        r = tf.norm(x, axis=-1, keepdims=True)
        dx = (1.0 - r**2) * x - 0.1 * tf.roll(x, shift=1, axis=-2)
        return x + 0.01 * dx
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config