import tensorflow as tf
from tensorflow.keras.layers import Layer

class HopfOscillator3D(Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)  # Explicit conversion and initialization
        
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        
        self.kernel = self.add_weight(
            name='kernel',
            shape=[last_dim, self.units],
            initializer='glorot_uniform',
            trainable=True
        )
        self.phase = self.add_weight(
            name='phase',
            shape=[self.units],
            initializer='random_uniform',
            trainable=True
        )
        
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        
        # Reshape to 2D for matmul
        x = tf.reshape(inputs, [-1, tf.shape(inputs)[-1]])
        x = tf.matmul(x, self.kernel)
        
        # Reshape back to original dimensions
        output_shape = tf.concat([input_shape[:-1], [self.units]], axis=0)
        x = tf.reshape(x, output_shape)
        
        # Hopf dynamics
        r = tf.norm(x, axis=-1, keepdims=True)
        dx = (1.0 - r**2) * x - 0.1 * tf.roll(x, shift=1, axis=-2)
        return x + 0.01 * dx
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config
