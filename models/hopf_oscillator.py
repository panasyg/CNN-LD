import tensorflow as tf
from tensorflow.keras.layers import Layer

class HopfOscillator3D(Layer):
    def init(self, units, alpha=1.0, beta=0.1, gamma=1.0, **kwargs):
        super().init(**kwargs)
        self.units = units
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            name='kernel'
        )
        self.phase = self.add_weight(
            shape=(self.units,),
            initializer='random_uniform',
            name='phase'
        )
        
    def call(self, inputs):
        # Проекція вхідних даних
        x = tf.einsum('...ij,jk->...ik', inputs, self.kernel)
        
        # Осциляторна динаміка (спрощена модель Хопфа)
        r = tf.norm(x, axis=-1, keepdims=True)
        dx = (self.alpha - self.gamma * r**2) * x - self.beta * tf.roll(x, shift=1, axis=1)
        x = x + 0.1 * dx  # Крок інтегрування
        
        # Фазова модуляція
        return x * tf.math.cos(self.phase)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma
        })
        return config