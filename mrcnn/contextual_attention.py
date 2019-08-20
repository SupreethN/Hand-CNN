# Note: For this implementation we have used batchsize of 1.

import numpy as np
import keras
import keras.backend as K
import keras.layers as KL
import tensorflow as tf
from keras import initializers, constraints

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
      to make changes if needed.
  
      Batch normalization has a negative effect on training if batches are small
      so this layer is often frozen (via setting in Config class) and functions
      as linear layer.
    """
    def call(self, inputs, training=None):
        """  
          Note about training values:
              None: Train BN layers. This is the normal mode
              False: Freeze BN layers. Good when batch size is small
              True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)



def compute_distances(h, w):

    a = np.array([(i, j) for i in range(h) for j in range(w)])
    sum_squared = np.sum(np.square(a), axis=1, keepdims=True)
    dists = np.sqrt(sum_squared - 2 * np.dot(a, np.transpose(a)) + np.transpose(sum_squared))
    dists_mean = np.mean(dists)
    dists_std = np.std(dists)
    
    return dists, dists_mean, dists_std
  

class Contextual_Attention(KL.Layer):

    def __init__(self, N, dim1, dim2, channels, intermediate_dim, base_name='attn', **kwargs):

        super(Contextual_Attention, self).__init__(**kwargs)

        self.dim1 = dim1
        self.dim2 = dim2
        self.channels = channels
        self.N = N
        self.base_name = base_name
        self.intermediate_dim = intermediate_dim

        dists, mean_dists, std_dists = compute_distances(self.dim1, self.dim2)
        dists_new = np.repeat(dists[:,:,np.newaxis], self.N, axis=2)
        self.dists = K.constant(dists_new)

        self.mu_initializer = initializers.Constant(value=mean_dists)
        self.sigma_initializer = initializers.Constant(value=std_dists)
        self.alpha_initializer = initializers.Constant(value=1.0)
        self.mu_constraint = constraints.get(None)
        self.sigma_constraint = constraints.get(None)
        self.alpha_constraint = constraints.get(None) 
    
    def build(self, input_shape):
        print(input_shape)
        self.mu = self.add_weight(shape=(1,1,1,self.N), name=self.base_name+'mu', initializer=self.mu_initializer, 
            constraint=self.mu_constraint, trainable=True)
        self.sigma = self.add_weight(shape=(1,1,1,self.N), name=self.base_name+'sigma', initializer=self.sigma_initializer, 
            constraint=self.sigma_constraint, trainable=True)
        self.alpha = self.add_weight(shape=(1,1,1,self.N), name=self.base_name+'alpha', initializer=self.alpha_initializer, 
            constraint=self.alpha_constraint, trainable=True)
        
        self.conv_theta = KL.Conv2D(self.intermediate_dim, (1, 1), name=self.base_name+'conv_theta',
            padding='same', use_bias=True)
        self.conv_theta.build(input_shape)
        
        self.conv_phi = KL.Conv2D(self.intermediate_dim, (1, 1), name=self.base_name+'conv_phi',
            padding='same', use_bias=True)
        self.conv_phi.build(input_shape)
        
        self.conv_delta = KL.Conv2D(self.N, (1, 1), name=self.base_name+'conv_delta',
            padding='same', use_bias=True)
        self.conv_delta.build(input_shape)
        
        self.conv_g = KL.Conv2D(self.intermediate_dim, (1, 1), name=self.base_name+'conv_g',
            padding='same', use_bias=True)
        self.conv_g.build(input_shape)
        
        self.conv_y = KL.Conv2D(self.channels, (1, 1), name=self.base_name+'conv_y',
            padding='same', use_bias=True)
        self.conv_y.build((input_shape[0], input_shape[1], input_shape[2], self.intermediate_dim))

        self.bn_y = KL.BatchNormalization(name=self.base_name+'bn_y', gamma_initializer='zeros')
        self.bn_y.build(input_shape)
        
        self.mat_mul_1 = KL.Dot(axes=2, name=self.base_name+'mat_mul_1')
        self.mat_mul_1.build([(input_shape[0], self.dim1*self.dim2, self.intermediate_dim), (input_shape[0], self.dim1*self.dim2, self.intermediate_dim)])

        self.mat_mul_2 = KL.Dot(axes=[2, 1], name=self.base_name+'mat_mul_2')
        self.mat_mul_2.build([(input_shape[0], self.dim1*self.dim2, self.dim1*self.dim2), (input_shape[0], self.dim1*self.dim2, self.intermediate_dim)])

        
        self._trainable_weights += self.conv_theta.trainable_weights + self.conv_phi.trainable_weights + self.conv_delta.trainable_weights + self.conv_g.trainable_weights + self.conv_y.trainable_weights + self.bn_y.trainable_weights
        
        
        super(Contextual_Attention, self).build(input_shape)
        
    def call(self, ip):

        """
        Note on the 'training' argument for the BatchNorm layer:

            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences

        We will use the normal mode to enable training of BatchNorm layers
        by setting training=None in all the BatchNorm layers.
        """

        ip_shape = tf.shape(ip)
        batch_size, dim1, dim2, C  = ip_shape[0], ip_shape[1], ip_shape[2], ip_shape[3]

        self.dists = K.expand_dims(self.dists, axis=0)

        theta = self.conv_theta(ip)
        theta = K.reshape(theta, (batch_size, self.dim1*self.dim2, self.intermediate_dim))
        
        phi = self.conv_phi(ip)
        phi = K.reshape(phi, (batch_size, self.dim1*self.dim2, self.intermediate_dim))
        
        f = self.mat_mul_1([theta, phi])
        f = keras.activations.softmax(f, axis=-1)

        delta = self.conv_delta(ip)
        delta = keras.activations.softmax(delta, axis=[1,2])
        delta = K.reshape(delta, (batch_size, self.dim1*self.dim2, self.N))
        delta = K.expand_dims(delta, axis=1)
        
        P = self.dists - self.mu
        P = K.square(P)
        P = K.constant([-1]) * P
        P = K.exp(P)
        sigma_sqr = K.square(self.sigma)
        P = P * (1.0/(sigma_sqr))
        P = P * delta
        P = P * self.alpha
        P = P * K.constant([1.0/self.N])
        P = K.sum(P, axis=3, keepdims=False)
        g = self.conv_g(ip)
        g = K.reshape(g, (batch_size, self.dim1*self.dim2, self.intermediate_dim))

        f = f + P
        y = self.mat_mul_2([f, g])
        y = K.reshape(y, (batch_size, dim1, dim2, self.intermediate_dim))
        y = self.conv_y(y)
        
        # Initialize the scale parameter of the final batchnorm layer to zeros
        # so that during the start of training the model behaves like the pretrained ResNet.
        y = self.bn_y(y, training=None)
        
        y = y + ip
        
        return y
