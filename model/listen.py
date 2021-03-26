#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

class pyramidial_BLSTM(tf.keras.layers.Layer):
    '''
    Implementation of Pyramidal Bidirectional Long Short-Term Memory cell 
    using Keras from Tensorflow: 
    
    - https://arxiv.org/pdf/1508.01211.pdf
    - https://www.youtube.com/watch?v=WfA6OGtM4UI  
    
    '''
    def __init__(self, dimension):
        '''
        Creation of the BLSTM cell:
        '''
        super().__init__()
        self.dimension = dimension
        self.LSTM = tf.keras.layers.LSTM(self.dimension, return_sequences=True)
        self.BLSTM = tf.keras.layers.Bidirectional(self.LSTM)

    def call(self, x):
        y = self.BLSTM(x)
        # If dimension of input is odd, add null adapted vector at the end of the input
        if tf.shape(x)[1] % 2 == 1:
          y = tf.keras.layers.ZeroPadding1D(padding=(0, 1))(y)
        #Concatenation
        y = tf.keras.layers.Reshape(target_shape=(-1,4*self.dimension))(y)
        return y

    
def Listen(X):
    dimension = len(X)
    inputs = tf.keras.Input(shape=(None, dimension))
    X = pyramidial_BLSTM(dimension//4)(inputs)
    X = pyramidial_BLSTM(dimension//4)(X)
    h = pyramidial_BLSTM(dimension//4)(X)
    
    model = tf.keras.Model(inputs, h, name="Listen Encoder")
    model.summary()