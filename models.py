import tflearn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
import tensorflow as tf
import os
os.environ['KERAS_BACKEND']='theano'
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model,Sequential

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers, optimizers


def lstm_keras(inp_dim, vocab_size, embed_size, num_classes, learn_rate):
#     K.clear_session()
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=inp_dim, trainable=True))
    model.add(Dropout(0.25))
    model.add(LSTM(embed_size))
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    print model.summary()
    return model


def cnn(inp_dim, vocab_size, embed_size, num_classes, learn_rate):
    tf.reset_default_graph()
    network = input_data(shape=[None, inp_dim], name='input')
    network = tflearn.embedding(network, input_dim=vocab_size, output_dim=embed_size, name="EmbeddingLayer")
    network = dropout(network, 0.25)
    branch1 = conv_1d(network, embed_size, 3, padding='valid', activation='relu', regularizer="L2", name="layer_1")
    branch2 = conv_1d(network, embed_size, 4, padding='valid', activation='relu', regularizer="L2", name="layer_2")
    branch3 = conv_1d(network, embed_size, 5, padding='valid', activation='relu', regularizer="L2", name="layer_3")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.50)
    network = fully_connected(network, num_classes, activation='softmax', name="fc")
    network = regression(network, optimizer='adam', learning_rate=learn_rate,
                         loss='categorical_crossentropy', name='target')
    
    model = tflearn.DNN(network, tensorboard_verbose=0)
    return model


def blstm(inp_dim,vocab_size, embed_size, num_classes, learn_rate):   
#     K.clear_session()
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=inp_dim, trainable=True))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(embed_size)))
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return model


class AttLayer(Layer):

    def __init__(self, **kwargs):
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1],),
                                      initializer='random_normal',
                                      trainable=True)
        super(AttLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def blstm_atten(inp_dim, vocab_size, embed_size, num_classes, learn_rate):
#     K.clear_session()
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=inp_dim))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(embed_size, return_sequences=True)))
    model.add(AttLayer())
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    adam = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.summary()
    return model

def get_model(m_type,inp_dim, vocab_size, embed_size, num_classes, learn_rate):
    if m_type == 'cnn':
        model = cnn(inp_dim, vocab_size, embed_size, num_classes, learn_rate)
    elif m_type == 'lstm':
        model = lstm_keras(inp_dim, vocab_size, embed_size, num_classes, learn_rate)
    elif m_type == "blstm":
        model = blstm(inp_dim, vocab_size, embed_size, num_classes, learn_rate)
    elif m_type == "blstm_attention":
        model = blstm_atten(inp_dim, vocab_size, embed_size, num_classes, learn_rate)
    else:
        print "ERROR: Please specify a correst model"
        return None
    return model