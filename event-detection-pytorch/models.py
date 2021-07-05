# -*- coding: utf-8 -*-

import math
from torch.nn import functional as F
from torch import nn
import torch
import keras
from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer
from keras import backend as K
from keras.layers import CuDNNLSTM, Bidirectional, GlobalAveragePooling1D
from keras.layers import SpatialDropout1D, add
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import (Conv1D, Input, Dropout, GlobalMaxPooling1D,
                          Dense)
import warnings
import numpy as np
np.random.seed(42)  # for reproducibility
warnings.filterwarnings("ignore")


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def build_model(size_embeddings, window_length, number_words,
                number_positions, number_labels, embeddings=None):

    LSTM_UNITS = 128
    DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

    if embeddings is not None:

        size_embeddings = embeddings.shape[1]
        embedding_layer = Embedding(number_words,
                                    size_embeddings,
                                    weights=[embeddings],
                                    input_length=window_length,
                                    trainable=False,
                                    name='embedded_words')
    else:
        embedding_layer = Embedding(number_words,
                                    size_embeddings,
                                    input_length=window_length,
                                    trainable=False,
                                    name='embedded_words')
    embedding_distance_layer = Embedding(number_positions,
                                         50,
                                         input_length=window_length,
                                         trainable=True,
                                         name='embedded_distances')

    sequence_sent_input = Input(shape=(window_length,),
                                dtype='int32',
                                name='sequence_words')
    embedded_sent = embedding_layer(sequence_sent_input)
    embedded_sent = Dropout(0.3)(embedded_sent)

    sequence_dist_input = Input(shape=(window_length,),
                                dtype='int32',
                                name='sequence_distances')
    embedded_dist = embedding_distance_layer(sequence_dist_input)

    merged = concatenate([embedded_sent, embedded_dist])

    x = SpatialDropout1D(0.3)(merged)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add(
        [hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add(
        [hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    merged = Dense(number_labels, activation='softmax')(hidden)
    model = Model(inputs=[sequence_sent_input, sequence_dist_input],
                  outputs=[merged])

    model.summary()

    return model


def get_bi(size_embeddings, window_length, number_words,
           number_positions, number_labels, embeddings=None):
    if embeddings is not None:

        size_embeddings = embeddings.shape[1]
        embedding_layer = Embedding(number_words,
                                    size_embeddings,
                                    weights=[embeddings],
                                    input_length=window_length,
                                    trainable=False,
                                    name='embedded_words')
    else:
        embedding_layer = Embedding(number_words,
                                    size_embeddings,
                                    input_length=window_length,
                                    trainable=False,
                                    name='embedded_words')
    embedding_distance_layer = Embedding(number_positions,
                                         50,
                                         input_length=window_length,
                                         trainable=True,
                                         name='embedded_distances')

    sequence_sent_input = Input(shape=(window_length,),
                                dtype='int32',
                                name='sequence_words')
    embedded_sent = embedding_layer(sequence_sent_input)
    embedded_sent = Dropout(0.3)(embedded_sent)

    sequence_dist_input = Input(shape=(window_length,),
                                dtype='int32',
                                name='sequence_distances')
    embedded_dist = embedding_distance_layer(sequence_dist_input)

    merged = concatenate([embedded_sent, embedded_dist])

    x = Dropout(0.2)(merged)
    word_embeddings = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    word_embeddings = Attention(window_length)(word_embeddings)  # New
    word_embeddings = Dense(256, activation="relu")(word_embeddings)
    word_embeddings = Dropout(0.1)(word_embeddings)

    merged = Dense(number_labels, activation='softmax')(word_embeddings)
    model = Model(inputs=[sequence_sent_input, sequence_dist_input],
                  outputs=[merged])

    model.summary()

    return model


def build_baseline_model_2018(
        size_embeddings,
        window_length,
        number_words,
        number_positions,
        number_labels,
        embeddings=None):
    ngram_filters = [3, 4, 5, 6, 7]
    filters = [300] * 10
    graph_in = Input(shape=(None, size_embeddings + 50))
    convs = []
    for kernel_size, filter_length in zip(ngram_filters, filters):
        conv = Conv1D(
            filters=filter_length,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='tanh',
            kernel_initializer=keras.initializers.he_normal(
                seed=42))(graph_in)

        conv = GlobalMaxPooling1D()(conv)
        convs.append(conv)

    x = concatenate(convs)

    convolutions = Model(inputs=graph_in, outputs=x, name='convolutions')

    if embeddings is not None:

        size_embeddings = embeddings.shape[1]
        embedding_layer = Embedding(number_words,
                                    size_embeddings,
                                    weights=[embeddings],
                                    input_length=window_length,
                                    trainable=False,
                                    name='embedded_words')
    else:
        embedding_layer = Embedding(number_words,
                                    size_embeddings,
                                    input_length=window_length,
                                    trainable=True,
                                    name='embedded_words')

    embedding_distance_layer = Embedding(number_positions,
                                         50,
                                         input_length=window_length,
                                         trainable=True,
                                         name='embedded_distances')

    sequence_sent_input = Input(shape=(window_length,),
                                dtype='int32',
                                name='sequence_words')
    embedded_sent = embedding_layer(sequence_sent_input)
    embedded_sent = Dropout(0.3)(embedded_sent)

    sequence_dist_input = Input(shape=(window_length,),
                                dtype='int32',
                                name='sequence_distances')
    embedded_dist = embedding_distance_layer(sequence_dist_input)

    merged = concatenate([embedded_sent, embedded_dist])
    merged = convolutions(merged)
    merged = Dropout(0.5)(merged)

    merged = Dense(number_labels, activation='softmax')(merged)
    model = Model(inputs=[sequence_sent_input, sequence_dist_input],
                  outputs=[merged])

    model.summary()

    return model


def build_baseline_model_pytorch_2019(
        size_embeddings,
        window_length,
        number_words,
        number_positions,
        number_labels,
        embeddings=None):

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.word_embeds = nn.Embedding(number_words, size_embeddings)
            if embeddings is not None:
                self.pre_word_embeds = True
                self.word_embeds.weight = nn.Parameter(
                    torch.FloatTensor(embeddings))
            else:
                self.pre_word_embeds = False

            self.dropout = nn.Dropout(0.5)

            ngram_filters = [1, 2, 3]
            filters = [300] * 10
            convs = []
            for kernel_size, filter_length in zip(ngram_filters, filters):

                conv = nn.Conv1d(filter_length, kernel_size=kernel_size)
                convs.append(conv)

            self.conv_concat = torch.cat(convs, dim=1)
            self.max_pool = nn.AdaptiveMaxPool1d(5)

#            self.conv2_drop = nn.Dropout2d()
#            self.fc1 = nn.Linear(2304, 256)
            self.fc = nn.Linear(256, number_labels)
#

        def forward(self, sentence):
            embeds = self.word_embeds(sentence)
            x = F.relu(self.conv_concat(embeds))
            x = self.dropout(x)
            x = self.max_pool(x)
#            x = F.relu(F.max_pool2d(self.conv1(x), 2))
#            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#            x = x.view(x.size(0), -1) # Flatten layer
#            x = F.relu(self.fc1(x))
#            x = F.dropout(x, training=self.training)
#            x = self.fc2(x)
            return F.softmax(x, dim=-1)

    return Net()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def build_baseline_model_2019(
        size_embeddings,
        window_length,
        number_words,
        number_positions,
        number_labels,
        embeddings=None):
    #    ngram_filters = [3, 4, 5, 6, 7]
    ngram_filters = [1, 2, 3]
    filters = [300] * 10
    graph_in = Input(shape=(None, size_embeddings + 50))
    convs = []
    for kernel_size, filter_length in zip(ngram_filters, filters):
        conv = Conv1D(
            filters=filter_length,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu',
            kernel_initializer=keras.initializers.Orthogonal(
                gain=1.0,
                seed=42))(graph_in)

        convs.append(conv)

    x = concatenate(convs)
    x = GlobalMaxPooling1D()(x)

#    x = concatenate([
#        GlobalMaxPooling1D()(x),
#        GlobalAveragePooling1D()(x),
#    ])

    convolutions = Model(inputs=graph_in, outputs=x, name='convolutions')

    if embeddings is not None:

        size_embeddings = embeddings.shape[1]
        embedding_layer = Embedding(number_words,
                                    size_embeddings,
                                    weights=[embeddings],
                                    input_length=window_length,
                                    trainable=False,
                                    name='embedded_words')
    else:
        embedding_layer = Embedding(number_words,
                                    size_embeddings,
                                    input_length=window_length,
                                    trainable=True,
                                    name='embedded_words')

    embedding_distance_layer = Embedding(number_positions,
                                         50,
                                         input_length=window_length,
                                         trainable=True,
                                         name='embedded_distances')

    sequence_sent_input = Input(shape=(window_length,),
                                dtype='int32',
                                name='sequence_words')
    embedded_sent = embedding_layer(sequence_sent_input)
    embedded_sent = Dropout(0.3)(embedded_sent)

    sequence_dist_input = Input(shape=(window_length,),
                                dtype='int32',
                                name='sequence_distances')
    embedded_dist = embedding_distance_layer(sequence_dist_input)

    merged = concatenate([embedded_sent, embedded_dist])
    merged = convolutions(merged)
    merged = Dropout(0.5)(merged)

    merged = Dense(number_labels, activation='softmax')(merged)
    model = Model(inputs=[sequence_sent_input, sequence_dist_input],
                  outputs=[merged])

    model.summary()

    return model


def build_baseline_model_2015(
        size_embeddings,
        window_length,
        number_words,
        number_positions,
        number_labels,
        embeddings=None):
    ngram_filters = [2, 3, 4, 5]
    filters = [150] * 10
    graph_in = Input(shape=(None, size_embeddings + 50))
    convs = []
    for kernel_size, filter_length in zip(ngram_filters, filters):
        conv = Conv1D(filters=filter_length,
                      kernel_size=kernel_size,
                      strides=1,
                      padding='same')(graph_in)
        convs.append(conv)

    x = concatenate(convs)
    x = GlobalMaxPooling1D()(x)

    convolutions = Model(inputs=graph_in, outputs=x, name='convolutions')

    if embeddings is not None:

        size_embeddings = embeddings.shape[1]
        embedding_layer = Embedding(number_words,
                                    size_embeddings,
                                    weights=[embeddings],
                                    input_length=window_length,
                                    trainable=True,
                                    name='embedded_words')
    else:
        embedding_layer = Embedding(number_words,
                                    size_embeddings,
                                    input_length=window_length,
                                    trainable=True,
                                    name='embedded_words')

    embedding_distance_layer = Embedding(number_positions,
                                         50,
                                         input_length=window_length,
                                         trainable=True,
                                         name='embedded_distances')

    sequence_sent_input = Input(shape=(window_length,),
                                dtype='int32',
                                name='sequence_words')
    embedded_sent = embedding_layer(sequence_sent_input)

    sequence_dist_input = Input(shape=(window_length,),
                                dtype='int32',
                                name='sequence_distances')
    embedded_dist = embedding_distance_layer(sequence_dist_input)

    merged = concatenate([embedded_sent, embedded_dist])
    merged = convolutions(merged)
    merged = Dropout(0.5)(merged)

    merged = Dense(number_labels, activation='softmax')(merged)
    model = Model(inputs=[sequence_sent_input, sequence_dist_input],
                  outputs=[merged])

    model.summary()

    return model
