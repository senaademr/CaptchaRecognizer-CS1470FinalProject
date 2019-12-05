from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random

import pdb


class Model(tf.keras.Model):
    def __init__(self):
        """
        This is the model class.
        """
        super(Model, self).__init__()

        # Hyperparameters
        self.stddev = 0.1
        self.lstm_units = 256
        self.batch_size = 64
        self.num_classes = 37 #26 letters + 10 digits + 1 blank
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=self.stddev)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Layer numbers follow the convolution numbers so they skip some according to the paper's model
        # CNN Layers
        self.conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(
            1, 1), padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)
        self.max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(
            1, 1), padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)
        self.max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.conv_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(
            1, 1), activation='relu',  padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)

        self.conv_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(
            1, 1), padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)
        self.max_pool_4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.conv_5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(
            1, 1), padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)
        self.batch_norm_5 = tf.keras.layers.BatchNormalization()

        #self.conv_6 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(
            #1, 1), padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)
        #self.batch_norm_6 = tf.keras.layers.BatchNormalization()
        #self.max_pool_6 = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(2, 2))

        #self.conv_7 = tf.keras.layers.Conv2D(filters=512, kernel_size=2, strides=(
            #1, 1), padding='VALID', kernel_initializer=self.initializer, bias_initializer=self.initializer)

        # Map to Sequence may need to have some layers here

        # RNN Layers
        self.lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, return_state=True,
                                                    use_bias=True, kernel_initializer=self.initializer, bias_initializer=self.initializer), merge_mode='concat')
        self.lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, return_state=True,
                                                    use_bias=True, kernel_initializer=self.initializer, bias_initializer=self.initializer), merge_mode='concat')

        # Transcription - this might be wrong
        self.dense = tf.keras.layers.Dense(self.num_classes)

    def call(self, inputs):
        conv_layer_1 = self.conv_1(inputs)
        conv_layer_1 = self.max_pool_1(conv_layer_1)
        conv_layer_1 = tf.nn.relu(conv_layer_1)

        conv_layer_2 = self.conv_2(conv_layer_1)
        conv_layer_2 = self.max_pool_2(conv_layer_2)
        conv_layer_2 = tf.nn.relu(conv_layer_2)

        conv_layer_3 = self.conv_3(conv_layer_2)

        conv_layer_4 = self.conv_4(conv_layer_3)
        conv_layer_4 = self.max_pool_4(conv_layer_4)
        conv_layer_4 = tf.nn.relu(conv_layer_4)

        conv_layer_5 = self.conv_5(conv_layer_4)
        conv_layer_5 = self.batch_norm_5(conv_layer_5)
        conv_layer_5 = tf.nn.relu(conv_layer_5)

        #conv_layer_6 = self.conv_6(conv_layer_5)
        #conv_layer_6 = self.batch_norm_6(conv_layer_6)
        #conv_layer_6 = self.max_pool_6(conv_layer_6)

        #conv_layer_7 = self.conv_7(conv_layer_6)

		# Map to Sequence not sure about this
        lstm_input = tf.transpose(conv_layer_5, [0,2,1,3])
        s = lstm_input.shape
        lstm_input = tf.reshape(lstm_input, [s[0],s[1],s[2]*s[3]])

        lstm_layer_1, forward_state_1, forward_state_2, backward_state_1, backward_state_2 = self.lstm_1(lstm_input)
        lstm_layer_2, forward_state_1, forward_state_2, backward_state_1, backward_state_2 = self.lstm_2(lstm_layer_1)

        dense_layer = self.dense(lstm_layer_2)

        return dense_layer

    def loss(self, logits, labels):
        """
        Calculates the model loss after one forward pass.
        :param logits
        :param labels
        :return: the loss of the model as a Tensor
        """
		# Find a way to calculate label_length and logit_length each tensor of shape [batch_size] so below will be changed
        label_length = tf.convert_to_tensor(np.full((labels.shape[0]), labels.shape[1]))
        logit_length = tf.convert_to_tensor(np.full((logits.shape[0]), logits.shape[1]))

        #the last index (self.num_classes-1) is the 'blank' index
        loss = tf.nn.ctc_loss(labels, logits, label_length, logit_length,
                              logits_time_major=False, blank_index=self.num_classes-1)
        avg_loss = tf.reduce_mean(loss)
        print(f'TRAINING LOSS ON BATCH: {avg_loss}')
        return avg_loss

    def accuracy(self, logits, labels):
        pdb.set_trace()
        logits = tf.transpose(logits, perm=[1,0,2])
        sequence_length = tf.cast(tf.convert_to_tensor(np.full((labels.shape[0]), 4)), tf.int32)
        #docs suggest that last index (self.num_classes-1) is 'blank' index for this fct
        sparse, _ = tf.nn.ctc_greedy_decoder(logits, sequence_length)
        decoded = tf.sparse.to_dense(sparse[0], default_value=-1)
        if np.any(decoded == -1):
            print('still learning...')
        results = np.all(decoded == labels, axis=1).astype(np.float32)
        return np.mean(results)
