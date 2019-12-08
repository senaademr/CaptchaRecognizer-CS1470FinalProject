from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random
import cv2

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

        # Layer numbers follow the convolution numbers so they skip some according to the paper's model
        # CNN Layers
        self.conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1),
            padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)
        self.max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(
            1, 1), padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)
        self.max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.conv_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(
            1, 1), padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)

        self.conv_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(
            1, 1), padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)
        self.max_pool_4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.conv_5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1),
            padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)

        self.batch_norm_5 = tf.keras.layers.BatchNormalization()

        self.conv_6 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1),
            padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)
        self.batch_norm_6 = tf.keras.layers.BatchNormalization()
        self.max_pool_6 = tf.keras.layers.MaxPool2D(pool_size=(1, 2))

        self.conv_7 = tf.keras.layers.Conv2D(filters=512, kernel_size=(2,2), strides=(
            2, 2), padding='VALID', kernel_initializer=self.initializer, bias_initializer=self.initializer)

        # Map to Sequence may need to have some layers here

        #RNN Layers
        self.lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True,
                    use_bias=True, kernel_initializer=self.initializer, bias_initializer=self.initializer), merge_mode='concat')
        self.lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True,
                    use_bias=True, kernel_initializer=self.initializer, bias_initializer=self.initializer), merge_mode='concat')

        # Transcription - this might be wrong
        self.dense = tf.keras.layers.Dense(self.num_classes)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    def call(self, inputs):
        """
        inputs - [batch_size, num_rows, num_cols, num_channels]
        """
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

        conv_layer_6 = self.conv_6(conv_layer_5)
        conv_layer_6 = self.batch_norm_6(conv_layer_6)
        conv_layer_6 = self.max_pool_6(conv_layer_6)

        conv_layer_7 = self.conv_7(conv_layer_6)
		# Map to Sequence not sure about this
        lstm_input = tf.transpose(conv_layer_7, [0,2,1,3])
        s = lstm_input.shape
        lstm_input = tf.reshape(lstm_input, [s[0],s[1],s[2]*s[3]])
        lstm_layer_1 = self.lstm_1(lstm_input)
        lstm_layer_2 = self.lstm_2(lstm_layer_1)

        dense_layer = self.dense(lstm_layer_2)

        return dense_layer

    def loss(self, logits, labels):
        """
        Calculates the model loss after one forward pass.
        :param logits - [batch_size, num_frames, num_classes]
        :param labels - [batch_size, max_seq_length]
        :return: the loss of the model as a Tensor
        """
		# Find a way to calculate label_length and logit_length each tensor of shape [batch_size] so below will be changed
        #label_length = tf.expand_dims(tf.convert_to_tensor(np.full((labels.shape[0]), labels.shape[1])), -1)
        #logit_length = tf.expand_dims(tf.convert_to_tensor(np.full((logits.shape[0]), logits.shape[1])), -1)
        label_length = tf.convert_to_tensor(np.full((labels.shape[0]), labels.shape[1]))
        logit_length = tf.convert_to_tensor(np.full((logits.shape[0]), logits.shape[1]))
        #the last index (self.num_classes-1) is the 'blank' index
        loss = tf.nn.ctc_loss(labels, logits, label_length, logit_length,
                              logits_time_major=False, blank_index=self.num_classes-1)
        #loss = tf.keras.backend.ctc_batch_cost(labels, logits, logit_length, label_length)
        avg_loss = tf.reduce_mean(loss)
        print(f'TRAINING LOSS ON BATCH: {avg_loss}')
        return avg_loss

    def accuracy(self, logits, labels):
        print(labels[0])
        print('argmax')
        print(np.argmax(logits, axis=2))
        logits = tf.transpose(logits, perm=[1,0,2])
        sequence_length = tf.cast(tf.convert_to_tensor(np.full((logits.shape[1]), logits.shape[0])), tf.int32)
        #docs suggest that last index (self.num_classes-1) is 'blank' index for this fct
        sparse, _ = tf.nn.ctc_greedy_decoder(logits, sequence_length)
        decoded = tf.sparse.to_dense(sparse[0], default_value=-1)
        print('decoded')
        print(decoded)
        if np.any(decoded == -1):
            print('still learning...')

        results = 0
        for i in range(decoded.shape[0]):
            sequence = decoded[i]
            predicted = sequence[sequence != -1]
            if predicted.shape[0] == labels.shape[1] and np.all(predicted == labels[i]):
                results += 1
        return results / labels.shape[0]
