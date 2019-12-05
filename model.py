from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random

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
		self.num_classes = 36
		self.initializer = tf.keras.initializers.TruncatedNormal(stddev=self.stddev)
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

		#Layer numbers follow the convolution numbers so they skip some according to the paper's model
		# CNN Layers
		self.conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)
		self.max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

		self.conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)
		self.max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

		self.conv_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), activation='relu',  padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)
		
		self.conv_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)
		self.max_pool_4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

		self.conv_5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)
		self.batch_norm_5 = tf.keras.layers.BatchNormalization()

		self.conv_6 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='SAME', kernel_initializer=self.initializer, bias_initializer=self.initializer)
		self.batch_norm_6 = tf.keras.layers.BatchNormalization()
		self.max_pool_6 = tf.keras.layers.MaxPool2D(pool_size=(1,2), strides=(2,2))

		self.conv_7 = tf.keras.layers.Conv2D(filters=512, kernel_size=2, strides=(1, 1), padding='VALID', kernel_initializer=self.initializer, bias_initializer=self.initializer)

		# Map to Sequence may need to have some layers here

		#RNN Layers
		self.lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, return_state=True, use_bias=True, kernel_initializer=self.initializer, bias_initializer=self.initializer), merge_mode='concat')
		self.lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, return_state=True, use_bias=True, kernel_initializer=self.initializer, bias_initializer=self.initializer), merge_mode='concat')

		#Transcription - this might be wrong
		self.dense = tf.keras.layers.Dense(self.num_classes, activation='softmax')

	def call(self, inputs):
		"""
		Runs a forward pass on an input batch of images.
		:param inputs: images
		:return: logits
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
		lstm_input = tf.squeeze(conv_layer_7, axis=1)

		lstm_layer_1, forward_state_1, forward_state_2, backward_state_1, backward_state_2 = self.lstm_1(lstm_input)
		lstm_layer_2, forward_state_1, forward_state_2, backward_state_1, backward_state_2 = self.lstm_2(lstm_layer_1)

		#Transcription not sure about this
		dense_layer = self.dense(lstm_layer_2)

		return dense_layer

	def loss(self, logits, labels):
		"""
		Calculates the model loss after one forward pass.
		:param logits
		:param labels
		:return: the loss of the model as a Tensor
		"""
		#Find a way to calculaye label_length and logit_length each tensor of shape [batch_size] so below will be changed
		length = np.full((self.batch_size),4)
		label_length = tf.convert_to_tensor(length)
		logit_length = tf.convert_to_tensor(length)
		logits = tf.reshape(logits, (-1, self.num_classes))
		#ctc_value = tf.nn.ctc_loss(labels, logits, label_length, logit_length)
		return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=False))

	# Dont know if we really need this from hw but keep it for now
	def accuracy(self, logits, labels):
		"""
		Calculates the model's prediction accuracy by comparing
		logits to correct labels – no need to modify this.
		:param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
		containing the result of multiple convolution and feed forward layers
		:param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

		NOTE: DO NOT EDIT

		:return: the accuracy of the model as a Tensor
		"""
		logits = tf.reshape(logits, (-1, self.num_classes))
		correct_predictions = tf.equal(tf.argmax(logits, 1), labels)
		return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
