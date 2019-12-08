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
		self.initializer = tf.keras.initializers.TruncatedNormal(stddev=self.stddev)
		self.lstm_units = 256
		self.batch_size = 16
		self.num_classes = 37 #26 letters + 10 digits + 1 blank
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)

		# Layer numbers follow the convolution numbers so they skip some according to the paper's model
		# CNN Layers
		self.sequence = tf.keras.Sequential()
		self.sequence.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=(60,160,1,), strides=(1, 1), padding='SAME', activation='relu', kernel_initializer=self.initializer, use_bias=False))
		self.sequence.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

		self.sequence.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='SAME', activation='relu', kernel_initializer=self.initializer, use_bias=False))
		self.sequence.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

		self.sequence.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='SAME', activation='relu', kernel_initializer=self.initializer, use_bias=False))

		self.sequence.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='SAME', activation='relu', kernel_initializer=self.initializer, use_bias=False))
		self.sequence.add(tf.keras.layers.MaxPool2D(pool_size=(2, 1)))

		self.sequence.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='SAME', activation='relu', kernel_initializer=self.initializer, use_bias=False))
		self.sequence.add(tf.keras.layers.BatchNormalization())

		self.sequence.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='SAME', activation='relu', kernel_initializer=self.initializer, use_bias=False))
		self.sequence.add(tf.keras.layers.BatchNormalization())
		self.sequence.add(tf.keras.layers.MaxPool2D(pool_size=(2, 1)))

		self.sequence.add(tf.keras.layers.Conv2D(filters=512, kernel_size=2, strides=(2, 2), padding='VALID', activation='relu', kernel_initializer=self.initializer, use_bias=False))
		self.sequence.add(tf.keras.layers.BatchNormalization())
		# Map to Sequence may need to have some layers here
		self.sequence.add(tf.keras.layers.Reshape((20,512)))

		# RNN Layers
		self.sequence.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True), merge_mode='concat'))
		self.sequence.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True), merge_mode='concat'))

		# Transcription - this might be wrong
		self.sequence.add(tf.keras.layers.Dense(self.num_classes))

	def call(self, inputs):
		return self.sequence(inputs)

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
		logits = tf.transpose(logits, perm=[1,0,2])
		#print(tf.math.argmax(logits, 2))
		#sequence_length = tf.cast(tf.convert_to_tensor(np.full((labels.shape[0]), 4)), tf.int32)
		sequence_length = np.full((labels.shape[0]), 20, dtype=np.float32)
		#docs suggest that last index (self.num_classes-1) is 'blank' index for this fct
		sparse, _ = tf.nn.ctc_beam_search_decoder(logits, sequence_length)
		#print(sparse[0].values)
		decoded = tf.sparse.to_dense(sparse[0], default_value=-1)
		print('Decoded')
		print(decoded)
		#print('Labels')
		#print(labels)
		if np.any(decoded == -1):
			print('still learning...')
		#results = np.all(decoded == labels, axis=1).astype(np.float32)
		results = 0
		for i in range(labels.shape[0]):
			if labels.shape[1] == decoded.shape[1]:
				#if np.any(decoded[i] == labels[i]):
				#	print('made some match')
				if np.all(decoded[i] == labels[i]):
					results+=1
		return results/labels.shape[0]
