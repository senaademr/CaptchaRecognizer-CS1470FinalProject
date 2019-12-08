from preprocess import get_data, split_into_letters
import cv2
import numpy as np
import tensorflow as tf
from model import Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import pdb

NUM_EPOCHS = 5

def train(model, train_inputs, train_labels):
	train_inputs = tf.reshape(train_inputs, (-1, 60, 160, 1))
	#train_labels = tf.reshape(train_labels, (1,4))
	num_examples = train_inputs.shape[0]
	indices = tf.random.shuffle(tf.range(num_examples))
	train_inputs = tf.gather(train_inputs, indices)
	train_labels = tf.gather(train_labels, indices)
	for i in range(0, num_examples, model.batch_size):
	#for i in range(0, model.batch_size*2, model.batch_size):
		batch_inputs = train_inputs[i:min(num_examples, i+model.batch_size)]
		batch_labels = train_labels[i:min(num_examples, i+model.batch_size)]
		with tf.GradientTape() as tape:
			logits = model.call(batch_inputs)
			loss = model.loss(logits, batch_labels)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		print(f'{i} out of {num_examples} processed for training')


def test(model, test_inputs, test_labels):
	"""
	Tests the model on the test inputs and labels. You should NOT randomly
	flip images or do any extra preprocessing.
	:param test_inputs: test data (all images to be tested),
	shape (num_inputs, width, height, num_channels)
	:param test_labels: test labels (all corresponding labels),
	shape (num_labels, num_classes)
	:return: test accuracy - this can be the average accuracy across
	all batches or the sum as long as you eventually divide it by batch_size
	"""
	sum = 0
	num_batches = 0
	num_examples = test_inputs.shape[0]
	for i in range(0, num_examples, model.batch_size):
	#for i in range(0, 64, model.batch_size):
		batch_inputs = test_inputs[i:min(num_examples, i+model.batch_size)]
		batch_labels = test_labels[i:min(num_examples, i+model.batch_size)]
		logits = model.call(batch_inputs)
		sum += model.accuracy(logits, batch_labels)
		num_batches += 1
	return sum / num_batches

def main():
	print('PREPROCESSING DATA...')
	train_examples, train_labels, test_examples, test_labels = get_data()
	print('DATA PREPROCESSED...')

	print('TRAINING...')
	model = Model()
	for i in range(NUM_EPOCHS):
		print(f'**************** EPOCH {i} ********************')
		train(model, train_examples, train_labels)
		print('Testing')
		accuracy = test(model, test_examples, test_labels)
		print(f'******************** TRAINING ACCURACY AFTER EPOCH {i} **********************')
		print(accuracy)
	print('TRAINING COMPLETE')


if __name__ == '__main__':
	main()
