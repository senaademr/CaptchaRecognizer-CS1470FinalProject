from preprocess import get_data, split_into_letters
import cv2
import numpy as np
import tensorflow as tf
from model import Model

import pdb

def train(model, train_inputs, train_labels):
	'''
	Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
	and labels - ensure that they are shuffled in the same order using tf.gather.
	To increase accuracy, you may want to use tf.image.random_flip_left_right on your
	inputs before doing the forward pass. You should batch your inputs.
	:param model: the initialized model to use for the forward pass and backward pass
	:param train_inputs: train inputs (all inputs to use for training),
	shape (num_inputs, width, height, num_channels)
	:param train_labels: train labels (all labels to use for training),
	shape (num_labels, num_classes)
	:return: None
	'''
	num_examples = train_inputs.shape[0]

	indices = tf.random.shuffle(tf.range(num_examples))
	train_inputs = tf.gather(train_inputs, indices)
	train_labels = tf.gather(train_labels, indices)
	for i in range(0, num_examples, model.batch_size):
		batch_inputs = train_inputs[i:min(num_examples, i+model.batch_size)]
		batch_labels = train_labels[i:min(num_examples, i+model.batch_size)]

		with tf.GradientTape() as tape:
			logits = model.call(batch_inputs)
			loss = model.loss(logits, batch_labels)

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

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
		batch_inputs = test_inputs[i:min(num_examples, i+model.batch_size)]
		batch_labels = test_labels[i:min(num_examples, i+model.batch_size)]
		logits = model.call(batch_inputs, is_testing=True)
		sum += model.accuracy(logits, batch_labels)
		num_batches += 1

	return sum / num_batches

def main():
    pdb.set_trace()
    train_examples, train_labels, test_examples, test_labels = get_data()
    train_examples, train_labels = split_into_letters(train_examples, train_labels)
    train_examples = np.concatenate(train_examples, axis=0 )
    train_examples = np.reshape(train_examples, (-1, 1, 32, 32))
    train_examples = np.transpose(train_examples, [0,2,3,1])
    print('TRAINING...')
    model = Model()
    train(model, train_examples, train_labels)
    print('TRAINING COMPLETE')
    print('TESTING...')
    accuracy = test(model, test_examples, test_labels)
    print(f'TESTING ACCURACY: {accuracy}')


if __name__ == '__main__':
    main()
