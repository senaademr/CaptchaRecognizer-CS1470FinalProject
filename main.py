from preprocess import get_data, split_into_letters
import cv2
import numpy as np
import tensorflow as tf
from model import Model
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import pdb

NUM_EPOCHS = 4

def train(model, train_inputs, train_labels, train_losses):
>>>>>>> ed8a92c9c93728d6e8cac243ff31fcd019fe4d46
    train_inputs = tf.reshape(train_inputs, (-1, 60, 160, 1))
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
            train_losses.append(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print('{} out of {} processed for training'.format(i, num_examples))


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
	test_inputs = tf.reshape(test_inputs, (-1, 60, 160, 1))
	for i in range(0, num_examples, model.batch_size):
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
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    if(len(sys.argv) > 1 and sys.argv[1] == 'restore'):
        print('RESTORING CHECKPOINT')
        checkpoint.restore(manager.latest_checkpoint)
    for i in range(NUM_EPOCHS):
        print('**************** EPOCH {} ********************'.format(i))
        train(model, train_examples, train_labels)

    train_losses = []
    for i in range(NUM_EPOCHS):
        print('**************** EPOCH {} ********************'.format(i))
        train(model, train_examples, train_labels, train_losses)
        print('MAKING GRAPH')
        plt.plot(np.arange(len(train_losses)), np.array(train_losses))
        plt.xlabel('Batch (size 16)')
        plt.ylabel('Training Loss Per Batch')
        plt.title('Training Loss Per Batch vs. Batch Number')
        plt.savefig('training_all_size_4')
        print('Testing')
        accuracy = test(model, test_examples, test_labels)
        print('******************** TRAINING ACCURACY AFTER EPOCH {} **********************'.format(i))
        print(accuracy)
        manager.save()
    print('TRAINING COMPLETE')


if __name__ == '__main__':
	main()
