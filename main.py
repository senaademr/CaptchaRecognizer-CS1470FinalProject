from preprocess import get_data
import numpy as np
import tensorflow as tf
from model import Model
import os
import sys
import string
import matplotlib
#matplotlib.use('Agg') #uncomment this if on gcp
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import pdb

NUM_EPOCHS = 100
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 60

def train(model, train_inputs, train_labels, train_lengths, train_losses):
    train_inputs = tf.reshape(train_inputs, (-1, 60, 160, 1))
    num_examples = train_inputs.shape[0]
    indices = tf.random.shuffle(tf.range(num_examples))
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)
    train_lengths = tf.gather(train_lengths, indices)
    for i in range(0, num_examples, model.batch_size):
        batch_inputs = train_inputs[i:min(num_examples, i+model.batch_size)]
        batch_labels = train_labels[i:min(num_examples, i+model.batch_size)]
        batch_lengths = train_lengths[i:min(num_examples, i+model.batch_size)]
        with tf.GradientTape() as tape:
            logits = model.call(batch_inputs)
            loss = model.loss(logits, batch_labels, batch_lengths)
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
    test_inputs = tf.reshape(test_inputs, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    for i in range(0, num_examples, model.batch_size):
        batch_inputs = test_inputs[i:min(num_examples, i+model.batch_size)]
        batch_labels = test_labels[i:min(num_examples, i+model.batch_size)]
        logits = model.call(batch_inputs)
        sum += model.accuracy(logits, batch_labels)
        num_batches += 1
    return sum / num_batches

def visualize_results(image_inputs, logits, image_labels):
    sequence_length = np.full((logits.shape[0]), logits.shape[1], dtype=np.float32)
    logits = tf.transpose(logits, perm=[1,0,2])
    sparse, _ = tf.nn.ctc_beam_search_decoder(logits, sequence_length)
    decoded = tf.sparse.to_dense(sparse[0], default_value=-1)
    image_inputs = image_inputs.numpy()
    image_labels = image_labels.numpy()

    fig, axs = plt.subplots(nrows=3, ncols=3)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        for i in range(3):
            ax[i].imshow(image_inputs[ind*3+i], cmap="Greys")
            predicted_label = decoded[ind*3+i]
            predicted_label = predicted_label[predicted_label != -1]
            actual_label = image_labels[ind*3+i]
            actual_label = actual_label[actual_label != -1]
            pl = np.array(list(string.digits + string.ascii_uppercase))[predicted_label.numpy()]
            al = np.array(list(string.digits + string.ascii_uppercase))[actual_label.astype(np.int32)]
            ax[i].set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax[i].get_xticklabels(), visible=False)
            plt.setp(ax[i].get_yticklabels(), visible=False)
            ax[i].tick_params(axis='both', which='both', length=0)

    plt.show()

def main():
    print('PREPROCESSING DATA...')
    train_examples, train_labels, train_lengths, test_examples, test_labels, test_lengths = get_data()
    print('DATA PREPROCESSED...')

    print('TRAINING...')
    model = Model()
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    if(len(sys.argv) > 1 and sys.argv[1] == '--restore'):
        print('RESTORING CHECKPOINT')
        checkpoint.restore(manager.latest_checkpoint)
        if(len(sys.argv) > 2 and sys.argv[2] == '--results'):
            print('VISUALIZING RESULTS')
            indices = np.random.choice(test_examples.shape[0], 9)
            images = tf.gather(test_examples, indices)
            image_labels = tf.gather(test_labels, indices)
            reshaped = tf.reshape(images, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
            visualize_results(images, model.call(reshaped), image_labels)
            return

    train_losses = []
    for i in range(NUM_EPOCHS):
        print('**************** EPOCH {} ********************'.format(i))
        train(model, train_examples, train_labels, train_lengths, train_losses)
        print('MAKING GRAPH')
        plt.plot(np.arange(len(train_losses)), np.array(train_losses))
        plt.xlabel('Batch (size 16)')
        plt.ylabel('Training Loss Per Batch')
        plt.title('Training Loss Per Batch vs. Batch Number')
        plt.savefig('training_multiple_lengths.png')
        print('Testing')
        accuracy = test(model, test_examples, test_labels)
        print('******************** TRAINING ACCURACY AFTER EPOCH {} **********************'.format(i))
        print(accuracy)
        manager.save()
    print('TRAINING COMPLETE')


if __name__ == '__main__':
    main()
