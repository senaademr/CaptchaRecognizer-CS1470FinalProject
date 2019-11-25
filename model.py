import tensorflow as tf
import numpy as np

import pdb

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = 64
        self.num_classes = 36
        self.conv1 = tf.keras.layers.Conv2D(20, (5,5), strides=(2,2), padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.conv2 = tf.keras.layers.Conv2D(50, (5,5), strides=(2,2), padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.dense1 = tf.keras.layers.Dense(500, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.num_classes)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


    def call(self, inputs):
        output = self.conv1(inputs)
        output = tf.nn.max_pool(output, (1,2,2,1), (1,2,2,1), padding='SAME')
        output = self.conv2(output)
        output = tf.nn.max_pool(output, (1,2,2,1), (1,2,2,1), padding='SAME')
        output = tf.reshape(output,[output.shape[0], 200])
        output = self.dense1(output)
        logits = self.dense2(output)
        return logits

    def loss(self, logits, labels):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), labels)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
