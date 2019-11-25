import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = 64
        self.num_classes = 36
        self.conv1 = tf.keras.layers.Conv2D(10, (3,3), strides=(2,2), padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.conv2 = tf.keras.layers.Conv2D(10, (3,3), strides=(2,2), padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.conv3 = tf.keras.layers.Conv2D(10, (3,3), strides=(2,2), padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.dense1 = tf.keras.layers.Dense(1000, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        output = conv1(inputs)

        output = conv2(output)
        output = conv3(output)
        return logits

    def loss(self, logits, labels):
        return tf.nn.softmax_cross_entropy_with_logits(labels, logits) # need sparse cross entropy

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
