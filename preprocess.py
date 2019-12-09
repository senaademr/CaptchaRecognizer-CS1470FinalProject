import numpy as np
import os
import glob
import cv2
import imutils
import string

import pdb

THRESHOLD = 200
OUTPUT_DIRECTORY = 'letters'

def get_data():
	os.chdir(os.path.dirname(os.path.realpath(__file__)))
	train_pngs = glob.glob('train_imgs/*.png')
	train_examples = []
	train_labels = []
	for png in train_pngs:
		img = cv2.imread(png,0)
		train_examples.append(img)
		train_labels.append(png)

	train_examples = np.array(train_examples).astype(np.float32)
	train_examples /= 255.0
	train_examples = 1-train_examples
	train_labels = [os.path.basename(label).replace('.png', '') for label in train_labels]
	alphanumeric = string.digits + string.ascii_uppercase
	train_labels = np.array([alphanumeric.find(char) for label in train_labels for char in label])
	train_labels = np.reshape(train_labels, [train_labels.shape[0]//4, 4])

	test_pngs = glob.glob('test_imgs/*.png')
	test_examples = []
	test_labels = []
	for png in test_pngs:
		img = cv2.imread(png,0)
		test_examples.append(img)
		test_labels.append(png)

	test_examples = np.array(test_examples).astype(np.float32)
	test_examples /= 255.0
	test_examples = 1-test_examples
	test_labels = np.array([os.path.basename(label).replace('.png', '') for label in test_labels])
	test_labels = np.array([alphanumeric.find(char) for label in test_labels for char in label])
	test_labels = np.reshape(test_labels, [test_labels.shape[0]//4, 4])

	return train_examples, train_labels, test_examples, test_labels
