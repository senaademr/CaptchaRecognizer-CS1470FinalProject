import numpy as np
import os
import glob
import cv2
import imutils
import string

def get_data():
	#get the training data
	os.chdir(os.path.dirname(os.path.realpath(__file__)))
	train_pngs = glob.glob('train_imgs/*.png')
	train_examples = []
	train_labels = []
	for png in train_pngs:
		img = cv2.imread(png,0)
		train_examples.append(img)
		train_labels.append(png)

	#process the images and get the strings for the training labels
	train_examples = np.array(train_examples).astype(np.float32)
	train_examples /= 255.0
	train_examples = 1-train_examples
	train_labels = [os.path.basename(label).replace('.png', '') for label in train_labels]
	train_lengths = np.array([len(str) for str in train_labels])
	max_len = max(train_lengths)

	#pad the training labels
	alphanumeric = string.digits + string.ascii_uppercase
	train_labels = [alphanumeric.find(char) for label in train_labels for char in label]
	to_fill = np.zeros([train_examples.shape[0], max_len], dtype=int)
	i = 0
	row = 0
	while i < len(train_labels):
		lst = train_labels[i:i+train_lengths[row]]
		lst.extend((max_len - train_lengths[row])*[-1])
		to_fill[row] = np.array(lst)
		i += train_lengths[row]
		row += 1

	train_labels = to_fill

	#get the testing data
	test_pngs = glob.glob('test_imgs/*.png')
	test_examples = []
	test_labels = []
	for png in test_pngs:
		img = cv2.imread(png,0)
		test_examples.append(img)
		test_labels.append(png)

	#process the images and get the strings for the testing labels
	test_examples = np.array(test_examples).astype(np.float32)
	test_examples /= 255.0
	test_examples = 1-test_examples
	test_labels = [os.path.basename(label).replace('.png', '') for label in test_labels]
	test_lengths = np.array([len(str) for str in test_labels])
	max_len = max(test_lengths)

	#pad the testing labels
	test_labels = [alphanumeric.find(char) for label in test_labels for char in label]
	to_fill = np.zeros([test_examples.shape[0], max_len], dtype=int)
	i = 0
	row = 0
	while i < len(test_labels):
		lst = test_labels[i:i+test_lengths[row]]
		lst.extend((max_len - test_lengths[row])*[-1])
		to_fill[row] = np.array(lst)
		i += test_lengths[row]
		row += 1

	test_labels = to_fill

	return train_examples, train_labels, train_lengths, test_examples, test_labels, test_lengths
