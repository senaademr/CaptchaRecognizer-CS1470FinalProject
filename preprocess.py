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
		img = cv2.imread(png)
		train_examples.append(img)
		train_labels.append(png)

	train_labels = [os.path.basename(label).replace('.png', '') for label in train_labels]

	test_pngs = glob.glob('test_imgs/*.png')
	test_examples = []
	test_labels = []
	for png in test_pngs:
		img = cv2.imread(png)
		test_examples.append(img)
		test_labels.append(png)

	test_labels = [os.path.basename(label).replace('.png', '') for label in test_labels]

	train_examples, train_labels = split_into_letters(train_examples, train_labels)
	train_examples = np.concatenate(train_examples, axis=0 )
	train_examples = np.reshape(train_examples, (-1, 1, 32, 32))
	train_examples = np.transpose(train_examples, [0,2,3,1]).astype(np.float32)

	test_examples, test_labels = split_into_letters(test_examples, test_labels)
	test_examples = np.concatenate(test_examples, axis=0 )
	test_examples = np.reshape(test_examples, (-1, 1, 32, 32))
	test_examples = np.transpose(test_examples, [0,2,3,1]).astype(np.float32)

	return train_examples, train_labels, test_examples, test_labels

def resize_to_fit(image, width, height):
	"""
	A helper function to resize an image to fit within a given size
	:param image: image to resize
	:param width: desired width in pixels
	:param height: desired height in pixels
	:return: the resized image
	"""

	# grab the dimensions of the image, then initialize
	# the padding values
	(h, w) = image.shape[:2]

	# if the width is greater than the height then resize along
	# the width
	if w > h:
		image = imutils.resize(image, width=width)

	# otherwise, the height is greater than the width so resize
	# along the height
	else:
		image = imutils.resize(image, height=height)

	# determine the padding values for the width and height to
	# obtain the target dimensions
	padW = int((width - image.shape[1]) / 2.0)
	padH = int((height - image.shape[0]) / 2.0)

	# pad the image then apply one more resizing to handle any
	# rounding issues
	image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
		cv2.BORDER_REPLICATE)
	image = cv2.resize(image, (width, height))

	# return the pre-processed image
	return image

def split_into_letters(examples, labels):
	alphanumeric = string.digits + string.ascii_uppercase
	letters = []
	letter_labels = []
	for (example, label) in zip(examples, labels):
		gray = cv2.cvtColor(example, cv2.COLOR_BGR2GRAY)
		gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		use_img = True
		bounding_boxes = []
		for contour in contours:
			(x, y, w, h) = cv2.boundingRect(contour)
			bounding_boxes.append((x, y, w, h))
		if len(bounding_boxes) != 4:
			use_img = False

		bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])

		if use_img:
			for (bounding_box, letter) in zip(bounding_boxes, label):
				x, y, w, h = bounding_box
				letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
				letter_image = resize_to_fit(letter_image,32,32)
				letters.append(letter_image)
				letter_labels.append(alphanumeric.find(letter))
				letter_image = resize_to_fit(letter_image,32,32)

	return letters, letter_labels

def main():
	train_examples, train_labels, test_examples, test_labels = get_data()
	letters, labels = split_into_letters(train_examples, train_labels)


if __name__ == '__main__':
	main()
