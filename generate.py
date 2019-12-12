from easy_captcha import ImageCaptcha
import os
import glob
import string
import random
import numpy as np

import pdb

NUM_TRAIN = 30000
NUM_TEST = 5000
CAPTCHA_MIN_LEN = 4
CAPTCHA_MAX_LEN = 8

def main():
    #create a captcha generator
    generator = ImageCaptcha(fonts=['font/Cantarell-Regular.ttf'], font_sizes=[42]) #width=160, height=60

    #create the directories for the training and testing images
    if not os.path.exists('train_imgs'):
        os.makedirs('train_imgs')
    if not os.path.exists('test_imgs'):
        os.makedirs('test_imgs')

    #generate training captchas
    for x in range(NUM_TRAIN):
        str = ''.join([random.choice(string.ascii_uppercase + string.digits) \
                       for n in range(np.random.randint(CAPTCHA_MIN_LEN,CAPTCHA_MAX_LEN))])
        generator.write(str, 'train_imgs/{}.png'.format(str))

    #generate testing captchas
    for x in range(NUM_TEST):
        str = ''.join([random.choice(string.ascii_uppercase + string.digits) \
                       for n in range(np.random.randint(CAPTCHA_MIN_LEN,CAPTCHA_MAX_LEN))])
        generator.write(str, 'test_imgs/{}.png'.format(str))

if __name__ == '__main__':
    main()
