from easy_captcha import ImageCaptcha
import os
import glob
import string
import random

import pdb

NUM_TRAIN = 1000
NUM_TEST = 100
CAPTCHA_LEN = 4

def main():
    #os.chdir(os.path.dirname(os.path.realpath(__file__)))
    #font_files = glob.glob('easy_fonts/*.ttf')
    generator = ImageCaptcha(fonts=['font/Cantarell-Regular.ttf'], font_sizes=[42]) #width=160, height=60, font_sizes=(42,50,56)

    for x in range(NUM_TRAIN):
        str = ''.join([random.choice(string.ascii_uppercase + string.digits) for n in range(CAPTCHA_LEN)])
        generator.write(str, f'train_imgs/{str}.png')

    for x in range(NUM_TEST):
        str = ''.join([random.choice(string.ascii_uppercase + string.digits) for n in range(CAPTCHA_LEN)])
        generator.write(str, f'test_imgs/{str}.png')

if __name__ == '__main__':
    main()
