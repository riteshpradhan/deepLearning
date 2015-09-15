#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ritesh
# @Date:   2015-09-14 11:18:31
# @Last Modified by:   ritesh
# @Last Modified time: 2015-09-15 10:12:42

import numpy as np
import scipy.misc
import pylab
from PIL import Image

def show_image(img):
    pylab.imshow(img)

def save_img(img):
    scipy.misc.imsave('outfile.jpg', img)

def get_img(img_name, img_size=256, batch_size=256):

    target_shape = (img_size, img_size, 3)
    img = scipy.misc.imread(img_name)  # x*x*3
    assert img.dtype == 'uint8', img_name
    # assert False

    if len(img.shape) == 2:
        img = scipy.misc.imresize(img, (img_size, img_size))
        img = np.asarray([img, img, img])
    else:
        if img.shape[2] > 3:
            img = img[:, :, :3]
        img = scipy.misc.imresize(img, target_shape)
        img = np.rollaxis(img, 2)
    if img.shape[0] != 3:
        print img_name
    return img

def convert_to_grey(img):
    imfile = '3wolfmoon_new.jpg';
    im = double(rgb2gray(imread(imfile))); # double and convert to grayscale
    im = imresize(im,[20,20]);  # change to 20 by 20 dimension
    im = im(:); # unroll matrix to vector
    im = im./max(im);

def wolfmoon_img():
    img_name = "3wolfmoon_new.jpg"
    img = get_img(img_name)
    print img, type(img), img.shape
    save_img(img)

def main():
    wolfmoon_img()
    convert_to_grey()



if __name__ == '__main__':
    main()

