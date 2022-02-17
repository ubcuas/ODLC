# https://pythonexamples.org/python-opencv-remove-green-channel-from-color-image/

import cv2
from cv2 import threshold
import numpy as np
import os

raw_folder = "../data/raw/11-16-2021/46m"
processed_folder = "../data/processed/46m"

for filename in os.listdir(raw_folder):
    #read image
    src = cv2.imread("%s/%s" % (raw_folder, filename), cv2.IMREAD_UNCHANGED)
    shape = src.shape

    # assign red, green channel to zeros
    src[:,:,1] = np.zeros([src.shape[0], src.shape[1]])
    src[:,:,2] = np.zeros([src.shape[0], src.shape[1]])

    # convert to gray scale
    # https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
    gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    inverted_gray_src = cv2.bitwise_not(gray_src)

    denoised_src = cv2.fastNlMeansDenoising(inverted_gray_src, h=10)

    (T, threshold_src) = cv2.threshold(denoised_src,240,255,cv2.THRESH_BINARY)

    # no green channel
    cv2.imwrite("%s/%s" % (processed_folder, filename),threshold_src)