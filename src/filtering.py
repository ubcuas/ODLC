# https://pythonexamples.org/python-opencv-remove-green-channel-from-color-image/

import cv2
from cv2 import threshold
import numpy as np
import os

dataset_names = ["5m", "10m", "20m", "46m"]

for dataset in dataset_names:
    raw_folder = "../data/raw/11-16-2021/%s" % dataset
    processed_folder = "../data/processed/%s" % dataset

    for filename in os.listdir(raw_folder):
        #read image
        src = cv2.imread("%s/%s" % (raw_folder, filename), cv2.IMREAD_UNCHANGED)
        src2 = src.copy()
        shape = src.shape

        # assign red, green channel to zeros
        src[:,:,1] = np.zeros([src.shape[0], src.shape[1]])
        src[:,:,2] = np.zeros([src.shape[0], src.shape[1]])

        # convert to gray scale
        # https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


        denoised_src = cv2.fastNlMeansDenoising(gray_src, h=10)

        (T, threshold_src) = cv2.threshold(denoised_src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        sectioned_image = cv2.bitwise_and(src2, src2, mask=threshold_src)

        cv2.imwrite("%s/%s" % (processed_folder, filename),sectioned_image)