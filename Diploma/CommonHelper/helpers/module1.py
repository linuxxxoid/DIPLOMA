import numpy as np
import os
import copy
import math
import cv2
import random

from abc import ABC, abstractmethod
from enum import Enum


        #augmentation = Augmentations_class_im8_executor(
        #[Type_augmentations_im.FLIP_HOR_IM],
        #[Type_augmentations_im.MEDIAN_FILT_IM, 3],
        #[Type_augmentations_im.BRIGHTNESS_RANDOM_IM, 20],
        #[Type_augmentations_im.CONTRAST_RANDOM_IM, 0.7, 1.3]
        #)
if __name__ == '__main__':
    img = cv2.imread(r'D:\mine\diploma\Dataset\Data\Skeleton\Other\1\3.jpg')

    flip_vert = cv2.flip(img, 1)
    cv2.imshow('flip_vert', flip_vert)
    cv2.waitKey(0)
    
    flip_hor = cv2.flip(img, 0)
    cv2.imshow('flip_hor', flip_hor)
    cv2.waitKey(0)
   
    med = cv2.medianBlur(img, 3)
    cv2.imshow('med', med)
    cv2.waitKey(0)
    
    br = img + random.uniform(-1, 1)
    cv2.imshow('br', br)
    cv2.waitKey(0)

    contr = img * random.uniform(0.1, 0.3)
    cv2.imshow('contr', contr)
    cv2.waitKey(0)
