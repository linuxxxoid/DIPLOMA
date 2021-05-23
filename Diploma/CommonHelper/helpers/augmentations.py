import numpy as np
import os
import copy
import math
import cv2
import random

from abc import ABC, abstractmethod
from enum import Enum

class Type_augmentations_im(Enum):
    FLIP_VERT_IM = 1
    FLIP_HOR_IM = 2
    MEDIAN_FILT_IM = 3
    BRIGHTNESS_RANDOM_IM = 4
    CONTRAST_RANDOM_IM = 5


class IAugmentations_executor(ABC):
    # применяем аугментацию к сэмплу
    @abstractmethod
    def execut(self, data):
        pass

    # всегда должна быть базовая аугментация (она просто копирует данные с тем же типом)
    @abstractmethod
    def _base_augment(self):
        pass


class Base_augmentations_im_executor(IAugmentations_executor):
    def __init__(self):
        self._store_augments = []

        self._size_mask_median = 0
        self._delta_brightness = 0
        self._lower_contrast = 0
        self._upper_contrast = 0

    def _add_aug(self, a):
        if   a[0] == Type_augmentations_im.FLIP_VERT_IM:
            self._store_augments.append(self._flip_vert_im)
        elif a[0] == Type_augmentations_im.FLIP_HOR_IM:
            self._store_augments.append(self._flip_hor_im)
        elif a[0] == Type_augmentations_im.MEDIAN_FILT_IM:
            self._store_augments.append(self._median_filt_im)
            self._size_mask_median = a[1]
        elif a[0] == Type_augmentations_im.BRIGHTNESS_RANDOM_IM:
            self._store_augments.append(self._brightness_random_im)
            self._delta_brightness = a[1]
        elif a[0] == Type_augmentations_im.CONTRAST_RANDOM_IM:
            self._store_augments.append(self._contrast_random_im)
            self._lower_contrast = a[1]
            self._upper_contrast = a[2]

    def _flip_vert_im(self, data):
        return cv2.flip(data, 1)
    
    def _flip_hor_im(self, data):
        return cv2.flip(data, 0)

    def _median_filt_im(self, data):
        return cv2.medianBlur(data, self._size_mask_median)
    
    def _brightness_random_im(self, data):
        return data + random.uniform(-self._delta_brightness, self._delta_brightness)
    
    
    def _contrast_random_im(self, data):
        return data * random.uniform(self._lower_contrast, self._upper_contrast)




#параметры для аугментации передаются таплом
class Augmentations_class_im8_executor(Base_augmentations_im_executor):
    def __init__(self, *args):
        super().__init__()
        self._base_augment()
        for a in args:
            super()._add_aug(a)


    # применяем аугментацию к сэмплу
    def execut(self, data):
        choose_augment = random.randrange(1, len(self._store_augments))
        aug_data = []
        for frame in data:
            aug = self._store_augments[choose_augment](frame)
            aug_data.append(aug)
        return aug_data
        #return self._store_augments[choose_augment](data)#

    # всегда должна быть базовая аугментация (она просто копирует данные)
    def _base_augment(self):
        self._store_augments.append(self._copy_im)

    def _copy_im(self, im):
        return im.copy()

    def _brightness_random_im(self, data):
        return np.clip(super()._brightness_random_im(data), 0, 255).astype(np.dtype('uint8'))

    def _contrast_random_im(self, data):
        return np.clip(super()._contrast_random_im(data), 0, 255).astype(np.dtype('uint8'))

