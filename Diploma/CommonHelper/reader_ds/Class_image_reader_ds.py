import cv2
import numpy as np
import random
import os
import sys

from reader_ds.Base_reader_ds import Base_reader_ds
from sampler_ds.Class_image_sampler_ds import Class_image_sampler_ds


class Class_image_reader_ds(Base_reader_ds):
    def __init__(self, 
                num_classes,
                augmentation,
                path_to_mapfile,
                percent_slice,
                step_folder, 
                desired_size_ds,
                type_load_im, 
                shape_to_resize,
                sequence_length,
                num_chanels_input, 
                coef_normalize):
        super().__init__(augmentation, path_to_mapfile, percent_slice, step_folder, desired_size_ds, type_load_im, shape_to_resize, sequence_length, num_chanels_input, coef_normalize)
        self._num_classes = num_classes
        super()._init_reader()

    def _create_sampler(self, parse_string):
        parse_string = parse_string.split('\t')
        path_to_video   = parse_string[0]
        index_class  = int(parse_string[1])
        sampler = Class_image_sampler_ds(path_to_video, index_class, self._num_classes, self._augmentation,
                                        self._type_load_im, self._shape_to_resize, self._num_chanels_input, self._coef_normalize)
        return sampler
