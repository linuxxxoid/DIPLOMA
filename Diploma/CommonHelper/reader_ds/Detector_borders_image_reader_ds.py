import cv2
import numpy as np
import random
import os
import sys
import cntk as C

from reader_ds.Base_image_reader_ds import Base_image_reader_ds
from sampler_ds.DetectorBorders_image_sampler_ds import DetectorHorBorders_image_sampler_ds
from helpers.augmentations import *

class Detector_borders_image_reader_ds(Base_image_reader_ds):
    def __init__(self,
                 path_to_anchor, # путь к анкерному изображению
                 index_class_anchor, # индекс анкерного изображения
                 index_class_with_borders, # индекс классов, у которых есть границы
                 shape_label, # размер инпута лейбла с границами
                 coef_resize_bbox, # коефицент, на сколько мы должны изменить значение наших границ из-за свертки в нейронке
                 augmentation, path_to_mapfile, percent_slice, step_folder, desired_size_ds,
                 type_load_im = cv2.IMREAD_COLOR, 
                 shape_to_resize = (-1, -1),
                 num_chanels_input = 3, 
                 coef_normalize = 255.0):
        super().__init__(augmentation, path_to_mapfile, percent_slice, step_folder, desired_size_ds, type_load_im, shape_to_resize, num_chanels_input, coef_normalize)
        self._num_classes = 1 # пока нет мультиклассификации навжно чему равен этот чувак 
        self._path_to_anchor = path_to_anchor
        self._index_class_anchor = index_class_anchor
        self._index_class_with_borders = index_class_with_borders
        self._shape_label = shape_label
        self._coef_resize_bbox = coef_resize_bbox
        super()._init_reader()

        # к анкеру не нужно применять аугментацию
        base_aug = Augmentations_detect_borders_imRgb8_executor()
        self._sampler_anchor = DetectorHorBorders_image_sampler_ds('', self._shape_label, 
            self._coef_resize_bbox, self._path_to_anchor, self._index_class_anchor, self._index_class_with_borders,
            base_aug, self._type_load_im, self._shape_to_resize, self._num_chanels_input, self._coef_normalize)
        self._input_anchor = self._sampler_anchor.get_data_sampler()[0]
    
    @property
    def get_sampler_anchor(self):
        return self._sampler_anchor

    def _create_sampler(self, parse_string):
        parse_string = parse_string.split('\t')
        path_to_im   = parse_string[0]
        index_class  = int(parse_string[1])
        return DetectorHorBorders_image_sampler_ds(parse_string, self._shape_label, self._coef_resize_bbox, 
                                                   path_to_im, index_class, self._index_class_with_borders, 
                                                   self._augmentation, self._type_load_im, self._shape_to_resize, self._num_chanels_input, self._coef_normalize)

    def get_next_batch(self, size_batch, is_train_data, is_rand):
        indexs = self._indexs_train_ds
        ds = self._train_ds
        if (not is_train_data):
            indexs = self._indexs_test_ds 
            ds = self._test_ds

        if (size_batch > len(indexs)):
            self._recreate_ds(indexs, ds)

        batchs = []
        for i in range(size_batch):
            choose_index = 0;
            if is_rand:
                choose_index = random.randrange(0, len(indexs))
            index_item = indexs.pop(choose_index)
            im, borders = ds[index_item].get_ref_data_sampler()
            data_sampler = ds[index_item].get_data_sampler()
            data_sampler.append(self._input_anchor)
            data_sampler.append(im)
            data_sampler.append(borders)
            batchs.append(data_sampler)

        num_bathes = len(batchs[0])
        res_batch  = []
        for i in range(num_bathes):
            batch = []
            for b in range(size_batch):
                batch.append(batchs[b][i])
            res_batch.append(np.asarray(batch))

        return res_batch


