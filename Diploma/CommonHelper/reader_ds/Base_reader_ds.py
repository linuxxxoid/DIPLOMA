import cv2
import numpy as np
import random
import os
import sys

from abc import ABC, abstractmethod
from sampler_ds.Base_sampler_ds import Base_sampler_ds

class Base_reader_ds(ABC):
    def __init__(self,
                augmentation, # аугментации для сэмплеров
                path_to_mapfile, # путь к мап файлу (текс документ с путями(или именами) к примерам датасет)
                percent_slice,  # сколько от всего количества примеров должно попасте в test_ds
                step_folder, # от куда относительно всего размера датасета начинать брать примеры для test_ds
                desired_size_ds = -1, # желаемы размер ds (диапазон от 0 до 1). Если не попадает в диапазон, то размер из количества объектов в мапфайле
                type_load_im = cv2.IMREAD_COLOR, 
                shape_to_resize = (-1, -1),
                sequence_length=10,
                num_chanels_input = 3, 
                coef_normalize = 255.0
                ):
        self._augmentation    = augmentation
        self._path_to_mapfile = path_to_mapfile
        self._percent_slice   = percent_slice
        self._step_folder     = step_folder 
        self._desired_size_ds = desired_size_ds

        self._type_load_im = type_load_im
        self._shape_to_resize = shape_to_resize 
        self._sequence_length = sequence_length
        self._num_chanels_input = num_chanels_input 
        self._coef_normalize = coef_normalize

        self._train_ds = []
        self._test_ds = []
        self._indexes_train_ds = []
        self._indexes_test_ds = []

        self.batch_start     = 0

#region public
    # размер train_ds
    @property
    def get_size_train_ds(self):
        return len(self._train_ds)

    # размер test_ds
    @property
    def get_size_test_ds(self):
        return len(self._test_ds)

    # получаем батч
    def get_next_batch(self, batch_size, is_train_data, is_rand):
        indexes = self._indexes_train_ds
        ds = self._train_ds
        if (not is_train_data):
            indexes = self._indexes_test_ds 
            ds = self._test_ds

        if (batch_size > len(indexes)):
            self._recreate_ds(indexes, ds)

        batchs = []
        for i in range(batch_size):
            choose_index = 0
            if is_rand:
                choose_index = random.randrange(0, len(indexes))
            index_item = indexes.pop(choose_index)
            batchs.append(ds[index_item].get_data_sampler())

        num_bathes = len(batchs[0])
        res_batch  = []
        for i in range(num_bathes):
            batch = []
            for b in range(batch_size):
                batch.append(batchs[b][i])
            res_batch.append(np.asarray(batch))
        
        return res_batch

#endregion


#region protected
    # считываем примеры ds из файлика и заполняем train_ds, test_ds
    def _init_reader(self):
        try:
            dict_samplers = dict()
            with open(self._path_to_mapfile, 'r') as file:
                for parse_string in file:
                    sampler  = self._create_sampler(parse_string)
                    index_class = sampler.get_index_class
                    if index_class in dict_samplers:
                        dict_samplers[index_class].append(sampler)
                    else:
                        dict_samplers[index_class] = [sampler]
        except Exception as e: 
            print(e)
            raise ValueError('error in read mapfile')

        # разделяем ds на train и test, и склеиваем
        try:
            for key in dict_samplers:
                size = len(dict_samplers[key]) if self._desired_size_ds > 1 or self._desired_size_ds <= 0  else int(len(dict_samplers[key]) * self._desired_size_ds)

                size_test_ds  = self._percent_slice * size
                start_test_ds = int(min(self._step_folder * size, size - size_test_ds))
                end_test_ds   = int(min(start_test_ds + size_test_ds, size))

                self._test_ds.extend(dict_samplers[key][start_test_ds : end_test_ds])
                self._train_ds.extend(dict_samplers[key][0 : start_test_ds])
                self._train_ds.extend(dict_samplers[key][end_test_ds : size])
        except Exception as e: 
            print(e)
            raise ValueError('error in split ds on train and test ds')

        self._indexes_train_ds = (list)(range(0, len(self._train_ds)))
        self._indexes_test_ds = (list)(range(0, len(self._test_ds)))


    # пересоздаем ds, когда больше не можем с него считать batch
    def _recreate_ds(self, indexes, ds):
        indexes.clear()
        [indexes.append(i) for i in range(len(ds))]
        self.batch_start = 0

    # по строчке из мап файла создаем объект
    @abstractmethod
    def _create_sampler(self, parse_string) -> (Base_sampler_ds):
        pass


#endregion
