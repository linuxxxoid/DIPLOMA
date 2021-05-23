import cv2
import numpy as np
from abc import ABC, abstractmethod

from helpers.augmentations import *

# базовый класс для всех примеров из датасета
class Base_sampler_ds(ABC):
    def __init__(self, augmentation):
        self._is_load_data = False
        self._augmentation = augmentation

#region public functions
    # возвращаем индекс класса (нужен для ридера)
    @property
    @abstractmethod
    def get_index_class():
        pass

    # получаем данные для обучения (обычно переводится в нумпай маяссив с типом флот)
    def get_data_sampler(self) -> list:
        #return self.__loading_data()
        self.__loading_data()#delete
        return self._convert_data() #delete

    # получаем реф. данные
    def get_ref_data_sampler(self):
        self.__loading_data()
        return self._get_data()

    def clear_data_sampler(self):
        self.__clearing_data()

#endregion

#region protected
    # загружаем наши данные
    @abstractmethod
    def _load_data(self):
        pass

    # ковертируем данные для инпута модели
    @abstractmethod
    def _convert_data(self):
        pass

    # выгружаем референсные данные
    @abstractmethod
    def _get_data(self):
        pass

    # удаляем загруженные данные
    @abstractmethod
    def _clear_data(self):
        pass

    # аугментация
    #@abstractmethod
    #def _do_
#endregion

#region private
    def __loading_data(self):
        if (self._is_load_data == False):
            data = self._load_data()
            self._is_load_data  = True
        else: #delete
            self._get_data() #delete
        #    return data
        #return self._get_data()
        

    def __clearing_data(self):
        self._is_load_data = False
        self._clear_data()

#endregion
