import cv2
import numpy as np
from sampler_ds.Base_video_sampler_ds import Base_video_sampler_ds 

# класс примера иза датасета для классификации
class Class_video_sampler_ds(Base_video_sampler_ds):
    def __init__(self,
                 path_to_video,  # путь к видео
                 index_class, # индекс класса
                 num_classes, # количество классов
                 augmentation,
                 type_load_im = cv2.IMREAD_COLOR, 
                 shape_to_resize = (-1, -1),
                 num_chanels_input = 3, 
                 coef_normalize = 127.5): # 255.0
        super().__init__(augmentation, type_load_im, shape_to_resize, num_chanels_input, coef_normalize)
        self._path_to_video = path_to_video
        self._index_class = index_class
        self._num_classes = num_classes

        self._video = None
        self._label = None
        
    @property
    def get_index_class(self):
        return self._index_class

    #
    def _load_data(self):
        self._video = super()._load_video(self._path_to_video, self._type_load_im, self._shape_to_resize, self._coef_normalize)
        self._label = self._load_label()
        #return self._video, self._label#

    #
    def _convert_data(self):
        video_aug = self._augmentation.execut(self._video)
#        return video_aug, self._label#
        return super()._convert_video(video_aug, self._coef_normalize), self._label

    #
    def _get_data(self):
        return self._video, self._label

    # Clearing the loaded data
    def _clear_data(self):
        self._video = None
        self._label = None

    # загружаем лайбел примера
    def _load_label(self):
        if (self._num_classes == 1): # бинарная классификация
            return np.asarray([self._index_class], np.dtype('f'))
        else: # мульти классификация
            label = np.zeros(self._num_classes, dtype = np.dtype('f'))
            label[self._index_class] = 1
            return label


   