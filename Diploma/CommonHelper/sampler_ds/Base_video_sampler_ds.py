import cv2
import numpy as np
from abc import ABC, abstractmethod
import random
import os

from sampler_ds.Base_sampler_ds import Base_sampler_ds

# базовый класс для примеров из датасета, где инпутом в нейронку является картинка
class Base_video_sampler_ds(Base_sampler_ds):
    def __init__(self, 
                 augmentation,
                 type_load_im = cv2.IMREAD_COLOR, # тип загружаемой картинки
                 shape_to_resize = (-1, -1), # размер (w, h) в который нужно ресайзить картинку
                 num_chanels_input = 3, # количество каналов, которая ожидает модель на вход
                 coef_normalize = 255.0): # коефицент нормализации яркости пикселей 
        super().__init__(augmentation)
        self._type_load_im = type_load_im
        self._shape_to_resize = shape_to_resize 
        self._num_chanels_input = num_chanels_input 
        self._coef_normalize = coef_normalize


#region protected
    # загружаем изображений @staticmethod 
    @classmethod
    def _load_video(cls, path_to_video, type_load_im, shape_to_resize, coef_normalize):
        try:
            files = os.listdir(path_to_video) # list of files and directories in a folder
            video_frames = []
            for file in files:
                file_path = os.path.join(path_to_video, file)
                if os.path.isfile(file_path):
                    im = cv2.imread(file_path, type_load_im)
                    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(rgb, shape_to_resize, interpolation=cv2.INTER_AREA)
                    im = np.array(im, dtype=np.float32)

                    video_frames.append(im)
            return video_frames
        except Exception as e: 
            print(e)
            raise ValueError('error load video')

    # конвертируем изображений TODO: удалить
    @classmethod
    def _convert_video(cls, video, coef_normalize):
        video_frames = []
        for im in video:
            # нормализуем фрейм
            norm_im = np.array(im, dtype=np.float32)
            norm_im -= coef_normalize
            norm_im /= coef_normalize

            # (channel, height, width)
            im = np.ascontiguousarray(np.transpose(norm_im, (2, 0, 1)))
            video_frames.append(im)
        return np.stack(video_frames, axis=1)
#endregion
