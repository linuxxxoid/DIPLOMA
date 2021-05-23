import cv2
import numpy as np
import random
import os
import sys
import cntk as C

import helpers.cntk_helper as cntk_helper
from store_models_cntk.Softmax_class import Softmax_class 

class Binary_class(Softmax_class):
    def __init__(self,
                 shape_input, 
                 threshold_testing,
                 path_to_backbone = ''
                 ):
        suprt().__init__(shape_input, (1), 1, threshold_testing, path_to_backbone)

    def test_class(self, data_input, data_label):
        res = np.squeeze(self._model(data_input))
        label = np.squeeze(data_label)
        return 1 if abs(res - label) > self._threshold_testing else 0

    def loss_binary(self):
        return C.binary_cross_entropy(self._model, self._input_label)

    # я так и не понял как закрыть функцию родителя в дочернем классе, по этому она просто будет дергать нужный метод
    def loss_softmax(self):
        return self.loss_binary()

    def metric_binary(self):
        return C.not_equal(self._input_label, C.greater(self._model, self._threshold_testing))

    def metric_softmax(self):
        return self.metric_binary()

    def build_model_base_vgg16_1(self):
        back_bone = C.load_model(self._path_to_backbone)

        freez = form_input_to_relu1_2 = cntk_helper.get_layers_by_names(back_bone, 'data', 'pool3', True)
        clone = form_input_to_relu1_2 = cntk_helper.get_layers_by_names(back_bone, 'pool3', 'pool4', False)

        res_backbone = clone(freez(self._input - C.Constant(114)))

        d1 = C.layers.Dense(256, C.relu, name='d1')(res_backbone)
        drop = C.layers.Dropout(name='drop')(d1)
        d2 = C.layers.Dense(1, C.sigmoid, name='d2')(drop)

        return d2