#import cv2
#import numpy as np
#import random
#import os
#import sys
#import cntk as C

#import helpers.cntk_helper as cntk_helper

#'''
#название мб не оч... Тут описаны функции, которые хендлеры модели обязаны переопределить, если за основу используют этот подход
#и написаны самые часто используемые фичи, которые используется при обучение моделей этого типа
#'''

#class Softmax_class():
#    def __init__(self,
#                 shape_input, #
#                 shape_input_label, #
#                 num_class, #
#                 threshold_testing,
#                 path_to_backbone = '' #
#                 ):
#        self._shape_input = shape_input
#        self._shape_input_label = shape_input_label
#        self._num_class = num_class
#        self._threshold_testing = threshold_testing
#        self._path_to_backbone = path_to_backbone

#        self._input = C.input_variable(self._shape_input, np.float32)
#        self._input_label = C.input_variable(self._shape_input_label, np.float32)
#        self._model = None


#    def init_trainer(self, booster, lr, momentums, num_epochs, loss, metric):
#        self._learner = booster(self._model.parameters, lr = lr, momentum = momentums)
#        progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=num_epochs)
#        self._trainer = C.Trainer(self._model, (loss, metric), [self._learner], [progress_printer])
#        C.logging.log_number_of_parameters(self._model) ; print()


#    def training_class(self, data_input, data_label):
#        self._trainer.train_minibatch({self._input: data_input, self._input_label: data_label})

#    def test_class(self, data_input, data_label):
#        res = np.squeeze(C.softmax(self._model(data_input)).eval())
#        label = np.squeeze(data_label)
#        index_class = np.argmax(label)
#        return 1 if res[index_class] < self._threshold_testing else 0

#    def loss_softmax(self):
#        return C.cross_entropy_with_softmax(self._model , self._input_label)

#    def metric_softmax(self):
#        return C.classification_error(self._model , self._input_label)

#    #
#    def build_model_base_vgg16_1(self):
#        back_bone = C.load_model(self._path_to_backbone)

#        freez = form_input_to_relu1_2 = cntk_helper.get_layers_by_names(back_bone, 'data', 'pool3', True)
#        clone = form_input_to_relu1_2 = cntk_helper.get_layers_by_names(back_bone, 'pool3', 'pool4', False)

#        res_backbone = clone(freez(self._input - C.Constant(114)))

#        d1 = C.layers.Dense(256, C.relu, name='d1')(res_backbone)
#        drop = C.layers.Dropout(name='drop')(d1)
#        d2 = C.layers.Dense(self._num_class , None, name='d2')(drop)

#        return d2

#    #
#    def build_model_base_resnet34_1(self):
#        pass

import cv2
import numpy as np
import random
import os
import sys
import cntk as C

import helpers.cntk_helper as cntk_helper

'''
название мб не оч... Тут описаны функции, которые хендлеры модели обязаны переопределить, если за основу используют этот подход
и написаны самые часто используемые фичи, которые используется при обучение моделей этого типа
'''

class Softmax_class():
    def __init__(self,
                 shape_input, #
                 shape_input_label, #
                 num_class, #
                 threshold_testing,
                 path_to_backbone = '' #
                 ):
        self._shape_input = shape_input
        self._shape_input_label = shape_input_label
        self._num_class = num_class
        self._threshold_testing = threshold_testing
        self._path_to_backbone = path_to_backbone

        self._input = C.input_variable(self._shape_input, np.float32)
        self._input_label = C.input_variable(self._shape_input_label, np.float32)
        self._model = None


    def init_trainer(self, booster, lr, momentums, num_epochs, loss, metric):
        self._learner = booster(self._model.parameters, lr = lr, momentum = momentums)
        progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=num_epochs)
        self._trainer = C.Trainer(self._model, (loss, metric), [self._learner], [progress_printer])
        C.logging.log_number_of_parameters(self._model) ; print()


    def training_class(self, data_input, data_label):
        self._trainer.train_minibatch({self._input: data_input, self._input_label: data_label})

    def test_class(self, data_input, data_label):
        res = np.squeeze(C.softmax(self._model(data_input)).eval())
        label = np.squeeze(data_label)
        index_class = np.argmax(label)
        return 1 if res[index_class] < self._threshold_testing else 0

    def loss_softmax(self):
        return C.cross_entropy_with_softmax(self._model , self._input_label)

    def metric_softmax(self):
        return C.classification_error(self._model , self._input_label)

    #
    def build_model_base_vgg16_1(self):
        back_bone = C.load_model(self._path_to_backbone)

        freez = form_input_to_relu1_2 = cntk_helper.get_layers_by_names(back_bone, 'data', 'pool3', True)
        clone = form_input_to_relu1_2 = cntk_helper.get_layers_by_names(back_bone, 'pool3', 'pool4', False)

        res_backbone = clone(freez(self._input - C.Constant(114)))

        d1 = C.layers.Dense(256, C.relu, name='d1')(res_backbone)
        drop = C.layers.Dropout(name='drop')(d1)
        d2 = C.layers.Dense(self._num_class , None, name='d2')(drop)

        return d2

    #
    def build_model_base_resnet34_1(self):
        pass