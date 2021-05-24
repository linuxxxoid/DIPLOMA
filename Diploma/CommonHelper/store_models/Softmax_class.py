import cv2
import numpy as np
import random
import os
import sys
import cntk as C

import helpers.cntk_helper as cntk_helper


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
        lr_schedule            = C.learning_parameter_schedule_per_sample(lr, epoch_size=num_epochs)
        mm_schedule            = C.momentum_schedule_per_sample([momentums])
        self._learner = booster(self._model.parameters, lr_schedule, mm_schedule, True)
        #self._learner = booster(self._model.parameters, lr=lr_schedule, momentum=mm_schedule,
        #                              gradient_clipping_threshold_per_sample=8,
        #                              gradient_clipping_with_truncation=True)
        progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=num_epochs)
        self._trainer = C.Trainer(self._model, (loss, metric), self._learner, [progress_printer])
        C.logging.log_number_of_parameters(self._model) ; print()


    def training_class(self, data_input, data_label):
        self._trainer.train_minibatch({self._model.arguments[0]: data_input, self._input_label: data_label})
        
        loss = self._trainer.previous_minibatch_loss_average
        eval_error = self._trainer.previous_minibatch_evaluation_average

        res = np.squeeze(C.softmax(self._model(data_input)).eval())
        label_indexes = np.argmax(data_label, 1)
        predict_label_indexes = np.argmax(res, 1)
        same = label_indexes == predict_label_indexes
        minibatch_size = len(data_input)
        accuracy = np.sum(same) / minibatch_size

        dict_metrics = dict()
        dict_metrics['train loss'] = loss
        dict_metrics['train average eval erorr (metric function)'] = eval_error
        dict_metrics['train accuracy'] = accuracy
        return dict_metrics


    def test_class(self, data_input, data_label):
        res = np.squeeze(C.softmax(self._model(data_input)).eval())
        label = np.squeeze(data_label)
        label_index = np.argmax(label)
        error = 1 if res[label_index] < self._threshold_testing else 0

        predict_label_index = np.argmax(res)
        same = label_index == predict_label_index
        minibatch_size = len(data_input) # для тестинга всегда 1
        accuracy = np.sum(same) / minibatch_size

        dict_metrics = dict()
        dict_metrics['test error'] = error # 1 - ошибка есть, 0 - ошибки нет
        dict_metrics['test accuracy'] = accuracy
        return dict_metrics


    def loss_softmax(self):
        return C.cross_entropy_with_softmax(self._model , self._input_label)


    def metric_softmax(self):
        #(C.classification_error(self._model, self._input_label)
        return C.element_not(C.classification_error(self._model , self._input_label))

    
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