import cv2
import numpy as np
import random
import os
import sys
import cntk as C

import helpers.cntk_helper as cntk_helper
from reader_ds.TripletLoss_image_reader_ds import TripletLoss_image_reader_ds

'''

'''

class Triplet_class():
    def __init__(self, size_dim, shape_input, model, path_to_backbone,
                 eps, threshold):
        self._path_to_backbone = path_to_backbone
        self._model = model
        self._size_dim = size_dim
        self._eps = eps
        self._threshold = threshold
        self._shape_input = shape_input
        self._input_a = C.input_variable(shape_input, np.float32)
        self._input_p = C.input_variable(shape_input, np.float32)
        self._input_n = C.input_variable(shape_input, np.float32)

        self._head_a = self._model(self._input_a - C.Constant(114))
        self._head_p = self._model(self._input_p - C.Constant(114))
        self._head_n = self._model(self._input_n - C.Constant(114))
        self._comb_model = C.splice(self._head_a, self._head_p, self._head_n)

    def init_trainer(self, booster, lr, momentums, num_epochs, loss, metric):
        self._learner = self._booster(self._model.parameters, lr = lr, momentum = momentums)
        progress_printer =  C.logging.ProgressPrinter(tag='Training', num_epochs = num_epochs)
        self._trainer = C.Trainer(self._comb_model, (loss, metric), [self._learner], [progress_printer])
        C.logging.log_number_of_parameters(self._model) ; print()

    def training_class(self, data_input_a, data_input_p, data_input_n):
        self._trainer.train_minibatch({self._input_a: data_input_a, 
                                       self._input_p: data_input_p, self._input_n: data_input_n})

    def test_class(self, res):
        anchor   = res[0:self._size_dim]
        positive = res[self._size_dim: 2 * self._size_dim]
        negative = res[2 * self._size_dim : self._size_dim * 3]

        pos_dist = np.sum( (np.square(anchor - positive)))
        neg_dist = np.sum( (np.square(anchor - negative)))

        error = 1 if pos_dist - neg_dist + self._threshold >= 0 else 0
        diff = neg_dist - pos_dist

        return error, diff, pos_dist, neg_dist

    def loss_triplet(self):
        return self._common_loss_metric(self._eps, 1000000000)

    def metric_triplet(self):
        error = self._common_loss_metric(self._threshold, 1.0)
        error = C.ceil(error)
        return error

    def _common_loss_metric(self, e, max):
        anchor   = self._comb_model[self._size_dim * 0 : self._size_dim * 1]
        positive = self._comb_model[self._size_dim * 1 : self._size_dim * 2]
        negative = self._comb_model[self._size_dim * 2 : self._size_dim * 3]

        pos_dist = C.reduce_sum(C.square ( C.minus(anchor,positive) ))
        neg_dist = C.reduce_sum(C.square ( C.minus(anchor,negative) ))

        basic_loss = pos_dist - neg_dist + e
        loss = C.clip(basic_loss, 0.0, max)
        return loss
    #
    def build_model_base_vgg16_1(self):
        back_bone = C.load_model(self._path_to_backbone)

        freez = form_input_to_relu1_2 = cntk_helper.get_layers_by_names(back_bone, 'data', 'pool3', True)
        clone = form_input_to_relu1_2 = cntk_helper.get_layers_by_names(back_bone, 'pool3', 'pool4', False)

        res_backbone = clone(freez)

        d1 = C.layers.Dense(256, C.relu, name='d1')(res_backbone)
        drop = C.layers.Dropout(0.5, name='drop')(d1)
        d2 = C.layers.Dense(self._size_dim, None, name='d2')(drop)

        return C.layers.BatchNormalization(name="bn_out")(d2)

    #
    def build_model_base_resnet34_1(self):
        pass
