import cv2
import numpy as np
import random
import os
import sys
import cntk as C

from abc import ABC, abstractmethod
from helper_model.Base_helper_model import Base_helper_model
from reader_ds.Base_reader_ds import Base_reader_ds


class Base_helper_cntk_model(Base_helper_model):
    def __init__(self,
                 reader_ds : Base_reader_ds, name_model, path_to_save_model, 
                 num_epochs, size_batch_train, size_batch_test,
                 testing_in_time_train, save_frequency, path_to_trained_model,

                 path_to_save_test = '', # если строка не пустая, тогда будет использ функция по тестовых примеров (для отладки)
                 lr = 0.001, # либо константа либо лист кортежей ( [[0.01, 0.5], [0.001, 0.7]] ) 
                 momentums = 0.9,
                 booster = C.learners.adam,

                 #loss_function = None, # одна из базовых лосс функций описанная в ...
                 #metric_function = None,
                 ):
        super().__init__(reader_ds, name_model, path_to_save_model, 
            num_epochs, size_batch_train, size_batch_test,
            testing_in_time_train, save_frequency, path_to_trained_model)
        self._path_to_save_test = path_to_save_test

        self._lr = lr
        self._momentums = momentums
        self._booster = booster
        self._lr_schedule = C.learning_parameter_schedule_per_sample(lr, epoch_size=num_epochs)
        self._mm_schedule = C.momentum_schedule_per_sample(momentums) # [momentums]

        pass

    def init(self):
        self._learner = self._booster(self.get_model.parameters, lr = self._lr_schedule, momentum = self._mm_schedule )
        progress_printer =  C.logging.ProgressPrinter(tag='Training', num_epochs=self._num_epochs)
        self._trainer = C.Trainer(self.self.get_model, (self._loss_function ,
                                         self._metric_function), self._learner, progress_printer)

    def _train_process(self, counter_epochs):
        self._process(self._size_batch_train, self._size_epoch_train, True, True, self._training)       
        self._trainer.summarize_training_progress()

        if self._testing_in_time_train:
            self.test()

        self._save_model(counter_epochs)

    def test(self):
        self._process(self._size_batch_test, self._size_epoch_test, False, False, self.__test_with_save)

    def __test_with_save(self, batch):
        self._testing(batch)
        if (not self._path_to_save_test == ''):
            self._save_test_sampler(batch)

    @abstractmethod
    def _save_test_sampler(self, batch):
        pass

    @abstractmethod
    def _loss_function(self):
        pass

    @abstractmethod
    def _metric_function(self):
        pass
