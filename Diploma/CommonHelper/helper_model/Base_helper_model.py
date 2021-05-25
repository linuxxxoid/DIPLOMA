import cv2
import numpy as np
import random
import os
import sys
import cntk as C
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from reader_ds.Base_reader_ds import Base_reader_ds

class Base_helper_model(ABC):
    def __init__(self,
                 reader_ds : Base_reader_ds,
                 name_model,
                 path_to_save_model, #
                 num_epochs,
                 size_batch_train,
                 size_batch_test,

                 testing_in_time_train, # запускать функцию тестирования во время обучения
                 save_frequency, # раз в сколько эпох сохраняем модель
                 path_to_trained_model, # путь к уже обученной модели, если строка пустая, то создается новая модель
                 ):

        self._name_model = name_model
        self._path_to_save_model = path_to_save_model

        self._reader_ds  = reader_ds
        self._num_epochs = num_epochs
        self._size_epoch_train = self._reader_ds.get_size_train_ds
        self._size_epoch_test  = self._reader_ds.get_size_test_ds
        self._size_batch_train = size_batch_train
        self._size_batch_test = size_batch_test

        self._testing_in_time_train = testing_in_time_train
        self._save_frequency = save_frequency
        self._path_to_trained_model = path_to_trained_model

        self._progress_printer = Progress_printer(self._path_to_save_model)
        self._counter_epochs = 0
        self._counter_samplers = 0

#region public
    #
    @property
    @abstractmethod
    def get_model(self):
        pass

    #
    def train(self):
        self._counter_epochs = 0
        for epoch in range(self._num_epochs):
            self._train_process(self._counter_epochs)
            self._counter_epochs += 1
        self._progress_printer.dispose()

    #
    def test(self):
        self._process(self._size_batch_test, self._size_epoch_test, False, True, self._testing)

    def init(self):
        self._model = self._build_model()
#endregion

#region protected
    def _train_process(self, counter_epochs):
        self._process(self._size_batch_train, self._size_epoch_train, True, True, self._training)       

        if self._testing_in_time_train:
            self.test()

        self._save_model(counter_epochs)

    def _process(self, size_batch, size_epoch, is_train, is_rand, processing):
        self._counter_samplers = 0 # size_batch
        while self._counter_samplers < size_epoch:
            self._counter_samplers += size_batch
            batch = self._reader_ds.get_next_batch(size_batch, is_train, is_rand)
            processing(batch)
            self._progress_printer.update(size_batch)
        self._progress_printer.summarize()
        self._progress_printer.print_to_consol()
        self._progress_printer.print_to_file(is_train, self._counter_epochs)
        self._progress_printer.add_legend()
        self._progress_printer.switch()

    def _save_model(self, counter_epochs):
        if (counter_epochs % self._save_frequency == 0):
            self._saving_model(counter_epochs)

    def _build_model(self):
        return self._building_model() if self._path_to_trained_model  == '' else self._loading_model()


    @abstractmethod
    def _training(self, batch):
        pass

    @abstractmethod
    def _testing(self, batch):
        pass

    @abstractmethod
    def _saving_model(self, counter_epochs):
        pass

    @abstractmethod
    def _building_model(self):
        pass

    @abstractmethod
    def _loading_model(self):
        pass
#region


# для сохранения прогресса обучения и тестирования
class Progress_printer():
    def __init__(self, path_file_to_save):
        self._storage_progress_train = [dict()]
        self._storage_progress_test  = [dict()]
        self._storage_progress_current = self._storage_progress_train[-1]
        self._is_train_storage_current = True
        self._counter_batchs = 0
        self._counter_samplers = 0
        self._path_to_save = path_file_to_save
        self._f_write = open(os.path.join(path_file_to_save, 'info.txt'), 'w')
        self._legend_train_metrics = dict()
        self._legend_test_metrics = dict()


    def dispose(self):
        self._f_write.close()
        pass

    @property
    def get_current_is_train_storage(self):
        return self._is_train_storage_current

#region adders
    def __add_value_normzed(self, key, value: float, flag):
        if key in self._storage_progress_current:
            self._storage_progress_current[key][0] += value
        else:
            self._storage_progress_current[key] = [value, flag]

    # добавляем значение, которой потом нужно будет поделить на количество примеров (_counter_samplers)
    def add_value_normzed_by_counter_samplers(self, key, value: float):
        self.__add_value_normzed(key, value, 0)

    # добавляем значение, которой потом нужно будет поделить на количество батчей (_counter_batchs)
    def add_value_normzed_by_counter_batchs(self, key, value: float):
        self.__add_value_normzed(key, value, 1)

    # просто добавялем значение, котрое никак не будет нормализоваться
    def add_value(self, key, value: float):
        self._storage_progress_current[key] = value
#endregion

    def __print_to(self):
        str = ''
        i = 0
        for key in self._storage_progress_current:
            str += '  {0}: {1};  '.format(key, self._storage_progress_current[key])
            if i % 3 == 0:
                str += '\r\n'
            i += 1
        str += '\r\n'
        return str

    # только после summarize !
    def print_to_consol(self):
        print(self.__print_to())

    # только после summarize !
    def print_to_file(self, is_train, counter_epochs):
        if (is_train):
            string = '\nEpoch: ' + str(counter_epochs) + '\n'
            self._f_write.write(string)
        self._f_write.write(self.__print_to())
        self._f_write.flush()

    def switch(self):
        self._is_train_storage_current = not self._is_train_storage_current
        new_curr = self._storage_progress_train if self._is_train_storage_current else self._storage_progress_test
        if (not len(new_curr[-1]) == 0):
            new_curr.append(dict())
        self._storage_progress_current = new_curr[-1]
        self._counter_batchs = 0
        self._counter_samplers = 0
        
        
    def add_legend(self):
        if self._is_train_storage_current:
            for metric in self._storage_progress_current:
                if metric in self._legend_train_metrics:
                    self._legend_train_metrics[metric].append(self._storage_progress_current[metric])
                else:
                    self._legend_train_metrics[metric] = [self._storage_progress_current[metric]]
        else:
            for metric in self._storage_progress_current:
                if metric in self._legend_test_metrics:
                    self._legend_test_metrics[metric].append(self._storage_progress_current[metric])
                else:
                    self._legend_test_metrics[metric] = [self._storage_progress_current[metric]]
        self.make_graphics()


    def update(self, size_batch):
        self._counter_batchs += 1
        self._counter_samplers += size_batch


    def summarize(self):        
        for key in self._storage_progress_current:
            stor = self._storage_progress_current
            if (not isinstance(stor[key], list)):
                continue
            if stor[key][1] == 0: # test
                self._storage_progress_current[key] = stor[key][0] / self._counter_samplers
            else: # train
                self._storage_progress_current[key] = stor[key][0] / self._counter_batchs


    # только после summarize !
    def make_graphics(self):
        storage = self._legend_train_metrics if self._is_train_storage_current else self._legend_test_metrics
        for metric in storage:
            name_fig = metric + '.png'
            path_to_save = os.path.join(self._path_to_save, name_fig)
            plt.ioff()
            plt.plot(storage[metric], label=metric)
            plt.ioff()
            plt.xlabel('Epochs')
            plt.ylabel(metric)
            plt.title(metric.split(' ')[0])
            plt.legend()
            plt.savefig(path_to_save)
            plt.close()
