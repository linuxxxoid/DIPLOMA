
import cv2
import numpy as np
import random
import os
import sys
import cntk as C


import helpers.cntk_helper as cntk_helper
from helpers.augmentations import *
from reader_ds.Class_video_reader_ds import Class_video_reader_ds
from helper_model.Base_helper_cntk_model import Base_helper_cntk_model
from store_models.Softmax_class import Softmax_class



class Saberfighting_model_softmax_mode(Base_helper_cntk_model, Softmax_class):
    def __init__(self,
                 threshold_testing,
                 shape_input, shape_input_label, num_class, path_to_backbone,
                 reader_ds, name_model, path_to_save_model, 
                 num_epochs, size_batch_train, size_batch_test, 
                 testing_in_time_train, save_frequency, path_to_trained_model, 
                 path_to_save_test='', lr=0.001, momentums=0.9, booster=C.learners.adam
                 ):
        Base_helper_cntk_model.__init__(self, reader_ds, name_model, path_to_save_model, 
                num_epochs, size_batch_train, size_batch_test, 
                testing_in_time_train, save_frequency, path_to_trained_model, 
                path_to_save_test, lr, momentums, booster)
        Softmax_class.__init__(self, shape_input, shape_input_label, num_class, threshold_testing, path_to_backbone)
        self._model = self._building_model()
        


    @property
    def get_model(self):
        self._model

    def init(self):
        Softmax_class.init_trainer(self, self._booster, self._lr, self._momentums, self._num_epochs, self._loss_function(), self._metric_function())

    def _training(self, batch):
        dict_metrics = Softmax_class.training_class(self, batch[0], batch[1])
        for metric in dict_metrics:
            self._progress_printer.add_value_normzed_by_counter_batchs(metric, dict_metrics[metric])

    def _testing(self, batch):
        dict_metrics = Softmax_class.test_class(self, batch[0], batch[1])
        for metric in dict_metrics:
            self._progress_printer.add_value_normzed_by_counter_samplers(metric, dict_metrics[metric])


    def _save_test_sampler(self, batch):
        file_mode = 'w'
        if self._counter_epochs != 0:
            file_mode = 'a'
        path_log = os.path.join(self._path_to_save_test, 'log.txt')
        with open(path_log, file_mode) as f:

            video = batch[0]
            label = np.argmax(batch[1])

            pred = np.squeeze(C.softmax(self._model(video)).eval())
            pred_label = np.argmax(pred)
            
            if (pred_label != label):
                f.write('{} epoch, {} test sample, predicted label = {}, true label = {}'.format(self._counter_epochs, self._counter_samplers, pred_label, label))
                

    def _saving_model(self, counter_epochs):
        self._model.save(os.path.join(self._path_to_save_model, '{1}_{0}.model'.format(self._name_model, counter_epochs)))

    def _building_model(self):

        conv3D1_1 = C.layers.Convolution3D(filter_shape=(3, 3, 3), num_filters=32, pad=True, name='conv3D1_1')(self._input)
        bn3D1_1 = C.layers.BatchNormalization(1, name='bn3D1_1')(conv3D1_1)
        relu3D1_1 = C.relu(bn3D1_1, name='relu3D1_1')
        pool3D1_1 = C.layers.MaxPooling(filter_shape=(1, 2, 2), strides=(1, 2, 2), name='pool3D1_1')(relu3D1_1)
        drop3D1_1 = C.layers.Dropout(dropout_rate=0.25, name='drop3D1_1')(pool3D1_1)

        conv3D2_1 = C.layers.Convolution3D(filter_shape=(3, 3, 3), num_filters=64, pad=True, name='conv3D2_1')(drop3D1_1)
        bn3D2_1 = C.layers.BatchNormalization(1, name='bn3D2_1')(conv3D2_1)
        relu3D2_1 = C.relu(bn3D2_1, name='relu3D2_1')
        pool3D2_1 = C.layers.MaxPooling(filter_shape=(1, 2, 2), strides=(1, 2, 2), name='pool3D2_1')(relu3D2_1)
        drop3D2_1 = C.layers.Dropout(dropout_rate=0.25, name='drop3D2_1')(pool3D2_1)

        conv3D2_2 = C.layers.Convolution3D(filter_shape=(3, 3, 3), num_filters=96, pad=True, name='conv3D2_2')(drop3D2_1)
        bn3D2_2 = C.layers.BatchNormalization(1, name='bn3D2_2')(conv3D2_2)
        relu3D2_2 = C.relu(bn3D2_2, name='relu3D2_2')
        pool3D2_2 = C.layers.MaxPooling(filter_shape=(1, 2, 2), strides=(1, 2, 2), name='pool3D2_2')(relu3D2_2)
        drop3D2_2 = C.layers.Dropout(dropout_rate=0.3, name='drop3D2_2')(pool3D2_2)
                          
        conv3D3_1 = C.layers.Convolution3D(filter_shape=(3, 3, 3), num_filters=96, pad=True, name='conv3D3_1')(drop3D2_2)
        bn3D3_1 = C.layers.BatchNormalization(1, name='bn3D3_1')(conv3D3_1)
        relu3D3_1 = C.relu(bn3D3_1, name='relu3D3_1')
        pool3D3_1 = C.layers.MaxPooling(filter_shape=(1, 2, 2), strides=(1, 2, 2), name='pool3D3_1')(relu3D3_1)
        drop3D3_1 = C.layers.Dropout(dropout_rate=0.5, name='drop3D3_1')(pool3D3_1)

        dense3D1_1 = C.layers.Dense(256, activation=C.relu, name='dense3D1_1')(drop3D3_1)
        drop3D1_1 = C.layers.Dropout(dropout_rate=0.5, name='drop3D1_1')(dense3D1_1)
        dense3D2_1 = C.layers.Dense(self._num_class, activation=None, name='dense3D2_1')(drop3D1_1)

        return C.layers.BatchNormalization(name='bn_out_softmax')(dense3D2_1)


    def _loading_model(self):
        pass


    def _loss_function(self):
        return Softmax_class.loss_softmax(self)


    def _metric_function(self):
        return Softmax_class.metric_softmax(self)


    def _make_nn_graph(self, model):
        name_img = self._name_model + '.png'
        name_doc = self._name_model + '.dot'
        path_img = os.path.join(self._path_to_save_model, name_img)
        path_doc = os.path.join(self._path_to_save_model, name_doc)

        C.logging.plot(model, path_img)
        C.logging.plot(model, path_doc)


    def convert_model_for_cpp(self, path_to_model):
        model = C.functions.load_model(path_to_model)
        model_softmax = C.ops.softmax(model, name='softmax_out')
        model_softmax.save(os.path.join(self._path_to_save_model, '{0}.model'.format(self._name_model)))
        self._make_nn_graph(model_softmax)



if __name__ == '__main__':
    print('classification_saberfighting_actions softmax mode')
    print(C.all_devices() )
    C.try_set_default_device(C.device.gpu(0))

    
    augmentation = Augmentations_class_im8_executor(
        [Type_augmentations_im.FLIP_HOR_IM],
        [Type_augmentations_im.MEDIAN_FILT_IM, 3],
        [Type_augmentations_im.BRIGHTNESS_RANDOM_IM, 20],
        [Type_augmentations_im.CONTRAST_RANDOM_IM, 0.7, 1.3]
        )

    reader_ds = Class_video_reader_ds(num_classes=3, 
                                      augmentation=augmentation,
                                      path_to_mapfile= r'D:\mine\diploma\Dataset\Softmax\map_file_new_concept.txt',
                                      percent_slice=0.1,
                                      step_folder=0,
                                      desired_size_ds=-1,
                                      type_load_im=cv2.IMREAD_COLOR,
                                      shape_to_resize=(224, 224), 
                                      sequence_length=10,
                                      num_chanels_input=3,
                                      coef_normalize=127.5)


    helper_model = Saberfighting_model_softmax_mode(
                                                 threshold_testing=0.5,
                                                 reader_ds=reader_ds, 
                                                 shape_input=(3, 10, 224, 224),
                                                 shape_input_label=(3),
                                                 num_class=3,
                                                 path_to_backbone=r'',
                                                 name_model='softmax_cnn_saberfighting',
                                                 path_to_save_model=r'D:\mine\diploma\Models\Siamese\with augmentation',#r'D:\mine\diploma\Models\Softmax_head\train_',
                                                 num_epochs=710,
                                                 size_batch_train=6,
                                                 size_batch_test=1,
                                                 testing_in_time_train=True,
                                                 save_frequency=10,
                                                 path_to_trained_model=r'D:\mine\diploma\Models\Siamese\with augmentation\700_softmax_cnn_saberfighting.model',#r'D:\mine\diploma\Models\Softmax_head\TrainingNewConcept\610_softmax_cnn_saberfighting.model!!!!!!',
                                                 path_to_save_test=r'D:\mine\diploma\Models\Siamese\with augmentation',#r'D:\mine\diploma\Models\Softmax_head\test_',
                                                 lr=0.001,
                                                 momentums=0.9,
                                                 booster=C.adam)

    #helper_model.init()
    #helper_model.train()
    helper_model.convert_model_for_cpp(r'D:\mine\diploma\Models\Siamese\with augmentation\700_softmax_cnn_saberfighting.model') # без ауг 610 
    print('Done')
