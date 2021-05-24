
import numpy as np
import cntk as C
import os
import matplotlib.pyplot as plt
import cv2
import random

import helpers.cntk_helper as cntk_helper
from reader_ds.TripletLoss_image_reader_ds import TripletLoss_image_reader_ds
from helper_model.Base_helper_cntk_model import Base_helper_cntk_model
from sampler_ds.Class_image_sampler_ds import Class_image_sampler_ds
from store_models.Triplet_class import Triplet_class
from helpers.augmentations import *

        
class Saberfighting_model_triplet_mode(Base_helper_cntk_model, Triplet_class):
    def __init__(self,
                 size_dim,
                 beta,
                 shape_input,
                 #region Передаем параметры сюда, тк для создания ридера, тк он создается внутри helper'a модели
                 path_to_anchor, index_class_anchor, eps, threshold, top_k, counter_max_use_of_one_neg_sampler,
                 augmentation, path_to_mapfile, percent_slice, step_folder, desired_size_ds,
                 type_load_im, shape_to_resize, num_channels_input, coef_normalize,
                 #endregion
                 name_model, path_to_save_model, path_to_backbone,
                 num_epochs, size_batch_train, size_batch_test,
                 testing_in_time_train, save_frequency, path_to_trained_model, 
                 path_to_save_test='', lr=0.001, momentums=0.9, booster=C.learners.adam):


        self._incr_test = 0
        self._size_dim = size_dim
        self._shape_input = shape_input
        self._beta = beta
        self._eps = eps
        self._threshold = threshold

        # Параметры для модели
        self._path_to_backbone = path_to_backbone
        self._path_to_trained_model = path_to_trained_model

        self._model = self._build_model() # если уже есть обученная модель, то подгружаем ее, если нет, то создаем

        Triplet_class.__init__(self, size_dim, shape_input, self._model, self._path_to_backbone, eps, threshold)

        reader_ds = TripletLoss_image_reader_ds(self._model, self._input_anc, 
                                                path_to_anchor, index_class_anchor, eps, threshold, top_k, counter_max_use_of_one_neg_sampler,
                                                augmentation, path_to_mapfile, percent_slice, step_folder, desired_size_ds,
                                                type_load_im, shape_to_resize, num_channels_input, coef_normalize)

        super().__init__(reader_ds, name_model, path_to_save_model,
                       num_epochs, size_batch_train, size_batch_test,
                       testing_in_time_train, save_frequency, path_to_trained_model,
                       path_to_save_test, lr, momentums, booster)

        # Отрисуем граф построенной модели
        self._make_nn_graph(self._comb_model)


    @property
    def get_model(self):
        self._model


    def init(self):
        mm_schedule = C.learners.momentum_schedule(self._mm_schedule, \
                                                    epoch_size= self._size_epoch_train, \
                                                    minibatch_size=self._size_batch_train)
        self._learner = self._booster(self._model.parameters, lr = self._lr, momentum = mm_schedule,
                                      l2_regularization_weight=0.0002,#lr,
                                      gradient_clipping_threshold_per_sample=12,
                                      gradient_clipping_with_truncation=True)
       #self._learner = self._booster(self._model.parameters, lr=self._lr, momentum=self._momentums)
        progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=self._num_epochs)
        self._trainer = C.Trainer(self._comb_model, (self._loss_function(), self._metric_function()), [self._learner], [progress_printer])
        C.logging.log_number_of_parameters(self._model) ; print()


    def _training(self, batch):
        self._trainer.train_minibatch({self._input_anc: batch[0], self._input_pos: batch[1], self._input_neg: batch[2]})
        eval_error = self._trainer.previous_minibatch_evaluation_average
        loss_average = self._trainer.previous_minibatch_loss_average
        dict_metrics = batch[3]
        dict_metrics['train loss'] = loss_average
        dict_metrics['train average eval erorr'] = eval_error
        for metric in dict_metrics:
            self._progress_printer.add_value(metric, dict_metrics[metric]) #add_value_normzed_by_counter_batchs


    def _testing(self, batch):
        res = np.squeeze(self._comb_model(batch[0], batch[1], batch[2]))
        anchor = res[0 : self._size_dim]
        positive = res[self._size_dim : 2 * self._size_dim]
        negative = res[2 * self._size_dim : 3 * self._size_dim]

        pos_dist = np.sum(np.square(anchor - positive))
        neg_dist = np.sum(np.square(anchor - negative))

        diff = pos_dist - neg_dist

        error = 0 if diff < 0 else 1 # <= 0

        self._progress_printer.add_value_normzed_by_counter_samplers('test_error', error)
        self._progress_printer.add_value_normzed_by_counter_samplers('mean_diff', diff)
        self._progress_printer.add_value_normzed_by_counter_samplers('mean_pos_dist', pos_dist)
        self._progress_printer.add_value_normzed_by_counter_samplers('mean_neg_dist', neg_dist)


    def _create_layers(self, input):
        list_denses = []
        model_vgg = C.load_model(self._path_to_backbone)
        freez = cntk_helper.get_layers_by_names(model_vgg, 'data', 'pool3', True)
        clone = cntk_helper.get_layers_by_names(model_vgg, 'pool3', 'pool4', False)

        for i in range(10):
            name_i = 'horiz_squeeze_' + str(i)
            frame_input = C.squeeze(input[:, i, :, :], name=name_i)
            d1 = clone(freez(frame_input - C.Constant(114)))
            list_denses.append(d1)

        spl = C.ops.splice(*list_denses)

        custom_conv1_1 = C.layers.Convolution2D((1, 3), 512, pad = True, name='horiz_custom_conv1_1')(spl)
        custom_bn1_1 = C.layers.BatchNormalization(1, name='horiz_custom_bn1_1')(custom_conv1_1)
        custom_relu1_1 = C.relu(custom_bn1_1, name='horiz_custom_relu1_1')
        custom_pool1_1 = C.layers.MaxPooling(filter_shape=(1, 2), strides=(1, 2), name='horiz_custom_pool1_1')(custom_relu1_1)

        custom_conv2_1 = C.layers.Convolution2D((1, 3), 256, pad = True, name='horiz_custom_conv2_1')(custom_pool1_1)
        custom_bn2_1 = C.layers.BatchNormalization(1, name='horiz_custom_bn2_1')(custom_conv2_1)
        custom_relu2_1 = C.relu(custom_bn2_1, name='horiz_custom_relu2_1')
        custom_pool2_1 = C.layers.MaxPooling(filter_shape=(1, 2), strides=(1, 2), name='horiz_custom_pool2_1')(custom_relu2_1)


        custom_dense1_1 = C.layers.Dense(256, activation=C.relu, name='horiz_custom_dense1_1')(custom_pool2_1)
        custom_drop1_1 = C.layers.Dropout(0.5, name='vcustom_drop1_1')(custom_dense1_1)
        dense_out = C.layers.Dense(self._size_dim, activation=C.sigmoid, name='horiz_dense_out')(custom_drop1_1)

        return dense_out 


    def _building_model(self):
        input = C.input_variable(self._shape_input, name='horiz_input')
        output = self._create_layers(input)

        return C.layers.BatchNormalization(name='horiz_bn_out')(output)


    def _loss_function(self):
        anchor = self._comb_model[0 : self._size_dim]
        positive = self._comb_model[self._size_dim : 2 * self._size_dim]
        negative = self._comb_model[2 * self._size_dim : 3 * self._size_dim]
        
        # расстояние между анкером и позитивным сэмплом
        pos_dist = C.reduce_sum(C.square(C.minus(anchor, positive)))

        # расстояние между анкером и негативным сэмплом
        neg_dist = C.reduce_sum(C.square(C.minus(anchor, negative)))

        # считаем разницу позитивного и негативного расстояния
        # мы стремимся к увеличению негативного расстояния и соответственно уменьшению позитивного расстояния
        basic_loss = pos_dist - neg_dist + self._eps

        non_linear_pos = -C.log(-C.element_divide(pos_dist, self._beta) + 1 + self._eps)
        non_linear_neg = -C.log(-C.element_divide(self._size_dim - neg_dist, self._beta) + 1 + self._eps)
        non_linear_loss = non_linear_pos + non_linear_neg + self._eps
        return non_linear_loss
        #loss = C.clip(basic_loss, 0.0, 1000000000) # эквивалентно операции: min( max(x, min_val=0.0), max_val=1000000000 )
        ##losses = C.relu(basic_loss)
        #return C.mean(loss) #if size_average else losses.sum() C.mean()


    def _metric_function(self):
        anchor = self._comb_model[0 : self._size_dim]
        positive = self._comb_model[self._size_dim : 2 * self._size_dim]
        negative = self._comb_model[2 * self._size_dim : 3 * self._size_dim]

        # расстояние между анкером и позитивным сэмплом
        pos_dist = C.reduce_sum(C.square(C.minus(anchor, positive)))

        # расстояние между анкером и негативным сэмплом
        neg_dist = C.reduce_sum(C.square(C.minus(anchor, negative)))

        # считаем разницу позитивного и негативного расстояния
        # мы стремимся к увеличению негативного расстояния и соответственно уменьшению позитивного расстояния
        basic_difference = pos_dist - neg_dist + self._eps
        error = C.clip(basic_difference, 0.0, 1.0)
        qry_ans_similarity = C.cosine_distance_with_negative_samples(positive, \
                                                                 negative, \
                                                                 shift=1, \
                                                                 num_negative_samples=2)
        return qry_ans_similarity
        #return 1 - error


    def _loading_model(self):
        return C.load_model(self._path_to_trained_model)


    def _saving_model(self, counter_epochs):
        self._model.save(os.path.join(self._path_to_save_model, '{1}_{0}.model'.format(self._name_model, counter_epochs)))
       

    def _save_test_sampler(self, batch):
        file_mode = 'w'
        if self._counter_epochs != 0:
            file_mode = 'a'
        path_log = os.path.join(self._path_to_save_test, 'log.txt')
        with open(path_log, file_mode) as f:

            chooseitem = random.randrange(0, 2)
            true_label = 0
            index = 2 # neg
            opposite_index = 1 # pos
            if (chooseitem != 0):
                true_label = 1
                index = 1 # pos
                opposite_index = 2 # neg

            pred = np.squeeze(self._model.eval(batch[index][0]))
            anchor = self._reader_ds._embed_anchor
            opposite_pred = np.squeeze(self._model.eval(batch[opposite_index][0]))
            pred_dist = np.sum(np.square(anchor - pred))
            opposite_pred_dist = np.sum(np.square(anchor - opposite_pred))
            dist = pred_dist - opposite_pred_dist

            pred_label = 1 if dist < 0 else 0
 
            if (pred_label != true_label):
                f.write('{} epoch, {} test sample, predicted label = {}, true label = {}\n'.format(self._counter_epochs, self._counter_samplers, pred_label, true_label))


    def _make_nn_graph(self, model):
        name_img = self._name_model + '.png'
        name_doc = self._name_model + '.dot'
        path_img = os.path.join(self._path_to_save_model, name_img)
        path_doc = os.path.join(self._path_to_save_model, name_doc)

        #C.logging.plot(model, path_img)
        C.logging.plot(model, path_doc)


    # загружаем модель и конвертируем ее в модель для с++
    def convert_model_for_cpp(self):
        self._head_anc.save(os.path.join(self._path_to_save_model, '{0}.model'.format(self._name_model)))
        self._make_nn_graph(self._head_anc)


if (__name__ == '__main__'):
    print('classification_saberfighting_actions triplet mode')

    augmentation = Augmentations_class_im8_executor(
        [Type_augmentations_im.MEDIAN_FILT_IM, 3],
        [Type_augmentations_im.BRIGHTNESS_RANDOM_IM, 20],
        [Type_augmentations_im.CONTRAST_RANDOM_IM, 0.7, 1.3]
        )

    #test = Saberfighting_model_triplet_mode(size_dim=16, #32
    #                                shape_input=(3, 10, 224, 224), # (channels, frames, height, width)
    #                                path_to_anchor=r'D:\mine\diploma\Dataset\Data\Skeleton\Horiz\Good\horiz_good_28', 
    #                                index_class_anchor=1, 
    #                                eps=20,
    #                                threshold=2, 
    #                                top_k=5, 
    #                                counter_max_use_of_one_neg_sampler=32,
    #                                augmentation= augmentation,
    #                                path_to_mapfile=r'D:\mine\diploma\Dataset\Triplet\Horiz_triplet\map_file2.txt',
    #                                percent_slice=0.1,
    #                                step_folder=0.5,
    #                                desired_size_ds=-1, # желаемый размер ds (диапазон от 0 до 1). Если не попадает в диапазон, то размер из количества объектов в мапфайле
    #                                type_load_im=cv2.IMREAD_COLOR,
    #                                shape_to_resize=(224, 224), 
    #                                num_channels_input=3,
    #                                coef_normalize=127.5,
    #                                name_model='horiz_triplet_saberfighting',
    #                                path_to_save_model=r'',
    #                                path_to_backbone=r'',
    #                                num_epochs=1010,
    #                                size_batch_train=12,
    #                                size_batch_test=1,
    #                                testing_in_time_train=True,
    #                                save_frequency=10, # каждые save_frequency эпох сохраняем модель
    #                                path_to_trained_model=r'', 
    #                                path_to_save_test=r'D:\mine\diploma\Models\Triplet_head\Vertic_triplet\f\Testing',
    #                                lr=0.01,#0.1
    #                                momentums=0.9,
    #                                booster=C.adam)

    #test = Saberfighting_model_triplet_mode(size_dim=16,
    #                                beta=16,
    #                                shape_input=(3, 10, 224, 224), # (channels, frames, height, width)
    #                                path_to_anchor=r'D:\mine\diploma\Dataset\Data\Skeleton\Vertic\Good\vertic_good_29', 
    #                                index_class_anchor=1, 
    #                                eps=2.3,
    #                                threshold=2, 
    #                                top_k=2, 
    #                                counter_max_use_of_one_neg_sampler=32,
    #                                augmentation= augmentation,
    #                                path_to_mapfile=r'D:\mine\diploma\Dataset\Triplet\Vertic_triplet\map_file2.txt',
    #                                percent_slice=0.1,
    #                                step_folder=0.5,
    #                                desired_size_ds=-1, # желаемый размер ds (диапазон от 0 до 1). Если не попадает в диапазон, то размер из количества объектов в мапфайле
    #                                type_load_im=cv2.IMREAD_COLOR,
    #                                shape_to_resize=(224, 224), 
    #                                num_channels_input=3,
    #                                coef_normalize=127.5,
    #                                name_model='vertic_triplet_saberfighting',
    #                                path_to_save_model=r'D:\mine\diploma\Models\Siamese\with augmentation',#r'D:\mine\diploma\Models\Triplet_head\Vertic_triplet\new_loss_train',#r'D:\mine\diploma\Models\Triplet_head\Vertic_triplet\f\tt',
    #                                path_to_backbone=r'D:\mine\diploma\Models\VGG16_ImageNet_Caffe.model',
    #                                num_epochs=1010,
    #                                size_batch_train=12,
    #                                size_batch_test=1,
    #                                testing_in_time_train=True,
    #                                save_frequency=10, # каждые save_frequency эпох сохраняем модель
    #                                path_to_trained_model=r'D:\mine\diploma\Models\Siamese\with augmentation\650_vertic_triplet_saberfighting.model',#r'D:\mine\diploma\Models\Triplet_head\Vertic_triplet\f\tt\460_vertic_triplet_saberfighting.model', 
    #                                path_to_save_test=r'D:\mine\diploma\Models\Siamese\with augmentation',#r'D:\mine\diploma\Models\Triplet_head\Vertic_triplet\new_loss_test',#r'D:\mine\diploma\Models\Triplet_head\Vertic_triplet\f\t',
    #                                lr= 0.01,
    #                                momentums=[0]*20 + [0.9200444146293233]*20 + [0.9591894571091382],#0.9,
    #                                booster=C.adam)
    test = Saberfighting_model_triplet_mode(size_dim=16,
                                    beta=16,
                                    shape_input=(3, 10, 224, 224), # (channels, frames, height, width)
                                    path_to_anchor=r'D:\mine\diploma\Dataset\Data\Skeleton\Horiz\Good\horiz_good_28', 
                                    index_class_anchor=1, 
                                    eps=2.3,
                                    threshold=2, 
                                    top_k=2, 
                                    counter_max_use_of_one_neg_sampler=32,
                                    augmentation= augmentation,
                                    path_to_mapfile=r'D:\mine\diploma\Dataset\Triplet\Horiz_triplet\map_file2.txt',
                                    percent_slice=0.1,
                                    step_folder=0.5,
                                    desired_size_ds=-1, # желаемый размер ds (диапазон от 0 до 1). Если не попадает в диапазон, то размер из количества объектов в мапфайле
                                    type_load_im=cv2.IMREAD_COLOR,
                                    shape_to_resize=(224, 224), 
                                    num_channels_input=3,
                                    coef_normalize=127.5,
                                    name_model='horiz_triplet_saberfighting',
                                    path_to_save_model=r'D:\mine\diploma\Models\Siamese\with augmentation',#r'D:\mine\diploma\Models\Triplet_head\Horiz_triplet\new_loss_train',#r'D:\mine\diploma\Models\Triplet_head\Vertic_triplet\f\tt',
                                    path_to_backbone=r'D:\mine\diploma\Models\VGG16_ImageNet_Caffe.model',
                                    num_epochs=1010,
                                    size_batch_train=12,
                                    size_batch_test=1,
                                    testing_in_time_train=True,
                                    save_frequency=10, # каждые save_frequency эпох сохраняем модель
                                    path_to_trained_model=r'D:\mine\diploma\Models\Siamese\with augmentation\600_horiz_triplet_saberfighting.model',#r'D:\mine\diploma\Models\Triplet_head\Vertic_triplet\f\tt\460_vertic_triplet_saberfighting.model', 
                                    path_to_save_test=r'D:\mine\diploma\Models\Siamese\with augmentation',#r'D:\mine\diploma\Models\Triplet_head\Horiz_triplet\new_loss_test',#r'D:\mine\diploma\Models\Triplet_head\Vertic_triplet\f\t',
                                    lr= 0.01,
                                    momentums=[0]*20 + [0.9200444146293233]*20 + [0.9591894571091382],#0.9,
                                    booster=C.adam)
    
    #test.init()
    #test.train()
    #test.convert_model_for_cpp(r'D:\mine\diploma\Models\Triplet_head\Vertic_triplet\VerticReady\horiz_triplet_saberfighting_anc.model')
    test.convert_model_for_cpp()

    print('Done')

