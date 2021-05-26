import cv2
import numpy as np
import random
import os
import sys
import cntk as C

from reader_ds.Base_video_reader_ds import Base_video_reader_ds
from sampler_ds.Class_video_sampler_ds import Class_video_sampler_ds

# NOTE
# потом нужно будет сделать класс для мульти-классификации триплет лос
# и использовать его там, скорее всего объектов классификации не изменится

class TripletLoss_video_reader_ds(Base_video_reader_ds):
    def __init__(self, 
                model, # сингл-модель, которую мы обучаем. Нужна, чтобы крафтить батчи
                shape_input_var, # размер инпута в модель
                path_to_anchor, # путь к анкерному изображению
                index_class_anchor, # индекс анкерного изображения
                eps, # разница между расстояниями позитива и негатива до аннкера
                threshold, # это значение модет сильно коллерировать c eps (условно гвояр этот тот попрог, который мы будем использовать в анализе) (можно задвать как половина eps)
                top_k, # если в датасете есть ошибки, то мы будем позитивные примеры рассматривать как неготивные(и разница расстояния будет мала). и такой кейс не даст обучиться модели. Говорим сколько первых самых негативных примеров мы хотим пропустить (обычно 0 - 5)
                counter_max_use_of_one_neg_sampler, # сколько раз при комбинации триплета мы можем использоват идин и тот же нег.пример

                augmentation, path_to_mapfile, percent_slice, step_folder, desired_size_ds,
                type_load_im = cv2.IMREAD_COLOR, 
                shape_to_resize = (-1, -1),
                num_chanels_input = 3, 
                coef_normalize = 255.0):
        super().__init__(augmentation, path_to_mapfile, percent_slice, step_folder, desired_size_ds, type_load_im, shape_to_resize, num_chanels_input, coef_normalize)
        self._num_classes = 2 # пока нет мультиклассификации навжно чему равен этот чувак 
        super()._init_reader()

        # настроки комбинирования батчей
        self._eps = eps
        self._threshold = threshold
        self._top_k = top_k
        self._counter_max_use_of_one_neg_sampler = counter_max_use_of_one_neg_sampler

        self._info_combination_train = dict()
        self._info_combination_test = dict()
        self._comb_train_ds = [] # для обучения триплета нудна давать 3 картинки, в этом сете в кажом элементе лежат по 3 кобминированных примера после составления батча
        self._comb_test_ds  = []
        self._model = model(shape_input_var - C.Constant(114))
        #self._input_var = C.input_variable(shape_input_var)
        self._path_to_anchor = path_to_anchor
        self._index_class_anchor = index_class_anchor
        self._sampler_anchor = self._create_sampler('{0}\t{1}'.format(self._path_to_anchor, self._index_class_anchor))

        self._embed_anchor, _ = self._get_embed_sampler(self._sampler_anchor)
        print(self._embed_anchor)
        self._combination_samplers_ds(self._train_ds, self._comb_train_ds, self._info_combination_train, self._counter_max_use_of_one_neg_sampler)
        self._combination_samplers_ds(self._test_ds, self._comb_test_ds, self._info_combination_test, 1)

#region public
    # размер train_ds
    @property
    def get_size_train_ds(self):
        return len(self._comb_train_ds)

    @property
    def get_sampler_anchor(self):
        return self._sampler_anchor

    # размер test_ds
    @property
    def get_size_test_ds(self):
        return len(self._comb_test_ds)

    def get_next_batch(self, batch_size, is_train_data, is_rand):
        ref_ds = self._train_ds
        comb_ds = self._comb_train_ds
        info_comb_ds = self._info_combination_train
        count_max_dist = self._counter_max_use_of_one_neg_sampler
        if (not is_train_data):
            ref_ds = self._test_ds
            comb_ds = self._comb_test_ds
            info_comb_ds = self._info_combination_test
            count_max_dist = 1

        if (batch_size > len(comb_ds)):
            comb_ds.clear()
            self._embed_anchor, _ = self._get_embed_sampler(self._sampler_anchor)
            self._combination_samplers_ds(ref_ds, comb_ds, info_comb_ds, count_max_dist)

        batch_anchors = []
        batch_poses   = []
        batch_neges   = []

        batch_neges_samplers = []
        for i in range(batch_size):
            chooseitem = 0
            if (is_rand):
                chooseitem = random.randrange(0, len(comb_ds))
            apn = comb_ds.pop(chooseitem)

            a_im, _= apn[0].get_data_sampler()
            p_im, _= apn[1].get_data_sampler()
            n_im, _= apn[2].get_data_sampler()

            batch_anchors.append(a_im)
            batch_poses.append(p_im)
            batch_neges.append(n_im)

            batch_neges_samplers.append(apn[2])

        return [np.asarray(batch_anchors), np.asarray(batch_poses),
               np.asarray(batch_neges), info_comb_ds, batch_neges_samplers]

#endregion

#region protected
    #
    def _create_sampler(self, parse_string):
        parse_string = parse_string.split('\t')
        path_to_video   = parse_string[0]
        index_class  = int(parse_string[1])
        sampler = Class_video_sampler_ds(path_to_video, index_class, self._num_classes, self._augmentation,
                                        self._type_load_im, self._shape_to_resize, self._num_chanels_input, self._coef_normalize)
        return sampler

    #
    def _get_embed_sampler(self, sampler):
        video_to_input, index_class = sampler.get_data_sampler()
        embed = np.squeeze(self._model(video_to_input))
        return embed, sampler.get_index_class 

    # 
    def _combination_samplers_ds(self, 
                          ref_ds, # из какого данные берем примеры
                          comb_ds, # в какой ds записываем скомбинированные примеры
                          info_comb_ds, # куда записываем информацию процесса комбинации примеров датасета
                          count_max_dist
                          ):
        matrix_embeds = dict()
        for s in ref_ds:
            embed, index = self._get_embed_sampler(s)
            # TODO переделать, если растояние для положительных и негативных считается по разному
            dist = np.sum(np.square(self._embed_anchor - embed))
            if index in matrix_embeds:
                matrix_embeds[index].append([dist, s])
            else:
                matrix_embeds[index] = [[dist, s]]

        # отпут для проверки обучения
        index_pos = self._index_class_anchor
        index_neg = abs(self._index_class_anchor - 1)

        info_comb_ds['mean_pos'] = np.mean( np.asarray( matrix_embeds[index_pos] )[:, 0])
        info_comb_ds['mean_neg'] = np.mean( np.asarray( matrix_embeds[index_neg] )[:, 0])
        info_comb_ds['std_pos']  = np.std( np.asarray( matrix_embeds[index_pos] )[:, 0])
        info_comb_ds['std_neg']  = np.std( np.asarray( matrix_embeds[index_neg] )[:, 0])

        info_comb_ds['max_dist_pos']  = np.max( np.asarray( matrix_embeds[index_pos] )[:, 0])
        info_comb_ds['min_dist_neg']  = np.min( np.asarray( matrix_embeds[index_neg] )[:, 0])
        info_comb_ds['error_pos_samplers'] = len([i for i in np.asarray(matrix_embeds[index_pos])[:, 0] if i > info_comb_ds['mean_neg']])   
        info_comb_ds['error_neg_samplers'] = len([i for i in np.asarray(matrix_embeds[index_neg])[:, 0] if i < info_comb_ds['mean_pos']])  

        for key in matrix_embeds:
            random.shuffle(matrix_embeds[key]) 

        # комбинируем позитивные примеры с негативными с макс ошибкой по разнице расстояний(чем расстояние меньше)
        counters_neg_samplers = dict() #
        error_combination = 0
        sum_error_dist = 0
        for pos_dist in matrix_embeds[index_pos]:
            neg_dists = np.asarray( matrix_embeds[index_neg] )[:, 0]
            # TODO для другой функции расчета разности расстояний
            map_diff_dists = neg_dists - pos_dist[0]
            indexes_sorted_diff_dists = np.argsort(map_diff_dists)
            counter_top_k = self._top_k
            ch_index_neg_sampler = 0

            while (True):
                if (counter_top_k >= len(indexes_sorted_diff_dists)):
                    counters_neg_samplers = dict()
                    counter_top_k = self._top_k
                index = indexes_sorted_diff_dists[counter_top_k]
                if index in counters_neg_samplers:
                    if (counters_neg_samplers[index] < count_max_dist):
                        counters_neg_samplers[index] += 1
                        ch_index_neg_sampler = index
                        break
                else:
                    counters_neg_samplers[index] = 0
                    ch_index_neg_sampler = index
                    break
                counter_top_k += 1

            if map_diff_dists[ch_index_neg_sampler] < 0: #self._threshold:
                error_combination += 1
            sum_error_dist += max(-(map_diff_dists[ch_index_neg_sampler] - self._eps), 0)
            comb_sampler = [self._sampler_anchor, pos_dist[1], matrix_embeds[index_neg][ch_index_neg_sampler][1]]
            comb_ds.append(comb_sampler)

        info_comb_ds['error_combination'] = error_combination / len(comb_ds)
        info_comb_ds['sum_error_dist'] = sum_error_dist / len(comb_ds)
#endregion
        

