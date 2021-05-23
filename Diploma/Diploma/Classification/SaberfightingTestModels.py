import numpy as np
import cntk as C
import os
import matplotlib.pyplot as plt
import cv2

import helpers.cntk_helper as cntk_helper
from sampler_ds.Class_image_sampler_ds import Class_image_sampler_ds
from reader_ds.Class_image_reader_ds import Class_image_reader_ds
from helpers.augmentations import *


class Saberfighting_test_models():
    def __init__(self, shape_input, path_to_softmax_model, path_to_triplet_horiz_model, path_to_triplet_vertic_model, path_to_save):

        self._shape_input = shape_input
        self._path_to_softmax_head = path_to_softmax_model 
        self._path_to_triplet_head_horiz = path_to_triplet_horiz_model
        self._path_to_triplet_head_vertic = path_to_triplet_vertic_model
        self._path_to_save = path_to_save
        self._model = None

        # constants
        self._size_dim = 16 # dim embedding
        self._num_class_softmax = 3

        self.softmax_error = 0
        self.horiz_error = 0
        self.vertic_error = 0
        self.error = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0


    # Тестируем модели: - 1) Softmax 2) Triplet vertic 3) Triplet horiz
    def test(self, reader_ds):
        model_softmax_ = C.functions.load_model(self._path_to_softmax_head)
        model_vertic_ = C.functions.load_model(self._path_to_triplet_head_vertic)
        model_horiz_ = C.functions.load_model(self._path_to_triplet_head_horiz)
        
        model_softmax = model_softmax_.clone(C.CloneMethod.share)
        model_horiz = model_horiz_.clone(C.CloneMethod.share)
        model_vertic = model_vertic_.clone(C.CloneMethod.share)
        # классы
        vertic_softmax = 1
        horiz_softmax = 2

        vertic_bad = 2
        vertic_good = 3
        vertic_threshold = 2.1e-9# best #2.73e-9 без аугм #2.1e-9 с аугм

        horiz_bad = 4
        horiz_good = 5
        horiz_threshold = 2.7e-12# best #2.74 без аугм #2.5e-13 с аугм

         #без аугментации модельки+
       # vertic_anchor = np.array([-0.18884414, -0.0928681 ,  0.18590991, -0.0595417 ,  0.23007217,
       # 0.15286654,  0.24210662, -0.45788047, -0.19504097, -0.11219861,
       # 0.07228126, -0.28451955,  0.04778492,  0.55277246,  0.2672805 ,
       #-0.29293838], dtype=np.float32)
       # horiz_anchor = np.array([ 0.37825283, -0.32875   , -0.26116198, -0.19779913, -0.15653063,
       # 0.23179634,  0.24786645,  0.80768085,  0.86089367, -0.14151904,
       #-0.21637143,  0.2141767 ,  0.11507547, -0.54069775,  0.1916414 ,
       # 0.12011523], dtype=np.float32)

       # с аугментацией       
        vertic_anchor = np.array([-0.24746865, 0.03978111, 0.13912132, -0.01408572, 0.03013963, -0.06000096,
        -0.3303202, 0.1385255, 0.04360551, -0.05080905, -0.02672888, -0.11045513,
        0.10860564, -0.07108926, 0.05151974, -0.08624982], dtype=np.float32)
        horiz_anchor = np.array([ 0.00753562, -0.09878368, -0.0634775, 0.17690332, 0.11552318, 0.07101858,
        0.18993254, 0.02002085, -0.29981878, 0.0445806, 0.038764, -0.02385881,
        0.14823848, 0.27717984, 0.10368124, 0.13088007], dtype=np.float32)

        samples = reader_ds._train_ds
        cnt_samples = 0
        cnt_samples_vertic = 0
        cnt_samples_horiz = 0

        triplet_vertic_good, triplet_vertic_bad, triplet_horiz_good, triplet_horiz_bad = [], [], [], []

        path_log = os.path.join(self._path_to_save, 'log.txt')
        with open(path_log, 'w') as f:
            with open(reader_ds._path_to_mapfile, 'r') as mapfile:
                for item in samples:
                    video, data_label = item.get_data_sampler()
                    line = mapfile.readline()
                    splitten = line.split('\n')[0].split('\t')
                    label_softmax, label_triplet = int(splitten[1]), int(splitten[2])

                    predictions = np.squeeze(model_softmax.eval(video))

                    softmax_pred = np.argmax(predictions)

                    ## ==== test =======
                    #if label_triplet == vertic_good:
                    #    triplet_vertic_good.append(vertic_pred_dist)
                    #    print("V G")
                    #    print(vertic_pred_dist)
                    #    print(horiz_pred_dist)
                    #elif label_triplet == vertic_bad:
                    #    triplet_vertic_bad.append(vertic_pred_dist)
                    #    print("V B")
                    #    print(vertic_pred_dist)
                    #    print(horiz_pred_dist)
                    #elif label_triplet == horiz_good:
                    #    print("H G")
                    #    print(vertic_pred_dist)
                    #    print(horiz_pred_dist)
                    #    triplet_horiz_good.append(horiz_pred_dist)
                    #elif label_triplet == horiz_bad:
                    #    triplet_horiz_bad.append(horiz_pred_dist)
                    #    print("H B")
                    #    print(vertic_pred_dist)
                    #    print(horiz_pred_dist)
                    ## ==== test =======

                    if label_softmax == vertic_softmax: # something from vertic hits
                        if label_softmax != softmax_pred: # but predicted it is NOT vertic hit
                            self.softmax_error += 1
                            self.error += 1
                        else:
                            vertic_triplet_pred = np.squeeze(model_vertic.eval(video))

                            vertic_pred_dist = np.sum(np.square(vertic_anchor - vertic_triplet_pred))
                            
                            pred_label = vertic_good if vertic_pred_dist < vertic_threshold else vertic_bad
                            is_ok = False if label_triplet != pred_label else True
                            cnt_samples_vertic += 1
                            if is_ok and label_triplet == vertic_good:
                                self.TP += 1
                                triplet_vertic_good.append(vertic_pred_dist)
                            elif is_ok and label_triplet == vertic_bad:
                                self.TN += 1
                                triplet_vertic_bad.append(vertic_pred_dist)
                            elif not is_ok and label_triplet == vertic_good:
                                self.vertic_error += 1
                                self.FN += 1
                                self.error += 1
                                triplet_vertic_good.append(vertic_pred_dist)
                            else:
                                self.vertic_error += 1
                                self.FP += 1
                                self.error += 1
                                triplet_vertic_bad.append(vertic_pred_dist)

                    elif label_softmax == horiz_softmax: # something from horiz hits
                        if label_softmax != softmax_pred: # but predicted it is NOT horiz hit
                            self.softmax_error += 1
                            self.error += 1
                        else:
                            horiz_triplet_pred = np.squeeze(model_horiz.eval(video))
                            horiz_pred_dist = np.sum(np.square(horiz_anchor - horiz_triplet_pred))

                            pred_label = horiz_good if horiz_pred_dist < horiz_threshold else horiz_bad
                            is_ok = False if label_triplet != pred_label else True
                            cnt_samples_horiz += 1
                            if is_ok and label_triplet == horiz_good:
                                self.TP += 1
                                triplet_horiz_good.append(horiz_pred_dist)
                            elif is_ok and label_triplet == horiz_bad:
                                self.TN += 1
                                triplet_horiz_bad.append(horiz_pred_dist)
                            elif not is_ok and label_triplet == horiz_good:
                                self.horiz_error += 1
                                self.FN += 1
                                self.error += 1
                                triplet_horiz_good.append(horiz_pred_dist)
                            else:
                                self.horiz_error += 1
                                self.FP += 1
                                self.error += 1
                                triplet_horiz_bad.append(horiz_pred_dist)
                    else: # it is NOT fencing
                        if label_softmax != softmax_pred: # but predicted it IS fencing
                            self.softmax_error += 1
                            self.error += 1
                    cnt_samples += 1
                    item.clear_data_sampler()

                f.write('cnt samples: {} cnt_samples_vertic: {} cnt_samples_horiz: {}'.format(cnt_samples, cnt_samples_vertic, cnt_samples_horiz))
                print('cnt samples: {} cnt_samples_vertic: {} cnt_samples_horiz: {}'.format(cnt_samples, cnt_samples_vertic, cnt_samples_horiz))

                f.write('cnt error: {} cnt softmax_error: {} cnt vertic_triplet_error: {} cnt horiz_triplet_error: {}'.format(self.error, self.softmax_error,
                self.vertic_error, self.horiz_error))
                print('cnt error: {} cnt softmax_error: {} cnt vertic_triplet_error: {} cnt horiz_triplet_error: {}'.format(self.error, self.softmax_error,
                self.vertic_error, self.horiz_error))
                
                self.error /= cnt_samples
                self.softmax_error /= cnt_samples
                self.vertic_error /= cnt_samples_vertic
                self.horiz_error /= cnt_samples_horiz
                f.write('error: {} softmax_error: {} vertic_triplet_error: {} horiz_triplet_error: {}'.format(self.error, self.softmax_error,
                                                                                                              self.vertic_error, self.horiz_error))
                print('error: {} softmax_error: {} vertic_triplet_error: {} horiz_triplet_error: {}'.format(self.error, self.softmax_error,
                                                                                                            self.vertic_error, self.horiz_error))
                accuracy = (self.TP + self.TN) / cnt_samples
                precision = self.TP / (self.TP + self.FN)
                recall = self.TP / (self.TP + self.FP)
                f1_score = 2 * (1 / (1/precision + 1/recall))
                f.write('accuracy: {} precision: {} recall: {} f1_score: {}'.format(accuracy, precision, recall, f1_score))
                print('accuracy: {} precision: {} recall: {} f1_score: {}'.format(accuracy, precision, recall, f1_score))
        print('vertic good')
        print(triplet_vertic_good)
        print('MEAN')
        print(np.mean(np.asarray(triplet_vertic_good)))
        print('MIN')
        print(np.min(np.asarray(triplet_vertic_good)))
        print('MAX')
        print(np.max(np.asarray(triplet_vertic_good)))

        print('triplet_vertic_bad')
        print(triplet_vertic_bad)
        print('MEAN')
        print(np.mean(np.asarray(triplet_vertic_bad)))
        print('MIN')
        print(np.min(np.asarray(triplet_vertic_bad)))
        print('MAX')
        print(np.max(np.asarray(triplet_vertic_bad)))

        print('triplet_horiz_good')
        print(triplet_horiz_good)
        print('MEAN')
        print(np.mean(np.asarray(triplet_horiz_good)))
        print('MIN')
        print(np.min(np.asarray(triplet_horiz_good)))
        print('MAX')
        print(np.max(np.asarray(triplet_horiz_good)))

        print('triplet_horiz_bad')
        print(triplet_horiz_bad)
        print('MEAN')
        print(np.mean(np.asarray(triplet_horiz_bad)))
        print('MIN')
        print(np.min(np.asarray(triplet_horiz_bad)))
        print('MAX')
        print(np.max(np.asarray(triplet_horiz_bad)))



if __name__ == '__main__':
    print('Begin testing')
    #augmentation = Augmentations_class_im8_executor()
    augmentation = Augmentations_class_im8_executor(
    [Type_augmentations_im.MEDIAN_FILT_IM, 3],
    [Type_augmentations_im.BRIGHTNESS_RANDOM_IM, 20],
    [Type_augmentations_im.CONTRAST_RANDOM_IM, 0.7, 1.3]
    )
    reader_ds = Class_image_reader_ds(num_classes=3, 
                                      augmentation=augmentation,
                                      path_to_mapfile= r'D:\mine\diploma\Dataset\Siamese\map_file.txt',
                                      percent_slice=0,
                                      step_folder=0,
                                      desired_size_ds=-1,
                                      type_load_im=cv2.IMREAD_COLOR,
                                      shape_to_resize=(224, 224), 
                                      sequence_length=10,
                                      num_chanels_input=3,
                                      coef_normalize=127.5)
 #обычный триплет без аугм
    #tester = Saberfighting_test_models(shape_input = (3, 10, 224, 224),
    #                                            path_to_softmax_model = r'D:\mine\diploma\Models\Siamese\softmax_cnn_saberfighting.model',
    #                                            path_to_triplet_horiz_model = r'D:\mine\diploma\Models\Siamese\horiz_triplet_saberfighting.model',
    #                                            path_to_triplet_vertic_model = r'D:\mine\diploma\Models\Siamese\vertic_triplet_saberfighting.model',
    #                                            path_to_save = r'D:\mine\diploma\Models\Siamese')
# улучшенный триплет + аугментация
    tester = Saberfighting_test_models(shape_input = (3, 10, 224, 224),
                                                path_to_softmax_model = r'D:\mine\diploma\Models\Siamese\with augmentation\softmax_cnn_saberfighting.model',
                                                path_to_triplet_horiz_model = r'D:\mine\diploma\Models\Siamese\with augmentation\horiz_triplet_saberfighting.model',
                                                path_to_triplet_vertic_model = r'D:\mine\diploma\Models\Siamese\with augmentation\vertic_triplet_saberfighting.model',
                                                path_to_save = r'D:\mine\diploma\Models\Siamese\with augmentation')

    tester.test(reader_ds)
    print('Done')