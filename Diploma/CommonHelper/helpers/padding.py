import numpy as np


def add_pad_to_right_down_corner(img, stride, value_pad):
    height = img.shape[0]
    width = img.shape[1]
    sides = 4 # 0-up, 1-left, 2-down, 3-right

    pad = np.zeros(sides)
    if (height % stride != 0):
        pad[2] = stride - (height % stride) # down
    if (width % stride != 0):
       pad[3] = stride - (width % stride) # right


    padded_img = img
    shape_img = padded_img[0:1, :, :].shape
    up_pad = np.tile(np.zeros(shape_img) + value_pad, (pad[0], 1, 1))

    padded_img = np.concatenate((up_pad, padded_img), axis=0)
    shape_img = padded_img[:, 0:1, :].shape
    left_pad = np.tile(np.zeros(shape_img) + value_pad, (1, pad[1], 1))

    padded_img = np.concatenate((left_pad, padded_img), axis=1)
    shape_img = padded_img[-2:-1, :, :].shape
    down_pad = np.tile(np.zeros(shape_img) + value_pad, (pad[2], 1, 1))

    padded_img = np.concatenate((padded_img, down_pad), axis=0)
    shape_img = padded_img[:, -2:-1, :].shape
    right_pad = np.tile(np.zeros(shape_img) + value_pad, (1, pad[3], 1))

    padded_img = np.concatenate((padded_img, right_pad), axis=1)

    return padded_img, pad