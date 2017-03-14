import csv
import numpy as np
import scipy.io
import tensorflow as tf

import cv2
from constants import *


def _pre_process_image(image):
    # Convert to UV space
    # (log(r/b), log(g/b)) ???
    image = np.stack((np.ma.log(image[:, :, 0] / image[:, :, 2]),
                      np.ma.log(image[:, :, 1] / image[:, :, 2]))).T
    return image


def load_data(data_set, debug=False):
    print("Loading images...")
    images = []
    img_dir = os.path.join(data_dir, data_set, 'NEW')
    img_files = os.listdir(img_dir)
    if debug:
        img_files = img_files[:DEBUG_DATA_SIZE]
    for img_file in img_files:
        image = (scipy.io.loadmat(os.path.join(img_dir, img_file)))['img']
        # image = cv2.imread(os.path.join(img_dir, img_file))
        # TODO: hardcode the image size here
        image = np.reshape(image, IMAGE_SIZE)
        images.append(image)

    print("Loading labels...")
    csv_file = open(os.path.join(data_dir, data_set + '.csv'), 'r')
    csv_file_object = csv.reader(csv_file)
    data = []
    for row in csv_file_object:
        data.append([float(s) for s in row[-2:]])
    if debug:
        data = data[:DEBUG_DATA_SIZE]
    return np.array(images), np.array(data)


def split_to_patches(X_data, y_data):
    # Split images into patches, and repeat label accordingly
    with tf.Session() as sess:
        print(X_data.shape, X_data.dtype)
        patches = tf.extract_image_patches(images=X_data,
                                           ksizes=[1, PATCH_SIZE[0], PATCH_SIZE[1], 1],
                                           strides=[1, PATCH_SIZE[0], PATCH_SIZE[1], 1],
                                           rates=[1, 1, 1, 1],
                                           padding="VALID").eval()
        num_train_X, num_patch_row, num_patch_col, depth = patches.shape
        patch_X = tf.reshape(patches,
                             [num_train_X * num_patch_row * num_patch_col, PATCH_SIZE[0], PATCH_SIZE[1], 3]).eval()
    patch_y = np.repeat(y_data, num_patch_row * num_patch_col, axis=0)
    return patch_X, patch_y
