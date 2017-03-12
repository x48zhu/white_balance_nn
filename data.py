import csv
import numpy as np
import scipy.io
import tensorflow as tf

from constants import *


def _pre_process_image(image):
    # Convert to UV space
    # (log(r/b), log(g/b)) ???
    image = np.stack((np.ma.log(image[:, :, 0] / image[:, :, 2]),
                      np.ma.log(image[:, :, 1] / image[:, :, 2]))).T
    return image


def load_data(data_set):
    print("Loading images...")
    images = []
    img_dir = os.path.join(data_dir, data_set, 'NEW')
    for image_file in os.listdir(img_dir):
        image = (scipy.io.loadmat(os.path.join(img_dir, image_file)))['img']
        images.append(image)

    print("Loading labels...")
    csv_file = open(os.path.join(data_dir, data_set + '.csv'), 'r')
    csv_file_object = csv.reader(csv_file)
    data = []
    for row in csv_file_object:
        data.append(row[-2:])
    return images, data


def split_to_patches(X_data, y_data):
    # Split images into patches, and repeat label accordingly
    with tf.Session() as sess:
        patches = tf.extract_image_patches(images=X_data,
                                           ksizes=[1, PATCH_SIZE[0], PATCH_SIZE[1], 1],
                                           strides=[1, PATCH_SIZE[0], PATCH_SIZE[1], 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')
        num_train_X, num_patch_row, num_patch_col, depth = patches.shape
        patch_X = tf.reshape(patches,
                             [num_train_X * num_patch_row * num_patch_col, PATCH_SIZE[0], PATCH_SIZE[1], 3]).eval()
    patch_y = np.repeat(y_data, num_patch_row * num_patch_col, axis=0)
    return patch_X, patch_y
