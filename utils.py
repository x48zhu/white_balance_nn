import logging
import math
import numpy as np
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def angular_error_scalar(output, labels, debug=False):
    output_blue = 1 - output[:, 0] - output[:, 1]
    labels_blue = 1 - labels[:, 0] - labels[:, 1]
    output = np.concatenate((output, output_blue.reshape(1, len(output)).T), axis=1)
    labels = np.concatenate((labels, labels_blue.reshape(1, len(labels)).T), axis=1)

    # def dot_product(a, b):
    #     # return np.sum(np.multiply(a, b))
    #     return np.diag(np.dot(a, b.T))
    #
    # numerator = dot_product(output, labels)
    # output_norm = np.sqrt(dot_product(output, output))
    # labels_norm = np.sqrt(dot_product(labels, labels))
    # denominator = np.multiply(output_norm, labels_norm)
    # print(output.shape)
    # temp = np.dstack((output, labels))
    # print(temp.shape, numerator.shape, output_norm.shape, denominator.shape)
    # temp = np.dstack((temp, numerator, output_norm, labels_norm, denominator))
    # print(temp)
    # return np.arccos(numerator / denominator)

    angular_errors = []
    for i in range(len(output)):
        num = np.dot(output[i], labels[i])
        # print(output[i], labels[i])
        # print(np.dot(output[i], output[i]))
        denum = math.sqrt(np.dot(output[i], output[i])) * \
                math.sqrt(np.dot(labels[i], labels[i]))
        angular_error = math.acos(min(num / denum, 1))
        # print(num, denum, angular_error)
        angular_errors.append(angular_error)
    return np.mean(np.array(angular_errors))
