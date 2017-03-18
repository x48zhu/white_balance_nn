import logging
import math
import numpy as np
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def angular_error_scalar(output, labels):
    logger.debug(output, labels)
    def dot_product(a, b):
        return np.sum(np.multiply(a, b))
    numerator = dot_product(output, labels)
    denominator = math.sqrt(dot_product(output, output)) * math.sqrt(dot_product(labels, labels))
    return math.acos(numerator / denominator)