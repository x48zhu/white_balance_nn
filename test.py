import math
import numpy as np
import unittest

from utils import angular_error_scalar


class TestMethods(unittest.TestCase):
    def test_angular_error_calc(self):
        a = np.array([[0.1, 0.2], [0.1, 0.2]])
        b = np.array([[0.1, 0.2], [0.4, 0.5]])

        correct_result = math.acos(
            (0.4 * 0.1 + 0.5 * 0.2 + 0.1 * 0.7) / (
                math.sqrt(0.4 * 0.4 + 0.5 * 0.5 + 0.1 * 0.1) *
                math.sqrt(0.1 * 0.1 + 0.2 * 0.2 + 0.7 * 0.7))) / 2
        self.assertEqual(angular_error_scalar(a, b), correct_result)
