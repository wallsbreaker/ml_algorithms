__author__ = 'tao'

import numpy as np

from Kernel import Kernel


class LinearKernel(Kernel):
    def distance(self, sample1, sample2):
        return np.dot(sample1, sample2)