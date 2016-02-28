__author__ = 'tao'

import numpy as np

from Kernel import Kernel


class RBFKernel(Kernel):
    def __init__(self, gamma):
        self.gamma = gamma

    def distance(self, sample1, sample2):
        sample1_array = np.array(sample1)
        sample2_array = np.array(sample2)
        norm = np.linalg.norm(sample1_array-sample2_array, 2)
        dis = np.exp(-norm/(2*self.gamma**2))
        return dis