__author__ = 'tao'

import numpy as np

import Kernel

class PolynomialKernel(Kernel):
    def __init__(self, power):
        self.power = power

    def distance(self, sample1, sample2):
        sample1_array = np.array(sample1)
        sample2_array = np.array(sample2)
        result = np.sum(np.power(np.dot(sample1_array,sample2_array)+1, self.power))
        return result