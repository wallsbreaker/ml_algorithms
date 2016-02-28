#-*- coding:utf-8 -*-
__author__ = 'tao'

from Kernel import LinearKernel
from Kernel import RBFKernel

class SVM(object):
    def __init__(self, kernel_type="linear", C=0.1, tolerance=0.001, max_iter=1000):
        if not kernel_type:
            raise ValueError("kernel_type should not be None")

        if kernel_type.lower() == "linear":
            self.kernel = LinearKernel.LinearKernel()
        elif kernel_type.lower() == "rbf" or kernel_type.lower() == "gaussian":
            self.kernel = RBFKernel.RBFKernel()
        else:
            raise ValueError("just support Linear and RBF kernel now")
        self.C = C
        self.tolerance = tolerance
        self.max_iter = max_iter

    def train(self, data_x, data_y):
        #init by train samples
        self.data_x = data_x
        self.data_y = data_y
        self.data_size = len(data_x)
        self.alpha = [0.0] * self.data_size
        self.E_cache = self._update_E_cache()
        self.b = 0.0

        #第一次时，由于alpha都为0，所以先遍历一遍全部样本
        #以后的话如果在边界上的样本都满足了KKT条件，也同样需要遍历所有样本点
        check_all_sample = True
        not_satisfied_KKT_num = 0
        iter_num = 0
        while iter_num < self.max_iter and (check_all_sample or not_satisfied_KKT_num != 0):
            not_satisfied_KKT_num = 0
            for ix in xrange(self.data_size):
                if check_all_sample or (not check_all_sample and self.alpha[ix] > 0 and self.alpha[ix] < self.C):
                    if not self._is_satisfied_KKT(ix):
                        not_satisfied_KKT_num += 1
                        alpha1 = ix
                        alpha2 = self._select_alpha2(alpha1)
                        self._update(alpha1, alpha2)
                        self._update_E_cache()
                        if self._check_over():
                            return
            if not not_satisfied_KKT_num:
                check_all_sample = True
            else:
                check_all_sample = False

    def _update_E_cache(self):
        self.E_cache = {}
        for ix in xrange(self.data_size):
            if self.alpha[ix] > 0 and self.alpha[ix] < self.C:
                self.E_cache[ix] = self._function_value(ix) - self.data_y[ix]

    def _is_satisfied_KKT(self, ix):
        yg = self.data_y[ix]*self._function_value(ix)
        if self.alpha[ix] < 0 or yg <= 0:
            return False
        if self.alpha[ix]==0 and yg >= 1:
            return True
        elif self.alpha[ix]>0 and self.alpha[ix]<self.C and yg == 1:
            return True
        elif self.alpha[ix]==self.C and yg <= 1:
            return True
        else:
            return False

    def _function_value(self, sample_ix):
        result = 0.0
        for ix in xrange(self.data_size):
            result += self.alpha[ix] * self.data_y[ix] * self.kernel.distance(ix, sample_ix)
        result += self.b
        return result

    #选择|E1-E2|最大的alpha2
    def _select_alpha2(self, alpha1):
        E1 = self.E_cache[alpha1]
        max_value = 0.0
        max_index = -1
        for ix in xrange(self.data_size):
            if self.alpha[ix] > 0 and self.alpha[ix] < self.C:
                tmp = abs(E1 - self.E_cache[ix])
                if max_value < tmp:
                    max_value = tmp
                    max_index = ix
        return max_index

    def _update(self, alpha1, alpha2):
        if alpha1 in self.E_cache:
            E1 = self.E_cache[alpha1]
        else:
            E1 = self._function_value(alpha1) - self.data_y[alpha1]
        if alpha1 in self.E_cache:
            E2 = self.E_cache[alpha2]
        else:
            E2 = self._function_value(alpha2) - self.data_y[alpha2]

        eta = self.kernel.distance(self.data_x[alpha1], self.data_x[alpha1]) + self.kernel.distance(self.data_x[alpha2],
                            self.data_x[alpha2]) - 2 * self.kernel.distance(self.data_x[alpha1], self.data_x[alpha2])
        new_alpha2_uncut_value = self.alpha[alpha2] + self.data_y[alpha2] * (E1-E2) / eta
        if self.data_y[alpha1] == self.data_y[alpha2]:
            low_bound = max(0, self.alpha[alpha2]+self.alpha[alpha1]-self.C)
            up_bound = min(self.C, self.alpha[alpha2]+self.alpha[alpha1])
        else:
            low_bound = max(0, self.alpha[alpha2]-self.alpha[alpha1])
            up_bound = min(self.C, self.C+self.alpha[alpha2]-self.alpha[alpha1])
        if new_alpha2_uncut_value > up_bound:
            new_alpha2_value = up_bound
        elif new_alpha2_uncut_value < low_bound:
            new_alpha2_value = low_bound
        else:
            new_alpha2_value = new_alpha2_uncut_value
        #self.alpha[alpha2] = new_alpha2_value
        new_alpha1_value = self.alpha[alpha1]+self.data_y[alpha1]*self.data_y[alpha2]*(self.alpha[alpha1]-new_alpha2_value)
        #self.alpha[alpha1] = new_alpha1_value

        if new_alpha1_value>0 and new_alpha1_value<self.C:
            self.b = -E1 - self.data_y[alpha1]*self.kernel.distance(self.data_x[alpha1], self.data_x[alpha1])*(
                new_alpha1_value-self.alpha[alpha1]) - self.data_y[alpha2]*self.kernel.distance(self.data_x[alpha2],
                self.data_x[alpha1])*(new_alpha2_value-self.alpha[alpha2]) + self.b
        elif new_alpha2_value>0 and new_alpha2_value<self.C:
            self.b = -E2 - self.data_y[alpha1]*self.kernel.distance(self.data_x[alpha1], self.data_x[alpha2])*(
                new_alpha1_value-self.alpha[alpha1]) - self.data_y[alpha2]*self.kernel.distance(self.data_x[alpha2],
                self.data_x[alpha2])*(new_alpha2_value-self.alpha[alpha2]) + self.b
        else:
            b1= -E1 - self.data_y[alpha1]*self.kernel.distance(self.data_x[alpha1], self.data_x[alpha1])*(
                new_alpha1_value-self.alpha[alpha1]) - self.data_y[alpha2]*self.kernel.distance(self.data_x[alpha2],
                self.data_x[alpha1])*(new_alpha2_value-self.alpha[alpha2]) + self.b
            b2 = -E2 - self.data_y[alpha1]*self.kernel.distance(self.data_x[alpha1], self.data_x[alpha2])*(
                new_alpha1_value-self.alpha[alpha1]) - self.data_y[alpha2]*self.kernel.distance(self.data_x[alpha2],
                self.data_x[alpha2])*(new_alpha2_value-self.alpha[alpha2]) + self.b
            self.b = b1 + (b2-b1)/2

    def _check_over(self):
        for ix in xrange(self.data_y):
            if self.alpha[ix]>0 and self.alpha[ix]<self.C:
                E = self.E_cache[ix]
                if not abs(E) < self.tolerance:
                    return False
        return True