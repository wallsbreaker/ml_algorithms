#-*- coding:utf-8 -*-
__author__ = 'tao'

import random
import sys

'''
改进型的KMeans，主要改进地方如下：
'''
#TODO 待改进
class Kmeans(object):
    def __init__(self, samples, cluster_num):
        '''
        :param samples:  list of list
        :param cluster_num:
        '''
        self.samples = samples
        self.sample_size = len(samples)
        self.cluster_num = cluster_num
        self.clusters = [-1] * self.sample_size

        self.dimension = 0
        if self.sample_size:
            self.dimension = len(samples[0])

    def clustering(self):
        self.select_seed()

        ever_changed = True
        while ever_changed:
            ever_changed = False
            for sample_ix in xrange(self.sample_size):
                ix = self.nearest(self.samples[sample_ix])
                if self.clusters[sample_ix] != ix:
                    self.clusters[sample_ix] = ix
                    ever_changed = True

            #更新中心点
            each_cluster_sum = [[0.0]*self.dimension for _ in xrange(self.cluster_num)]
            each_cluster_num = [0] * self.cluster_num
            for ix in xrange(self.sample_size):
                for cx in xrange(self.dimension):
                    each_cluster_sum[self.clusters[ix]][cx] = self.samples[ix][cx]
                each_cluster_num[self.clusters[ix]] += 1

        clusters_list = [[] for _ in xrange(self.cluster_num)]
        for ix in xrange(self.sample_size):
            clusters_list[self.clusters[ix]].append(self.samples[ix])

        return clusters_list

    #选择种子，同时将种子留在原来的里面
    def select_seed(self):
        self.seeds = []
        for ix in xrange(self.cluster_num):
            seed = random.choice(self.samples)
            while seed in self.seeds:
                seed = random.choice(self.samples)
            self.seeds.append(seed[:])

    #欧氏距离
    def nearest(self, sample):
        min_distance = sys.maxint
        min_distance_ix = 0
        for ix in xrange(self.cluster_num):
            distance = sum([(sample[cx]-self.seeds[ix][cx])**2 for cx in xrange(self.dimension)])
            if distance < min_distance:
                min_distance = distance
                min_distance_ix = ix
        return min_distance_ix