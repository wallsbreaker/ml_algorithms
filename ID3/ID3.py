#!-*- coding:utf-8 -*-
__author__ = 'tao'

from collections import Counter
import math

from DataStruct import DecisionTree
from DataStruct import Node

class ID3(object):
    def __init__(self, sentry):
        self.sentry = sentry

    def train(self, train_data, train_data_label):
        self.train_data = train_data
        self.train_data_label = train_data_label

        if len(train_data) < 1 or not len(train_data_label) == len(train_data):
            raise ValueError("train_data must be not empty and as same size as train_data_label")

        #生成
        decision_tree = self.construct_tree()

    def construct_tree(self):
        #分析train_data, 得到特征集合和其各自的取值
        feature_num = len(self.train_data[0])
        values = [set([x[i] for x in self.train_data]) for i in range(feature_num)]

        tree = DecisionTree.DecisionTree()
        node = self.construct_node(self.train_data, self.train_data_label, range(feature_num), values)
        tree.set_root(node)

        return tree

    def construct_node(self, train_data, train_data_label, remain_features, values):
        #如果实例都属于同一类，则叶子节点
        if len(set(train_data_label)) == 1:
            node = Node.Node()
            node.set_leaf()
            return node
        #如果所有属性都用过了
        if len(set(remain_features)) == 1 and remain_features[0] == -1:
            counter = Counter(train_data_label)
            label, num = counter.most_common(1)
            node = Node.Node()
            node.set_leaf()
            return node

        #该节点选择的特征
        best_feautre_ix, information_gain = self.select_feature(train_data, train_data_label, remain_features, values)
        #如果信息增益值小于阈值
        if information_gain < self.sentry:
            counter = Counter(train_data_label)
            label, num = counter.most_common(1)
            node = Node.Node()
            node.set_leaf()
            return node
        remain_features[best_feautre_ix] = -1#将其置为已用过的特征

        value = values[best_feautre_ix]
        sub_train_data, sub_train_data_label = self._divide_by_feature(train_data, train_data_label, best_feautre_ix, value)

        child_nodes = {}
        for ix in sub_train_data.keys():
            data = sub_train_data[ix]
            label = sub_train_data_label[ix]
            child_node = self.construct_node(data, label, remain_features, values)
            child_nodes[ix] = child_node

        node = Node.Node(best_feautre_ix, values[best_feautre_ix], child_nodes)
        return node

    #根据一个特征进行划分
    def _divide_by_feature(self, train_data, train_data_label, feature_ix, this_feature_values):
        sub_train_data = {}
        sub_train_data_label = {}
        for ix in range(len(this_feature_values)):
            if ix not in sub_train_data:
                sub_train_data[ix] = []
                sub_train_data_label[ix] = []

            for cx in range(len(train_data)):
                if train_data[cx][feature_ix] in this_feature_values[ix]:
                    sub_train_data[ix].append(train_data[cx])
                    sub_train_data_label[ix].append(train_data_label[cx])

        return sub_train_data, sub_train_data_label

    #根据标签进行划分
    def _divide_by_label(self, train_data, train_data_label):
        keys = set(train_data_label)
        sub_train_data = {}
        sub_train_data_label = {}
        for key in keys:
            if key not in sub_train_data:
                sub_train_data[key] = []
                sub_train_data_label[key] = []

        for ix in range(len(train_data)):
            sub_train_data[train_data_label[ix]].append(train_data[ix])

        for key in keys:
            sub_train_data_label[key] = [key] * len(sub_train_data[key])

        return sub_train_data, sub_train_data_label

    def caculate_entropy(self, train_data):
        every_length = [len(x)+0.0 for x in train_data]
        all_length = sum(every_length)

        result = 0.0
        for length in every_length:
            ratio = length / all_length
            result -= (ratio) * (math.log(ratio, 2))

        return result

    #根据信息增益进行特征选择
    def select_feature(self, train_data, train_data_label, remain_features, values):
        partition_data, partition_label = self._divide_by_label(train_data, train_data_label)
        entropy = self.caculate_entropy(partition_data.items())

        best_feature_ix = -1
        max_information_gain = -1
        for feature_ix in range(len(remain_features)):
            if remain_features[feature_ix] == -1:
                continue
            partition_data, partition_label = self._divide_by_feature(train_data, train_data_label, feature_ix, values[feature_ix])
            all_length = sum([len(x)+0.0 for x in partition_label.items()])
            information_gain = 0.0
            for k, v in partition_data.iteritems():
                label = partition_label[k]
                sub_train_data, sub_train_data_label = self._divide_by_label(v, label)

                information_gain -= (len(label) / all_length) * self.caculate_entropy(sub_train_data.items())
            if max_information_gain < information_gain:
                best_feature_ix = feature_ix
                max_information_gain = information_gain

        return best_feature_ix, max_information_gain