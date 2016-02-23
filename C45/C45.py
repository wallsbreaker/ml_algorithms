#!-*- coding:utf-8 -*-
__author__ = 'tao'


import math

from DataStruct import DecisionTree
from DataStruct import Node

class C45(object):
    def __init__(self, sentry, alpha):
        """
        :param sentry: 生成过程中信息增益比的阈值，小于的作为叶子节点
        :param alpha: 剪枝过程中代价函数中惩罚项的系数
        :return:
        """
        self.sentry = sentry
        self.alpha = alpha

    def train(self, train_data, train_data_label):
        self.train_data = train_data
        self.train_data_label = train_data_label

        if len(train_data) < 1 or not len(train_data_label) == len(train_data):
            raise ValueError("train_data must be not empty and as same size as train_data_label")

        #生成
        decision_tree = self.construct_tree()
        #剪枝
        self.trim(decision_tree)

        return decision_tree

    def construct_tree(self):
        #分析train_data, 得到特征集合和其各自的取值
        feature_num = len(self.train_data[0])
        values = [set([x[i] for x in self.train_data]) for i in range(feature_num)]

        tree = DecisionTree.DecisionTree(values)
        node = self.construct_node(self.train_data, self.train_data_label, range(feature_num), values)
        tree.set_root(node)

        return tree

    def trim(self, tree):
        while True:
            nodes = tree.get_non_leaf_nodes()
            old_loss = self.loss(tree)
            over_flag = True
            for node in reversed(nodes):
                #如果把该颗子树作为叶节点
                node.set_leaf(False)
                new_loss = self.loss(tree)
                if new_loss < old_loss:
                    over_flag = False
                else:
                    node.set_leaf(True)
            if over_flag:
                break

    def loss(self, tree):
        leaves = tree.get_leaves()
        loss = 0.0
        for leaf in leaves:
            train_data = leaf.get_train_data()
            train_label = leaf.get_train_label()
            sub_data, sub_label = self._divide_by_label(train_data, train_label)
            loss += self.caculate_entropy(sub_data.items())

        loss += self.alpha * len(leaves)
        return loss

    def construct_node(self, train_data, train_data_label, remain_features, values):
        #如果实例都属于同一类，则叶子节点
        if len(set(train_data_label)) == 1:
            node = Node()
            node.set_leaf()
            node.set_train_data(train_data)
            node.set_train_label(train_data_label)
            return node
        #如果所有属性都用过了
        if len(set(remain_features)) == 1 and remain_features[0] == -1:
            node = Node()
            node.set_leaf()
            node.set_train_data(train_data)
            node.set_train_label(train_data_label)
            return node

        #该节点选择的特征
        best_feautre_ix, information_gain_ratio = self.select_feature(train_data, train_data_label, remain_features, values)
        #如果信息增益值小于阈值
        if information_gain_ratio < self.sentry:

            node = Node()
            node.set_leaf()
            node.set_train_data(train_data)
            node.set_train_label(train_data_label)
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

        node = DecisionTree.Node(best_feautre_ix, values[best_feautre_ix], child_nodes)
        node.set_train_data(train_data)
        node.set_train_label(train_data_label)
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

    #根据信息增益比进行特征选择
    def select_feature(self, train_data, train_data_label, remain_features, values):
        partition_data, partition_label = self._divide_by_label(train_data, train_data_label)
        entropy = self.caculate_entropy(partition_data.items())

        best_feature_ix = -1
        max_information_gain_ratio = -1
        for feature_ix in range(len(remain_features)):
            if remain_features[feature_ix] == -1:
                continue
            partition_data, partition_label = self._divide_by_feature(train_data, train_data_label, feature_ix, values[feature_ix])
            all_length = sum([len(x)+0.0 for x in partition_label.items()])
            information_gain = entropy
            for k, v in partition_data.iteritems():
                label = partition_label[k]
                sub_train_data, sub_train_data_label = self._divide_by_label(v, label)

                information_gain -= (len(label) / all_length) * self.caculate_entropy(sub_train_data.items())
            information_gain_ratio = information_gain / entropy
            if max_information_gain_ratio < information_gain_ratio:
                best_feature_ix = feature_ix
                max_information_gain_ratio = information_gain_ratio

        return best_feature_ix, information_gain_ratio