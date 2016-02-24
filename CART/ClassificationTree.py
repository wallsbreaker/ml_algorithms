#!-*- coding:utf-8 -*-
__author__ = 'tao'

import sys

from DataStruct import DecisionTree
from DataStruct import BinaryNode

class ClassificationTree(object):
    def __init__(self, gini_sentry, sample_num_sentry):
        self.gini_sentry = gini_sentry
        self.sample_num_sentry = sample_num_sentry

    def train(self, train_data_x, train_data_y):
        self.train_data_x = train_data_x
        self.train_data_y = train_data_y

        if len(train_data_x) < 1 or not len(train_data_y) == len(train_data_x):
            raise ValueError("train_data_x must be not empty and as same size as train_data_y")

        #生成
        decision_tree = self.construct_tree()
        #剪枝
        decision_trees = self.trim(decision_tree)
        return decision_trees

    def construct_tree(self):
        tree = DecisionTree.DecisionTree()
        node = self.construct_node(self.train_data_x, self.train_data_y)
        tree.set_root(node)

        return tree

    #自下而上剪枝直至剪到最后一个根节点并记录这过程中的
    #所有有效决策树，用交叉验证集来选择最优的决策树
    def trim(self, tree):
        result = [tree.deep_copy()]
        while not tree.get_root().is_leaf():
            non_leaf_nodes = tree.get_non_leaf_nodes()
            min_g = sys.maxint
            cut_node = None
            for node in non_leaf_nodes:
                tree = DecisionTree.DecisionTree(node)
                leaf_size = len(tree.get_leaves())
                loss_of_tree = self.loss_of_tree(tree)
                node.set_leaf(True)
                loss_of_node = self.loss_of_tree(tree)
                node.set_leaf(False)

                g = (loss_of_node-loss_of_tree+0.0) / (leaf_size-1)
                if g < min_g:
                    cut_node = node
            cut_node.set_leaf(True)
            result.append(tree.deep_copy())

    def construct_node(self, train_data_x, train_data_y):
        node = BinaryNode.BinaryNode()

        #如果实例都属于同一类或剩余样本点的个数太少，则叶子节点
        if len(set(train_data_y)) == 1 or len(train_data_x) < self.sample_num_sentry:
            node.set_leaf()
        else:
            #该节点选择的特征
            best_feature_ix, best_feature_value, gini = self.select_feature(train_data_x, train_data_y)
            if gini < self.gini_sentry:
                node.set_leaf()
            else:

                left_data_x, left_data_y, right_data_x, right_data_y = self._divide_by_feature(train_data_x, train_data_y,
                                                                                        best_feature_ix, best_feature_value)

                left_node = self.construct_node(left_data_x, left_data_y)
                right_node = self.construct_node(right_data_x, right_data_y)
                node.set_left_child(left_node)
                node.set_right_child(right_node)

        node.set_train_data(train_data_x)
        node.set_train_label(train_data_y)
        return node

    #根据一个特征进行划分
    def _divide_by_feature(self, train_data_x, train_data_y, feature_ix, best_feature_value):
        left_data_x = []
        left_data_y = []
        right_data_x = []
        right_data_y = []
        for ix in range(len(train_data_y)):
            if train_data_x[ix][feature_ix] <= best_feature_value:
                left_data_x.append(train_data_x[ix])
                left_data_y.append(train_data_y[ix])
            else:
                right_data_x.append(train_data_x[ix])
                right_data_y.append(train_data_y[ix])

        return left_data_x, left_data_y, right_data_x, right_data_y

    #根据平方误差最小化进行特征选择
    def select_feature(self, train_data_x, train_data_y):
        min_feature = -1
        min_feature_value = -1
        min_gini = sys.maxint
        for ix in xrange(len(train_data_x[0])):
            feature_candidates = list(set([x[ix] for x in train_data_x]))
            for feature_value in feature_candidates:
                left_x, left_y, right_x, right_y = self._divide_by_feature(train_data_x, train_data_y, ix, feature_value)

                sub_left_x, sub_left_y = self._divide_by_label(left_x, left_y)
                left_gini = self.gini(sub_left_y)
                sub_right_x, sub_right_y = self._divide_by_label(right_x, right_y)
                right_gini = self.gini(sub_right_y)
                tmp_gini = (left_gini * len(left_y) + right_gini * len(right_y)) / len(train_data_y)
                if tmp_gini < min_gini:
                    min_feature = ix
                    min_feature_value = feature_value

        return min_feature, min_feature_value, min_gini

    #根据一个特征进行划分
    def _divide_by_feature(self, train_data, train_data_label, feature_ix, feature_value):
        left_x = []
        left_y = []
        right_x = []
        right_y = []
        for ix in xrange(len(train_data_label)):
            if train_data[ix][feature_ix] == feature_value:
                left_x.append(train_data[ix])
                left_y.append(train_data_label[ix])
            else:
                right_x.append(train_data[ix])
                right_y.append(train_data_label[ix])

        return left_x, left_y, right_x, right_y

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

    def gini(self, sub_y):
        length = sum(len(x) for x in sub_y)
        result = 1.0
        for item in sub_y:
            result -= (len(item)/length) ** 2
        return result

    def loss_by_split(self, train_data_x, train_data_y, ix, feature_value):
        left_data_x, left_data_y, right_data_x, right_data_y = self._divide_by_feature(train_data_x, train_data_y, ix, feature_value)

        left_mean = (sum(left_data_y) + 0.0) / len(left_data_y)
        left_sum = sum([(x-left_mean)**2 for x in left_data_y])
        right_mean = (sum(right_data_y) + 0.0) / len(right_data_y)
        right_sum = sum([(x-right_mean)**2 for x in right_data_y])

        return left_sum + right_sum

    def loss_of_tree(self, tree):
        leaves = tree.get_leaves()
        result = 0.0
        for node in leaves:
            y = node.get_train_label()
            avg = (sum(y) + 0.0) / len(y)
            result += sum([(x-avg)**2 for x in avg])
        return result

