#!-*- coding:utf-8 -*-
__author__ = 'tao'

class Node(object):
    def __init__(self, ix=-1, values=[], child_nodes={}):
        #ix 该节点的特征
        #values list[list]
        self.ix = ix
        self.values = values
        self.child_nodes = child_nodes

        self.is_leaf = False
        self.label = None

    def is_leaf(self):
        return self.is_leaf

    def set_leaf(self):
        self.is_leaf = True

    def get_label(self):
        return self.label

    def set_label(self, label):
        self.label = label

    def get_child_node(self, data):
        value = data[self.ix]
        for ix in range(len(self.values)):
            if value in self.values[ix]:
                #TODO:对于训练集没有的情况怎么办？
                return self.child_nodes[ix]

    def add_node(self, ix, node):
        self.child_nodes[ix] = node

class DecisionTree(object):
    def __init__(self, values):
        self.values = values

        self.root = None

    def set_root(self, node):
        self.root = node

    def get_label(self, data):
        node = self.root
        while not node.is_leaf():
            node = node.get_child_node(data)
        return node.get_label()
