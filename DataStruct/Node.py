__author__ = 'tao'

from collections import Counter

class Node(object):
    def __init__(self, ix=-1, values=[], child_nodes={}):
        #ix 该节点的特征
        #values list[list]
        self.ix = ix
        self.values = values
        self.child_nodes = child_nodes

        self.is_leaf = False
        self.label = None
        self.train_data = None
        self.train_label = None

    def is_leaf(self):
        return self.is_leaf

    def set_leaf(self, flag=True):
        self.is_leaf = flag

    def get_label(self):
        if not self.is_leaf:
            return None
        elif not self.label:
            counter = Counter(self.train_data_label)
            label, num = counter.most_common(1)
            self.label = label

        return self.label

    def get_train_data(self):
        return self.train_data

    def set_train_data(self, train_data):
        self.train_data = train_data

    def get_train_label(self):
        return self.train_label

    def set_train_label(self, train_label):
        self.train_label = train_label

    def get_child_node(self, data):
        if self.is_leaf:
            return None
        else:
            value = data[self.ix]
            for ix in range(len(self.values)):
                if value in self.values[ix]:
                    #TODO:对于训练集没有的情况怎么办？
                    return self.child_nodes[ix]

    def get_child_nodes(self):
        return self.child_nodes

    def add_node(self, ix, node):
        self.child_nodes[ix] = node

