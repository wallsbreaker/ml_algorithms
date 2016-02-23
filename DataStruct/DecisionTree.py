#!-*- coding:utf-8 -*-
__author__ = 'tao'

class DecisionTree(object):
    def __init__(self, values):
        self.values = values

        self.root = None

    def set_root(self, node):
        self.root = node

    def get_root(self):
        return self.root

    def get_non_leaf_nodes(self):
        result = []
        if self.root.is_leaf():
            result.append(self.root)
        else:
            nodes = [self.root]
            while nodes:
                node = nodes.pop()
                if not node.is_leaf():
                    children = node.get_child_nodes()
                    nodes.extend(children)

                    result.append(node)
        return result

    def get_leaves(self):
        result = []
        if self.root.is_leaf():
            result.append(self.root)
        else:
            nodes = [self.root]
            while nodes:
                node = nodes.pop()
                children = node.get_child_nodes()
                for child in children:
                    if child.is_leaf():
                        result.append()
                    else:
                        nodes.append(child)
        return result

    def get_label(self, data):
        node = self.root
        while not node.is_leaf():
            node = node.get_child_node(data)
        return node.get_label()
