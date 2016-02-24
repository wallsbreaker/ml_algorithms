__author__ = 'tao'

import Node

class BinaryNode(Node):
    def set_left_child(self, node):
        if self.is_leaf:
            raise ValueError("This node is still a leaf, please set leaf false first")
        if not self.child_nodes:
            self.child_nodes = {0:node, 1:None}
        else:
            self.child_nodes[0] = node

    def set_right_child(self, node):
        if self.is_leaf:
            raise ValueError("This node is still a leaf, please set leaf false first")
        if not self.child_nodes:
            self.child_nodes = {1:node, 0:None}
        else:
            self.child_nodes[1] = node

    def get_left_child(self):
        if self.child_nodes:
            return self.child_nodes[0]
        else:
            return None

    def get_right_child(self):
        if self.child_nodes:
            return self.child_nodes[1]
        else:
            return None