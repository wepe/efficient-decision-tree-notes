# design a ClassList,store (label,leaf node,prediction,grad,hess) ordered by data index

import numpy as np


class ClassList(object):
    def __init__(self, label):
        self.dataset_size = label.shape[0]
        self.label = label
        self.corresponding_tree_node = [None for _ in range(self.dataset_size)]
        self.pred = np.empty(label.shape, dtype="float32")
        self.grad = np.empty(label.shape, dtype="float32")
        self.hess = np.empty(label.shape, dtype="float32")

    def sampling(self, row_mask):
        self.grad *= row_mask
        self.hess *= row_mask

    def statistic_given_inds(self, inds):
        # scan the given the index, calculate each alive tree node's (G,H)
        ret = {}
        for i in inds:
            tree_node = self.corresponding_tree_node[i]
            if tree_node.is_leaf:
                continue
            else:
                if tree_node not in ret:
                    ret[tree_node] = [0., 0.]
                ret[tree_node][0] += self.grad[i]
                ret[tree_node][1] += self.hess[i]
        return ret

    def update_corresponding_tree_node(self, tree_node, left_inds):
        # scan the class list, if the data fall into tree_node
        # then we see whether its index is in left_inds or right_inds, update the corresponding tree node
        for i in range(self.dataset_size):
            if self.corresponding_tree_node[i] is tree_node:
                if i in left_inds:
                    self.corresponding_tree_node[i] = tree_node.left_child
                else:
                    self.corresponding_tree_node[i] = tree_node.right_child

    def update_histogram_for_tree_node(self):
        # scan the class list
        # update histogram(Grad,Hess,num_sample) for each alive(new) tree node
        for i in range(self.dataset_size):
            if not self.corresponding_tree_node[i].is_leaf:
                self.corresponding_tree_node[i].Grad_add(self.grad[i])
                self.corresponding_tree_node[i].Hess_add(self.hess[i])
                self.corresponding_tree_node[i].num_sample_add(1)
