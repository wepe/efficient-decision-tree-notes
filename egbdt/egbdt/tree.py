from tree_node import TreeNode
import Queue
from multiprocessing import Pool
import numpy as np


class Tree(object):
    def __init__(self,
                 min_sample_split,
                 min_child_weight,
                 max_depth,
                 colsample,
                 rowsample,
                 reg_lambda,
                 gamma,
                 num_thread):
        self.root = None
        self.min_sample_split = min_sample_split
        self.min_child_weight = min_child_weight
        self.max_depth = max_depth
        self.colsample = colsample
        self.rowsample = rowsample
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.num_thread = num_thread
        self.feature_importance = {}
        self.alive_nodes = Queue.Queue()

    def calculate_leaf_score(self, G, H):
        """
        According to xgboost, the leaf score is : - G / (H+lambda)
        """
        return - G / (H + self.reg_lambda)

    def calculate_split_gain(self, G_left, H_left, G_total, H_total):
        """
        According to xgboost, the scoring function is:
          gain = 0.5 * (GL^2/(HL+lambda) + GR^2/(HR+lambda) - (GL+GR)^2/(HL+HR+lambda)) - gamma

        this gain is the loss reduction, We want it to be as large as possible.

        G_nan, H_nan from NAN faeture value data, if nan_direction==0, they go to the left child.
        """
        G_right = G_total - G_left
        H_right = H_total - H_left

        gain = 0.5 * (G_left**2/(H_left+self.reg_lambda)
                      + G_right**2/(H_right+self.reg_lambda)
                      - G_total**2/(H_total+self.reg_lambda)) - self.gamma
        return gain

    def build(self, attribute_list, class_list, col_sampler, bin_structure):
        while not self.alive_nodes.empty():
            for col in col_sampler.col_selected:
                # linear scan this column's attribute list, bin by bin
                col_attribute_list = attribute_list[col]
                col_attribute_list_cutting_index = attribute_list.attribute_list_cutting_index[col]

                for uint8_threshold in range(len(col_attribute_list_cutting_index)-1):
                    start_ind = col_attribute_list_cutting_index[uint8_threshold]
                    end_ind = col_attribute_list_cutting_index[uint8_threshold+1]
                    inds = col_attribute_list["index"][start_ind:end_ind]
                    tree_node_G_H = class_list.statistic_given_inds(inds)

                    for tree_node in tree_node_G_H.keys():
                        G, H = tree_node_G_H[tree_node]
                        G_left, H_left = tree_node.get_Gleft_Hleft(col, G, H)
                        G_total, H_total = tree_node.Grad, tree_node.Hess
                        gain = self.calculate_split_gain(G_left, H_left, G_total, H_total)
                        tree_node.update_best_gain(col, uint8_threshold, bin_structure[col][uint8_threshold], gain)

            # once scan all column, we can get the best (feature,threshold,gain) for each alive tree node
            cur_level_node_size = self.alive_nodes.qsize()
            new_tree_nodes = []
            for _ in range(cur_level_node_size):
                tree_node = self.alive_nodes.get()
                best_feature, best_uint8_threshold, best_threshold, best_gain = tree_node.get_best_feature_threshold_gain()
                if best_gain > 0:
                    nan_direction = 0  # TODO
                    left_child = TreeNode(depth=tree_node.depth+1)
                    right_child = TreeNode(depth=tree_node.depth+1)
                    tree_node.internal_node_setter(self, best_feature, best_uint8_threshold, best_threshold, nan_direction, left_child, right_child)

                    # update class_list.corresponding_tree_node
                    # TODO: can update class list one pass? current implementation is num_treenode pass
                    left_inds = attribute_list[best_feature]["index"][0:attribute_list.attribute_list_cutting_index[best_feature][best_uint8_threshold+1]]
                    class_list.update_corresponding_tree_node(tree_node, left_inds)

                    new_tree_nodes.append(left_child)
                    new_tree_nodes.append(right_child)

                else:
                    leaf_score = self.calculate_leaf_score(tree_node.Grad, tree_node.Hess)
                    tree_node.leaf_node_setter(leaf_score)

            # update histogram(Grad,Hess,num_sample) for each alive(new) tree node
            class_list.update_histogram_for_tree_node()

            # process the new tree nodes
            # satisfy max_depth? min_child_weight? min_sample_split?
            # if yes, it is leaf node, calculate its leaf score
            # if no, put into self.alive_node
            while len(new_tree_nodes) != 0:
                tree_node = new_tree_nodes.pop()
                if tree_node.depth >= self.max_depth \
                        or tree_node.Hess < self.min_child_weight \
                        or tree_node.num_sample <= self.min_sample_split:
                    tree_node.leaf_node_setter(self.calculate_leaf_score(tree_node.Grad, tree_node.Hess))
                else:
                    self.alive_nodes.put(tree_node)

        # when finish building this tree, update the class_list.pred, grad, hess
        # class_list.update_pred()

    def fit(self, attribute_list, class_list, row_sampler, col_sampler, bin_structure):
        # when we start to fit a tree, we first conduct row and column sampling
        col_sampler.shuffle()
        row_sampler.shuffle()
        class_list.sampling(row_sampler.row_mask)

        # then we create the root node, initialize histogram(Gradient sum and Hessian sum)
        root_node = TreeNode(depth=1)
        root_node.Grad_setter(class_list.grad.sum())
        root_node.Hess_setter(class_list.hess.sum())
        self.root = root_node

        # put it into the alive_node, and fill the class_list, all data are assigned to root node initially
        self.alive_nodes.put(root_node)
        for i in range(class_list.dataset_size):
            class_list.corresponding_tree_node[i] = root_node

        # then build the tree util there is no alive tree_node to split
        self.build(attribute_list, class_list, col_sampler, bin_structure)
        self.clean_up()

    def _predict(self, feature):
        """
        :param feature: feature of a single sample
        :return:
        """
        cur_tree_node = self.root
        while not cur_tree_node.is_leaf:
            if feature[cur_tree_node.split_feature] <= cur_tree_node.split_threshold:
                cur_tree_node = cur_tree_node.left_child
            else:
                cur_tree_node = cur_tree_node.right_child
        return cur_tree_node.leaf_score

    def predict(self, features):
        pool = Pool()
        preds = pool.map(self._predict, features)
        pool.close()
        return np.array(preds)

    def clean_up(self):
        del self.alive_nodes, self.min_sample_split, self.min_child_weight, self.rowsample,\
            self.colsample, self.max_depth, self.reg_lambda, self.gamma