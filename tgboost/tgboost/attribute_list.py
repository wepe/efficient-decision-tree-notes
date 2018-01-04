# design a AttributeList, store (sorted feature, index) for each feature
import numpy as np


class AttributeList(object):
    def __init__(self, features, bin_structure):
        self.features = features
        self.feature_dim = features.shape[1]
        self.dataset_size = features.shape[0]

        self.feature_missing_cnt = self.missing_value_count()
        self.missing_value_attribute_list = [np.empty((self.feature_missing_cnt[i], ), dtype="int32") for i in range(self.feature_dim)]

        self.attribute_list = [np.empty((self.dataset_size-self.feature_missing_cnt[i],),
                                        dtype=[("attribute", "uint8"), ("index", "int32")]) for i in range(self.feature_dim)]
        self.attribute_list_cutting_index = [[] for _ in range(self.feature_dim)]

        self.bin_structure = bin_structure
        self.construct_attribute_list()

    def missing_value_count(self):
        # for each feature, how many missing value are there
        ret = []
        for i in range(self.feature_dim):
            ret.append(np.isnan(self.features[:, i]).sum())
        return ret

    def construct_attribute_list(self):
        for i in range(self.feature_dim):
            # scan each feature, see each example fall into which bin
            bin_upper_bounder = sorted(self.bin_structure[i].values())
            bin_upper_bounder[-1] = np.inf
            bin_number = len(bin_upper_bounder)
            bin_example_index = [[] for _ in range(bin_number)]
            missing_value_index = []

            for index in range(self.dataset_size):
                value = self.features[index, i]

                if np.isnan(value):
                    missing_value_index.append(index)
                    break

                # binary search
                low, high = 0, bin_number-1
                while low < high:
                    mid = (low + high) // 2
                    if value < bin_upper_bounder[mid]:
                        high = mid
                    elif value > bin_upper_bounder[mid]:
                        low = mid + 1
                    else:
                        bin_example_index[mid].append(index)
                        break
                if low == high:
                    bin_example_index[low].append(index)

            # fill the attribute list for this feature
            acc_cnt = 0
            self.attribute_list_cutting_index[i].append(acc_cnt)
            for k in range(bin_number):
                inds = bin_example_index[k]
                self.attribute_list[i]["index"][acc_cnt:(acc_cnt+len(inds))] = inds
                self.attribute_list[i]["attribute"][acc_cnt:(acc_cnt+len(inds))] = [k for _ in range(len(inds))]
                acc_cnt += len(inds)
                self.attribute_list_cutting_index[i].append(acc_cnt)

            # fill the missing value attribute list for this feature
            self.missing_value_attribute_list[i] = missing_value_index

        self.clean_up()

    def update_grad_hess_missing_for_tree_node(self, class_list):
        # scan all missing_value_attribute_list
        for col in range(self.feature_dim):
            all_missing_inds = self.missing_value_attribute_list[col]
            for i in all_missing_inds:
                tree_node = class_list.corresponding_tree_node[i]
                if not tree_node.is_leaf:
                    tree_node.Grad_missing[col] += class_list.grad[i]
                    tree_node.Hess_missing[col] += class_list.hess[i]

    def __getitem__(self, item):
        return self.attribute_list[item]

    def clean_up(self):
        # clear not necessary instance attribute
        del self.features, self.bin_structure, self.dataset_size

#TODO: parallel construct_attribute_list
#TODO: support nan value