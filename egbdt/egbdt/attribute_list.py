# design a AttributeList, store (sorted feature, index) for each feature
import numpy as np


class AttributeList(object):
    def __init__(self, features, bin_structure):
        self.features = features
        self.feature_dim = features.shape[1]
        self.dataset_size = features.shape[0]
        self.attribute_list = [np.empty((self.dataset_size,), dtype=[("attribute", "uint8"), ("index", "int32")]) for _ in range(self.feature_dim)]
        self.attribute_list_cutting_index = [[] for _ in range(self.feature_dim)]
        self.bin_structure = bin_structure
        self.construct_attribute_list()

    def construct_attribute_list(self):
        for i in range(self.feature_dim):
            # scan each feature, see each example fall into which bin
            bin_upper_bounder = sorted(self.bin_structure[i].values())
            bin_upper_bounder[-1] = np.inf
            bin_number = len(bin_upper_bounder)
            bin_example_index = [[] for _ in range(bin_number)]

            for index in range(self.dataset_size):
                value = self.features[index, i]
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

            # fill the attribute list for that feature
            acc_cnt = 0
            self.attribute_list_cutting_index[i].append(acc_cnt)
            for k in range(bin_number):
                inds = bin_example_index[k]
                self.attribute_list[i]["index"][acc_cnt:(acc_cnt+len(inds))] = inds
                self.attribute_list[i]["attribute"][acc_cnt:(acc_cnt+len(inds))] = [k for _ in range(len(inds))]
                acc_cnt += len(inds)
                self.attribute_list_cutting_index[i].append(acc_cnt)

        self.clean_up()

    def __getitem__(self, item):
        return self.attribute_list[item]

    def clean_up(self):
        # clear not necessary instance attribute
        del self.features, self.bin_structure, self.dataset_size

#TODO: parallel construct_attribute_list
