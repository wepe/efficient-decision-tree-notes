import numpy as np
from egbdt.bining import BinStructure
from egbdt.attribute_list import AttributeList


features = np.random.randint(1000, size=(1000000, 1))
print features.nbytes / 1
feature_dim = features.shape[1]

bin_structure = BinStructure(features)
attribute_list = AttributeList(features, bin_structure)

for i in range(feature_dim):
    print attribute_list[i]
    print attribute_list[i]["index"]
    print type(attribute_list[i])
    print attribute_list.attribute_list_cutting_index[i]
