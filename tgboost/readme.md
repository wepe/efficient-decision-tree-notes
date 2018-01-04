This GBDT implementation is based on XGBoost and SLIQ. It first bining the features. Then build the tree by a level-wise way. The only requirment is Numpy. Support parallel learning on a single machine. 

## TODO

- support handling missing value

- reduce memory consumption, the current implementation use multiprocessing, which copy attribute list, class list to each subprocess
