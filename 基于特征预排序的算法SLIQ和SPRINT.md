## 一．SLIQ算法

SLIQ是论文"SLIQ: A Fast Scalable Classifier for Data Mining"提出的Scalable的决策树，创新点主要包括三点：特征pre-sort，breadth-first的tree building方式，以及剪枝算法．这些特点使得SLIQ可以处理disk-resident(常驻磁盘)的大数据集．

