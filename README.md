# efficient-decision-tree-notes

这个笔记记录高效的决策树系列算法，主要阅读论文，结合一些开源框架，希望在弄清算法的基础上，尝试着改进算法，尝试着工程实现．

我们知道，目前比较流行的两个GBDT开源框架是XGBoost和LightGBM，无论内存占用还是计算速度，它们都做到了淋漓尽致．在LightGBM的[Feature](http://lightgbm.readthedocs.io/en/latest/Features.html)上提到了，XGBoost的decision tree用的是pre-sorted based的算法，也就是在tree building之前对各维特征先排序，代表性的算法是SLIQ[1]和SPRINT[2]．而LightGBM的decision tree是histogram based的算法，也就是先将特征离散化，代表性的算法是CLOUDS[3],Mcrank[4]和Machado[5]．

SLIQ和SPRINT算法的特点决定了树生长的方式是level-wise(breadth-first)的，与之对应的是leaf-wise(depth-wise，best-wise[6])的方式，LightGBM正是采用leaf-wise的方式．


