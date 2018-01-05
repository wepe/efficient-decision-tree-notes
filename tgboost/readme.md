- TGBoost是一个纯Python实现的GBDT工具包，借鉴了XGBoost（主要是打分函数和增益计算方式）和SLIQ(决策树的高效构建方式)，也借鉴了histogram的思想，在训练开始前对原始数据进行分箱．当然，决策树构建过程是基于SLIQ的pre-sort方式，维护attribute list和class list，level-wise地构建决策树．

- TGBoost支持大多数XGBoost支持的特性

- TGBoost借助auto_grad，实现了损失函数自动求导

- TGBoost在处理缺失值时，不仅会尝试将有缺失值的样本分到左右分支(XGBoost采用的），还尝试了分到第三个分支．比较这三种切分方法，选择增益最大的那种方式．因此，TGBoost的决策树是三叉树.


### TODO
- 完善更多特性，scale_pos_weight，feature_importance，cv等

- 由于python多线程不能真正并行，所以只能用多进程．但多进程不共享内存空间，造成在并行处理attribute list时，不能共享同一份attribute list和class list，需要给每个subprocess复制一份，造成了内存消耗，以及速度降低．由于attribute list和class list里面有很复杂的数据结构类型，很难改写支持多进程共享的．所以这一点目前暂不知道如何解决．或许用C++或者Java，用多线程会更好

- 增加更多测试用例，验证缺失值的处理方式的优势
