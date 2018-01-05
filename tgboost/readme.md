- TGBoost是一个纯Python实现的GBDT工具包，借鉴了XGBoost（主要是打分函数和增益计算方式）和SLIQ(决策树的高效构建方式)，也借鉴了histogram的思想，在训练开始前对原始数据进行分箱．当然，决策树构建过程是基于SLIQ的pre-sort方式，维护attribute list和class list，level-wise地构建决策树．

- TGBoost支持大多数XGBoost支持的特性

- TGBoost借助auto_grad，实现了损失函数自动求导

- TGBoost在处理缺失值时，不仅会尝试将有缺失值的样本分到左右分支(XGBoost采用的），还尝试了分到第三个分支．比较这三种切分方法，选择增益最大的那种方式．因此，TGBoost的决策树是三叉树.
