- LightGBM采用histogram based决策树，对原始数据分箱后，只需存储 #data * #feature * 1 byte，因为256个bin一般效果就够好了，只需要8bit即1byte表示原始的浮点类型数据

- 寻找分割点时，LightGBM只需要遍历一遍数据，O(#data * #data)，每遍历一个特征时，就统计每个bin的Gradient 和Hessian的和，遍历完这个特征后，可以得到统计信息[G1,H1;G2,H2....Gn,Hn]，据这个能得到这个特征的最佳切分．因为LightGBM中每个树节点保留自己的数据，所以相当于遍历了一遍原始数据，即可对所有当前alive节点进行分裂．pre-sort的算法，采用level-wise建树，也只需遍历一遍原始数据

- 数据分割．histogram　based算法是O(#data)．基于pre-sort的算法由于采用level-wise，共用一个class list，实际上无需分割，只要更新class list中的对应树节点信息即可．但是由于pre-sort类型的算法，每列特征各自排序，顺序不一致，这种随机访问会造成cache miss


- 计算增益次数，pre-sort的算法，需要精确切分所有切分点，相应的增益计算次数较多．而histogram based的算法，只需要计算#bin次．

- LightGBM应用了直方图做差．每个节点分裂后，只需要统计一个节点的直方图，通过做差得到另一个的e

- 分布式时, histogram based的通信量更低．所以分布式版本的XGBoost也用histogram分箱

- LightGBM两个创新．GOSS，保留梯度绝对值大的样本，抽样梯度绝对值小的；　EFB，数据一般是稀疏的，很多特征是exclusive的，可以组成一个＂大特征＂，这样可以降低构建直返图时需遍历的特征数量．两个措施，分别减少样本和特征