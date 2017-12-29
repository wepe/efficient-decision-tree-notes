## 基于特征预排序的算法SLIQ

SLIQ是论文"SLIQ: A Fast Scalable Classifier for Data Mining"提出的Scalable的决策树，创新点主要包括三点：特征pre-sort，breadth-first的tree building方式，以及剪枝算法．这些特点使得SLIQ可以处理disk-resident(常驻磁盘)的大数据集．

在处理大数据集时，论文[1]提出在每个树节点分裂前先对样本抽样，并将数值也在离散化．论文[2]提出将数据分块，分别训练决策树，最后组合预测结果．这类方法虽然可以减小训练时间，但是损失了精度．SLIQ可以做到在不损失精度的情况下显著减少训练时间．



### 决策树构建过程

决策树构建过程主要包括tree-building 树生长，tree-pruning树剪枝．下图显示了递归方式的树生长过程．

[Selection_003.png]()

对于树剪枝，一般将数据分为两部分，一部分训练，一部分作为验证，用于剪枝，寻找最小验证集误差的树结构．


### 可扩展性问题(Scalability Issues）

决策树训练过程中最耗时的步骤是split finding，即为树节点寻找最佳分裂特征和特征的阈值(feature, threshold)．所谓最佳就是分裂后gain最大，有多种评价gain的指标: 信息增益，信息增益比，Gini index等，非本文重点，不展开．

对于数值型特征来说，在节点分裂前，需要预先对每维特征排序，然后扫一遍特征，找出每维特征的最佳阈值，再比较得出所有特征中的最佳特征的最佳阈值．假设一个节点有n个样本，每个样本k维特征，那么split　finding的复杂度取决于排序过程，为O(knlogn)．如果在树生长过程中，每个树节点都需要重新排序，那么可想而知会非常慢．针对这个问题，SLIQ提出了特征pre-sorting，特征只需要在训练开始前排序一遍，之后反复使用．

对于类别型特征来说，需要找到最佳的特征子集，举个例子，假如某个类别特征有n种取值，则一共有2＾n个子集，指数的复杂度．


### SLIQ算法

#### 数值型特征预排序和宽度优先的树生长策略（Pre-Sorting and Breadth-First Growth）

- 数据结构attribute list和class list的设计

SLIQ设计了attribute list和class list两种数据结构来实现pre-sorting，其结构如下图所示：
[Selection_004.png]()

图中训练数据有6个样本，age和salary是属性(特征)，class是类别．每个特征对应一个attribute list,　该结构包括两列，一列是属性的值，按从小到大排好了序，另一列是样本对应的行索引.　　class list结构按行索引顺序存储了所有样本，同样包括两列，一列存储了样本的类别，另一列存储当前该样本所属的叶节点．SLIQ算法假设class list可以常驻内存(memory-resident)，因为在SLIQ算法中class list被频繁地读写．当内存不够大时，attribute list可以存储在磁盘．所以SLIQ支持out-of-core　learning.

- split finding过程

我们接着看split finding的过程，由于有上面两个list结构，只需要扫一遍数据，就可以找到同一层每个树节点的最佳分裂特征和特征阈值（这也是为什么SLIQ是breadth-first/level-wise的），下图是算法流程:
[Selection_005.png]()

再通过一个例子加深理解：
[Selection_006.png]()

首先，每个叶节点都会维护一个直方图，本例子中有两种类别G和B，所以根据叶节点上每个样本所属类别就可以统计出该叶节点的GB直方图．在开始扫描salary　list前，这个直方图就已经构建好了，而且初始时，split threshold是-inf，所有样本分到右子节点，所以L对应的直方图G,B取值都是0.　

接着，第一次扫描到(salary 15, index 2)，找到class list中第二行(B,N2)，于是叶节点N2的split threshold变成15, index 2的样本被分到N2的左子节点，直方图也因此更新（见上图步骤１），根据更新后的直方图可以计算这个split threshold对应的增益．

第二次扫描到(salary 40, index 4)，找到class list中第四行(B,N3)，于是叶节点N3的split threshold变成40, index 4的样本被分到N3的左子节点，直方图也因此更新（见上图步骤２），根据更新后的直方图可以计算这个split threshold对应的增益．

这样，当扫描完salary属性的list后，每个叶子节点的salary属性的最佳切分阈值就得到了．当扫描完所有属性的list后，每个叶子节点的所有属性的最佳切分阈值就得到了，也就是说，只需要扫一遍数据，就可以得到每个叶子节点的最佳特征和最佳阈值．

- 更新class list

找到每个叶节点的最佳特征和最佳阈值后，下一步要对节点进行分裂，以及更新class list．该过程如下图所示:

[Selection_007.png]()

在上一个步骤split finding中，我们知道了属性salary被选中为最佳特征(可能被多个叶节点选中，如本例N2和N3都选中了salary)，所以我们需要再扫描一遍salary，如上图所示，当扫描到第二个样本(40,4)的时候，我们找到class list第４行，知道它在节点N3，而N3的分裂阈值是50，所以该样本被分到N3的左子节点N6，更新class list的leaf 列．当扫描完所有被选中的特征后，class list就更新完毕了，当class list更新完毕后，我们需要扫描一遍class list，构建每个新叶子节点的直方图．

- 进一步优化

在树生长过程中，有些叶节点会变成＂纯＂节点，或者满足了停止分裂条件，那么落入这些节点的样本将不会再被使用，所以将这些样本从attribute list中删除掉，可以减少后续遍历attribute list的时间．


#### 类别型特征的子集搜索方法

假设一个类别型特征有n种取值，那么一共就有2＾n种子集．当n＜＝10时，SLIQ暴力枚举每种分裂方法．当n＞10时，SLIQ采用贪心算法，维护空集S，每次从n种值里面选１个最佳的加入S，直到没有增益为止．


#### 树剪枝

SLIQ的剪枝算法基于MDL原则(Minimum Description Length )，本文略去这部分．


### SLIQ算法的并行化

假设有N个计算节点，那么针对attribute list，可以将其平分为N部分，每个计算节点处理1/N的属性．但是对于class list，则要复杂一些，有两种并行化方法：

第一种，replicate，每个计算节点都复制一份完整的class list，那么在更新class list时，每个计算节点都要与其它计算节点通信，更新其它计算节点的class list．

第二种，partition，将class list切分为N份，每个计算节点维护一份．这样一来，每个计算节点上有(N-1)/N的样本需要经过通信获取其对应的class list信息，这个通信量相比第一种方案更大．

###　参考文献

[1] J. Catlett. Megainduction: Machine Learning on Very Large Databases. PhD thesis,University of Sydney, 
[2] P. K. Chan and S. J. StoIfo. Meta-learning for multistrategy and parallel learning. In Proc. Second Intl. Workshop on Mu&strategy Learning, pages 150-165, 1993. 
