## 基于特征预排序的算法SPRINT

SPRINT是SLIQ算法的改进版．我们知道，SLIQ算法需要频繁地查询和更新class list，所以要求class list常驻内存．但是对于大数据集来说，class list的大小可能会超出内存，这给SLIQ算法带来了局限．SPRINT算法因此被提出，它同样采用了特征预排序的方法，但是设计了不同的数据结构，摒弃了class list，重新设计了attribute list，使得所有数据不需要常驻内存．


### 数据结构的设计

#### Attribute list

SPRINT采用的attribute list结构如下图所示，包括三列，第一列是属性的值，按从小到大排好了序，第二列是样本的类别，第三列是样本的索引 rid．当叶节点分裂时，每个attribute list也被切分为两个，分配给每个子节点．并且切分后的attribute list上的特征的值仍然是有序的．

[Selection_008.png]()


#### split finding过程

- 数值型特征

对于数值型特征，split　finding的过程跟SLIQ一致．如下图所示，扫一遍age list，边扫边更新直方图，同时计算增益，就可以得到age属性的最佳切分阈值.　只不过在SPRINT里，每个叶节点维护自己的attribute list，得到的也就只是这个叶节点的最佳切分阈值．

[Selection_009.png]()

- 类别型特征

对类别型特征，在扫一遍attribute list后，构建出count matrix，根据count matrix再去找最佳子集．

[Selection_010.png]()


#### 节点分裂过程

根据找到的最佳特征和最佳切分阈值，我们需要对节点进行分裂，同时切分attribute list．仍以下图为例:

[Selection_008.png]()

节点0选中了age属性，阈值27.5，那么对于age list，可以直接切分为两部分，分别分配给节点１和２．而对于car type list的切分就没那么直接了．我们需要在切分age list时，用哈希表记录(rid,leaf_node)，即记录样本的索引及其对应的叶节点．根据这个哈希表我们就可以对car type list进行切分了．当然，这个哈希表无需记录所有样本的索引及其叶节点，只需要记录两个子节点中分配到的样本较少的那个．

在对attibute list切分的过程中，可以同时统计两个子节点的直方图

