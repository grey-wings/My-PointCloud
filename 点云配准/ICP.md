ICP基本原理：  
![image](https://user-images.githubusercontent.com/74122331/131496214-dbbbb878-ceee-4482-bc74-c0d3f0873afc.png)  

用到的一些知识介绍：  
## 1.奇异值分解  
[参考链接](https://zhuanlan.zhihu.com/p/29846048)  
首先从特征值的概念出发。对于一个n阶方阵**A**，如果满足![](https://www.zhihu.com/equation?tex=Ax%3D%5Clambda+x)，那么这个矩阵是一个正交矩阵。它有n个特征值和特征向量。
把n个特征向量组成n×n的矩阵，再进行正交化得到矩阵**W**，它满足![](https://www.zhihu.com/equation?tex=W%5E%7BT%7DW%3DI)，是一个酉矩阵。  
那么原矩阵**A**就可以改写成：  
![](https://pic3.zhimg.com/80/v2-f51625f69655c3ad594ff8062e1427e6_720w.jpg)  
其中，**Σ**是以n个特征值为主对角线的对角矩阵。  
这样的分解要求**A**必须是方阵，当**A**不是方阵时，就需要用奇异值分解（SVD）来完成这一过程。  
SVD的公式为：  
![](https://pic3.zhimg.com/80/v2-a71a3b4be58eaea23992595d495c55ce_720w.jpg)  
其中，**U**是m×m酉矩阵，**V**是n×n酉矩阵。**Σ**是m×n对角矩阵，主对角线上的每个元素都称为奇异值。  


# Efficient Variants of the ICP Algorithm论文解读  
ICP的各种变体可以理解为影响下面六个步骤中的一个或几个：  
1.从待配准的两个点云（也可以只用其中的一个点云）中选择一些顶点  
2.将这些点和另一个点云中的样本点进行匹配  
3.适当对相应的点对进行加权  
4.在单独观察每一对或者考虑整个点集的基础上拒绝某些对  
5.基于点对分配误差指标  
6.最小化误差指标  
