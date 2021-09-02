# DGCNN论文解读&笔记  
[原文链接](https://arxiv.org/pdf/1801.07829.pdf)  

## 1.综述  
卷积神经网络(CNNs)最近在图像分析方面的巨大成功表明了将来自CNN的洞察力应用于点云世界的价值。点云天生缺乏拓扑信息，因此设计一个模型来恢复拓扑可以丰富点云的表示能力。
DGCNN的核心是EdgeConv。它作用于每一层网络所计算的动态图像。EdgeConv有几个吸引人的特性:它结合了局部邻域信息；它可以堆叠应用于学习全局形状属性；并且在多层系统中，特征空间中的相似性捕获原始嵌入中潜在长距离上的语义特征。  
它捕获局部几何结构，同时保持排列不变性。  
EdgeConv易于实现并集成到现有的深度学习模型中，以提高其性能。该论文将EdgeConv集成到了PointNet中。  

## 2.架构  
### 1.EdgeConv  
边缘卷积（EdgeConv）的性质介于置换不变性和局部性之间。它运用了CNN的思想，通过构造局部邻域图并在连接相邻点对的边上应用卷积式运算来利用局部几何结构。  
与CNN不同的是，在这个模型里用到的图不是固定的，而是根据网络层数的变化而更新的。  
对于一个F维点云点云X，满足![image](https://user-images.githubusercontent.com/74122331/131762264-7704defb-bb1b-42f3-8863-dcbe8697edd3.png)。F最简单的情况下是3，表示xyz坐标。但是也可以大于3，增加法线、颜色等信息。
我们计算一个表示局部特征的有向图G = (V, E)，其中V = {1，n}，E ⊆ V × V分别表示顶点和边。在最简单的情况下，我们将G构造为RF中X的k-最近邻（k-NN）图。该图包括自循环，这意味着每个节点也指向自身。我们将边缘特征定义为eij=hΘ（xi，xj），其中hΘ：RF×RF→RF′是一个具有一组可学习参数的非线性函数。然后通过一个通道对称的聚合操作囗来处理边信息。EdgeConv的第i个顶点的输出由以下公式给出： ![image](https://user-images.githubusercontent.com/74122331/131852523-1e71315b-2bed-4e21-ab85-74a094ef7f00.png)  
类似于图像的卷积，我们重新计算中心像素和{xj:（i，j）∈ E} 作为其周围的补丁（参见下图）。总的来说，给定一个具有n个点的F维点云，EdgeConv生成一个具有相同点数目的F′维点云。
![image](https://user-images.githubusercontent.com/74122331/131852732-57003cfe-4c0f-447c-bab6-3f070aba9fd8.png)
左边：从点对xi，xj计算边缘特征eij；右边：EdgeConv操作。EdgeConv的输出是通过聚合与每个连接顶点发出的所有边相关联的边特征来计算的。  

边函数h和聚合操作囗的选择对EdgeConv的影响很大。在这篇文章中，采用了一种不对称边函数![image](https://user-images.githubusercontent.com/74122331/131854020-03ad32a9-cc07-4bc8-bca4-6bf3d426e233.png)。这结合了由补丁（patch）中心xi捕获的全局形状结构和xj - xi捕获的局部邻近信息。在这里定义![image](https://user-images.githubusercontent.com/74122331/131854275-70139e16-a95e-4c29-bd3f-2eb75d4fb2c5.png)和![image](https://user-images.githubusercontent.com/74122331/131854338-c13fe8f9-9a08-4641-bbde-45fe187acb8c.png)

## 2.动态图像更新  
在每一层，有一个动态的图![image](https://user-images.githubusercontent.com/74122331/131855356-a9d0b60f-6161-44f8-b227-4d50f2fcd649.png)，其中第l层边的形式为![image](https://user-images.githubusercontent.com/74122331/131855415-5e134e64-c3fb-4aef-8d7f-f30940332155.png)![image](https://user-images.githubusercontent.com/74122331/131855608-e73985e0-2880-4d8f-a8eb-055aad98f934.png)是离![image](https://user-images.githubusercontent.com/74122331/131855510-36f4c5d8-ea2c-481a-be19-b799b34ac40c.png)最近的kl个点
