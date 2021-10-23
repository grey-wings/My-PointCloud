# Feature-metric Registration: A Fast Semi-supervised Approach for Robust Point Cloud Registration without Correspondences 论文解读  

## 第三节 原理  
在本节中，首先描述了注册框架的问题形成和概述。其次，详细介绍了Encoder模块。第三，我们展示了如何学习一个特征并解决多任务流中的注册问题。第四，详细说明了损耗函数。  
### 1.问题公式化  
给定两个点云，配准的目标是找到刚性变化参数g（旋转矩阵R∈SO(3)和平移向量t）将点云Q与P最佳对齐如下图所示:![image](https://user-images.githubusercontent.com/74122331/138546569-89b72dfd-57b8-4bff-82d8-cd353db6a425.png)
其中，![image](https://user-images.githubusercontent.com/74122331/138547372-fba3e0aa-1226-4309-a3c3-826354e1d24f.png)是P和Q的feature-metric误差。
![image](https://user-images.githubusercontent.com/74122331/138547398-da3233db-587f-45ef-84b7-58a8914f16f7.png)是点云P的特征。K为特征维数(实验中为1024)，F为Encoder模块学习的特征提取函数。
为了求解上述方程(1)，我们提出了一个结合经典非线性算法和无监督学习技术优点的特征度量配准框架。框架可以以半监督或无监督的方式进行训练。图2显示了该算法的概述。首先，对两个输入点云提取两个旋转注意特征;然后将特征输入到多任务模块中。在第一个分支(Task1)中，设计了一个解码器以无监督的方式训练encoder模块。在第二个分支中，计算投影误差来表示两个输入特征之间的差异，并通过最小化特征差异来估计最佳变换。迭代运行变换估计，通过运行逆合成(IC)算法[2]来估计每一步的变换增量(△θ):
![image](https://user-images.githubusercontent.com/74122331/138547440-0fd4bd99-9b57-4f49-ac59-375c57fdbe21.png)  
其中r是特征度量投影误差，![image](https://user-images.githubusercontent.com/74122331/138547456-88a92a7d-b4ad-4f28-9c6d-bbcf30b69c01.png)是r相对于变换参数(θ)的雅可比矩阵。参考[2]，在2D图像中直接计算J是非常昂贵的。该方法应用于经典二维图像，利用链式法则将雅可比矩阵估计为图像梯度和翘曲雅可比矩阵两个部分。然而，这种方法并不适用于无序的3D点云，因为没有网格结构允许我们计算x,y和z方向的3D点云的梯度[1]。

