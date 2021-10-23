# Feature-metric Registration: A Fast Semi-supervised Approach for Robust Point Cloud Registration without Correspondences 论文解读  

## 第三节 原理  
在本节中，首先描述了注册框架的问题形成和概述。其次，详细介绍了Encoder模块。第三，我们展示了如何学习一个特征并解决多任务流中的注册问题。第四，详细说明了损耗函数。  
### 1.问题公式化  
给定两个点云，配准的目标是找到刚性变化参数g（旋转矩阵R∈SO(3)和平移向量t）将点云Q与P最佳对齐如下图所示:![image](https://user-images.githubusercontent.com/74122331/138546569-89b72dfd-57b8-4bff-82d8-cd353db6a425.png)
其中，![image](https://user-images.githubusercontent.com/74122331/138547372-fba3e0aa-1226-4309-a3c3-826354e1d24f.png)是P和Q的feature-metric误差。
![image](https://user-images.githubusercontent.com/74122331/138547398-da3233db-587f-45ef-84b7-58a8914f16f7.png)是点云P的特征。K为特征维数(实验中为1024)，F为Encoder模块学习的特征提取函数。
