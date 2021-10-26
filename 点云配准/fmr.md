# Feature-metric Registration: A Fast Semi-supervised Approach for Robust Point Cloud Registration without Correspondences 论文解读  

## 一、原理  
在本节中，首先描述了注册框架的问题形成和概述。其次，详细介绍了Encoder模块。第三，我们展示了如何学习一个特征并解决多任务流中的注册问题。第四，详细说明了损耗函数。  
### 1.问题公式化  
给定两个点云，配准的目标是找到刚性变化参数g（旋转矩阵R∈SO(3)和平移向量t，SO(3)：特殊正交群，是行列式为1的正交矩阵。关于李群和李代数的知识见下文）。
其中，![image](https://user-images.githubusercontent.com/74122331/138547372-fba3e0aa-1226-4309-a3c3-826354e1d24f.png)是P和Q的feature-metric误差。
![image](https://user-images.githubusercontent.com/74122331/138547398-da3233db-587f-45ef-84b7-58a8914f16f7.png)是点云P的特征。K为特征维数(实验中为1024)，F为Encoder模块学习的特征提取函数。
为了求解上述方程(1)，我们提出了一个结合经典非线性算法和无监督学习技术优点的特征度量配准框架。框架可以以半监督或无监督的方式进行训练。图2显示了该算法的概述。首先，对两个输入点云提取两个旋转注意特征;然后将特征输入到多任务模块中。在第一个分支(Task1)中，设计了一个解码器以无监督的方式训练encoder模块。在第二个分支中，计算投影误差来表示两个输入特征之间的差异，并通过最小化特征差异来估计最佳变换。迭代运行变换估计，通过运行逆合成(IC)算法[2]来估计每一步的变换增量(△θ):
![image](https://user-images.githubusercontent.com/74122331/138547440-0fd4bd99-9b57-4f49-ac59-375c57fdbe21.png)  
其中r是特征度量投影误差，![image](https://user-images.githubusercontent.com/74122331/138547456-88a92a7d-b4ad-4f28-9c6d-bbcf30b69c01.png)是r相对于变换参数(θ)的雅可比矩阵。参考[2]，在2D图像中直接计算J是非常昂贵的。该方法应用于经典二维图像，利用链式法则将雅可比矩阵估计为图像梯度和翘曲雅可比矩阵两个部分。然而，这种方法并不适用于无序的3D点云，因为没有网格结构允许我们计算x,y和z方向的3D点云的梯度[1]。  
为了有效地计算雅可比矩阵，我们使用不同的有限梯度来计算雅可比矩阵，而不是用随机梯度法计算雅可比矩阵，参见[1]:  
![image](https://user-images.githubusercontent.com/74122331/138547465-1c16bd42-332a-4d20-927c-356f8b6ea445.png)
其中ξ= (Ri, ti)为迭代过程中变换参数的无穷小扰动。本文给出了扰动的六个运动参数，旋转时给出三个角度参数(v1, v2, v3)，平移时给出三个抖动参数(t1, t2, t3)。请参阅[1]，在所有迭代中，小的固定值(2∗e−2)将生成最佳结果。经过几次迭代时间(实验使用10次)，该方法输出最终的变换矩阵(R和t)和特征度量投影误差![image](https://user-images.githubusercontent.com/74122331/138547488-8f56b885-93c3-4da4-9464-2c688a8dab92.png)。  
### 2.Encoder  
编码器模块的目标是学习一个特征提取函数F，该函数可以为输入点云生成一个独特的特征。编码器网络设计的主要原则是生成的特征应注意旋转，以反映变换估计时的旋转差异。参考PointNet[19]，特征是通过两个mlp层和一个max-pool层提取的。我们丢弃了输入变换层和特征变换层，以使特征感知旋转差异。  
### 3.多任务分支  
特征提取完成后，下一步是特征学习和点云变换估计。通过直接最小化特征投影误差来确定估计结果，不需要搜索对应。本设计将学习适合特征空间迭代优化方法的特征。  
#### 3.1 Encoder-Decoder分支(任务1):    
受[11]启发，编码器模块生成了独特的特征后，我们使用一个解码器模块将特征恢复到三维点云中。这个编码器-解码器分支可以在无监督的方式下进行训练，这有助于Encoder模块学习生成感知旋转差异的显著特征。对于一个点云的两个旋转副本PC1和PC2，该分支的原理是编码器模块对P1和P2产生不同的特征，解码器可以将不同的特征恢复到对应的点云旋转副本。这一原则保证了无监督学习在训练配准问题的显著特征提取器方面的成功。  
解码器块由四层全连接层组成，由LeakyReLU激活。解码器模块的输出与输入点云的维度相同。  
#### 3.2 feature-metric配准分支(任务2):  
为了解决配准问题，如图2所示，我们使用逆合成算法(非线性优化)估计变换参数，以最小化特征度量投影误差。定义特征度量投影误差为  
![image](https://user-images.githubusercontent.com/74122331/138547726-1df555e0-505c-43e2-92a0-c07e761ad008.png)  
其中，![image](https://user-images.githubusercontent.com/74122331/138547730-dd7b8cfb-1063-45a3-b169-da32244c8c02.png)是点云P或g·Q的全局特征。g是变换矩阵（R和t）。  
为了更好地了解特征网络在非线性优化过程中提取了什么，我们将特征地图和迭代过程可视化，如图3所示。我们将全局特征映射重构为方阵，并将其显示为图像。图3为点云P和变换后的点云Q的特征图，以及第一次迭代、第5次迭代和最后10次迭代的特征图差异。查看图3的最后三列，当对齐变得更准确时(底部行)，特征图的差异(顶部行)变得更小。  
 ![image](https://user-images.githubusercontent.com/74122331/138547922-dcafba19-b744-4aed-b2ce-2c63eb850806.png)
图注：特征提取网络生成的特征图以及迭代过程中的特征差异。第一次迭代时，P与变换后的Q之间的feature map差值(上)较大，对齐(下)不好。在第十次迭代时，点cloudPand变换后的q的特征图差异极小，对齐近乎完美。 
![image](https://user-images.githubusercontent.com/74122331/138551272-7a23fb0e-107d-4a82-9315-8a3530f5ac1d.png)  
(1)-encoder提取两个输入点云(P和Q)的特征。(2)-然后，多任务半监督神经网络旨在解决没有对应的配准问题。在task1中，Decoder解码特征。整个编解码器分支以无监督的方式训练编码器网络。在Task2中，基于两个输入特征(FP和FQ)计算特征度量投影误差(r)。将特征误差输入到非线性优化算法中，估计变换增量(△θ)，更新变换参数。利用更新后的变换参数(θk+1)，对输入点云Q进行变换，并迭代运行整个过程。  
### 4.损失函数  
训练的关键任务是用于提取旋转注意特征的Encoder模块。我们提出两个损失函数来提供半监督框架。直接忽略监督几何损失可以很容易地扩展到无监督框架。  
#### 4.1 倒角损失(Chamfer loss)  
编码器-解码器分支可以在无监督的方式下进行训练。参考[11]，使用倒角距离损耗:  
![image](https://user-images.githubusercontent.com/74122331/138548048-0ef87b90-83fb-4f79-ab3a-cb08cc6092ae.png)  


## 二、李群和李代数简介  
在原作者提供的代码中，用到了李群和李代数的相关知识。下面进行简要介绍。  
[参考链接](https://zhuanlan.zhihu.com/p/358455662)将点云Q与P最佳对齐如下图所示:![image](https://user-images.githubusercontent.com/74122331/138546569-89b72dfd-57b8-4bff-82d8-cd353db6a425.png)  
设有两个坐标系，一个的正交基底是[e1, e2, e3],，还有一个的正交基底是[e1', e2', e3']。他们的原点是相同的，也就是只有旋转关系。同一个点在这两个坐标系中的坐标分别为[a1, a2, a3]和[a1', a2', a3']。则他们的关系如下：  
![](https://www.zhihu.com/equation?tex=+%5Be_1%2Ce_2%2Ce_3%5D+%5Cleft+%5B+%5Cbegin%7Bmatrix%7D+a_1%5C%5C+a_2%5C%5C+a_3+%5Cend%7Bmatrix%7D+%5Cright+%5D+%3D+%5Be_1%5E%7B%27%7D%2Ce_2%5E%7B%27%7D%2Ce_3%5E%7B%27%7D%5D+%5Cleft+%5B+%5Cbegin%7Bmatrix%7D+a_1%5E%7B%27%7D%5C%5C+a_2%5E%7B%27%7D%5C%5C+a_3%5E%7B%27%7D+%5Cend%7Bmatrix%7D+%5Cright+%5D+)  
为了能更好的表示这个点，在新旧坐标系下的坐标值的关系，我们整理一下，两边同时左乘一个![](https://www.zhihu.com/equation?tex=%5Be_1%5ET%2Ce_2%5ET%2Ce_3%5ET%5D%5ET) ，于是等式变成了：  
![](https://www.zhihu.com/equation?tex=+%5Cleft+%5B+%5Cbegin%7Bmatrix%7D+a_1%5C%5C+a_2%5C%5C+a_3+%5Cend%7Bmatrix%7D+%5Cright+%5D+%3D%5Cbegin%7Bbmatrix%7D+e_1%5ETe_1%5E%7B%27%7D+%26+e_1%5ETe_2%5E%7B%27%7D++%26+e_1%5ETe_3%5E%7B%27%7D+%5C%5C++e_2%5ETe_1%5E%7B%27%7D+%26+e_2%5ETe_2%5E%7B%27%7D++%26+e_2%5ETe_3%5E%7B%27%7D+%5C%5C++e_3%5ETe_1%5E%7B%27%7D+%26+e_3%5ETe_2%5E%7B%27%7D++%26+e_3%5ETe_3%5E%7B%27%7D++%5Cend%7Bbmatrix%7D+%5Cleft+%5B+%5Cbegin%7Bmatrix%7D+a_1%5E%7B%27%7D%5C%5C+a_2%5E%7B%27%7D%5C%5C+a_3%5E%7B%27%7D+%5Cend%7Bmatrix%7D+%5Cright+%5D+)    
用R来表示那个3 * 3的矩阵，称为旋转矩阵。  
这个旋转矩阵是一个行列式为1的正交矩阵。它的逆就是它的转置，表示把这个旋转操作再逆回去。反之，行列式为1的正交矩阵也可以作为旋转矩阵。它们构成一个集合：![](https://www.zhihu.com/equation?tex=+%5Cbegin%7BBmatrix%7D+R+%5Cin+%5Cmathbb%7BR%7D%5E%7Bn%5Ctimes+n%7D+%7C+R+%5Ctimes+R%5ET+%3D+I+%7C+det%28R%29+%3D+1+%5Cend%7BBmatrix%7D+)  
这类矩阵称为特殊正交群(Special Orthogonal Group)，在三维空间中，n = 3，就称为SO(3):![](https://www.zhihu.com/equation?tex=+SO%283%29+%3D+%5Cbegin%7BBmatrix%7D+R+%5Cin+%5Cmathbb%7BR%7D%5E%7B3%5Ctimes+3%7D+%7C+R+%5Ctimes+R%5ET+%3D+I+%7C+det%28R%29+%3D+1+%5Cend%7BBmatrix%7D+)  
在齐次坐标系中，有这样一类集合：  
![](https://www.zhihu.com/equation?tex=+SE%283%29+%3D+%5Cleft+%5C%7B+T+%3D+%5Cbegin%7Bbmatrix%7D+R+%26+t%5C%5C++0%5ET+%26+1++%5Cend%7Bbmatrix%7D+%5Cin+%5Cmathbb%7BR%7D%5E%7B4%5Ctimes+4%7D+%7C+R+%5Cin+SO%283%29%2C+t+%5Cin+%5Cmathbb%7BR%7D%5E%7B3%7D%5Cright+%5C%7D+)  
它的左上角是一个旋转矩阵R，右侧是平移向量t，左下角为0块，右下角为1.
