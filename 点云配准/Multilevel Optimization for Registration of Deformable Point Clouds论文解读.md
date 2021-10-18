# Multilevel Optimization for Registration of Deformable Point Clouds论文解读  

## 三、问题公式化  
考虑两个三维点集![image](https://user-images.githubusercontent.com/74122331/137153979-59ea2f6f-3bf8-45ab-960f-576508eed560.png)以及![image](https://user-images.githubusercontent.com/74122331/137154041-20983e98-0d30-4e7a-a08d-5f1de8c796c0.png)![image](https://user-images.githubusercontent.com/74122331/137154113-e1c9bb77-1b01-4005-9c50-8c2fa9da5704.png)，P为模型（model），Q为数据（data）。任务是对齐Q和P，使每个顶点和ground truth（可以理解为正确答案的意思，解释可以看[这里](https://www.zhihu.com/question/22464082)）的对应误差（correspondence error）最小。在一般情况下，我们要寻找一个任意的全局变换函数T，使![image](https://user-images.githubusercontent.com/74122331/137156609-4c228ced-3d93-4ee1-826d-cc1766aef4f5.png)成立。  
但对于可变形点云模型，全局变换函数不能正确恢复点对应关系，需要对每个点的变换函数进行逼近。将每个点的变换函数记为![image](https://user-images.githubusercontent.com/74122331/137156983-2accd77a-f040-476d-a415-60bc28878288.png)，那么变换形式就是![image](https://user-images.githubusercontent.com/74122331/137157065-2817ca49-769b-4e55-ac08-c25062f03085.png)
。然而，估计云中的每个点的![image](https://user-images.githubusercontent.com/74122331/137156983-2accd77a-f040-476d-a415-60bc28878288.png)是一项极其困难的任务，通常使用简化变形模型的概念以一种易于处理的方式来解决这个问题。该模型的思想是将点聚集成少量的组，其中邻近的点可能经历类似类型的转换。那么变形公式可以写为：  
![image](https://user-images.githubusercontent.com/74122331/137157370-b7203a41-54e9-4f02-9407-d10038dfc27c.png)  
因子1用于齐次坐标。下面是对齐次坐标的解释（[参考链接](https://www.cnblogs.com/zhizhan/p/3999885.html)）：  
在投影空间中，两条平行线会相交于一点，就像下面的铁路：  
![r_railroad](https://user-images.githubusercontent.com/74122331/137577866-21b0f7a7-946b-4453-8586-35f11217d631.jpg)  
在笛卡尔空间里面这个问题是无法解决的，因此引入齐次坐标的概念。齐次坐标用 N + 1个分量来描述 N 维坐标。比如，2D 齐次坐标是在笛卡尔坐标(X, Y)的基础上增加一个新分量 w，变成(x, y, w)，其中笛卡尔坐标系中的大X，Y 与齐次坐标中的小x，y有如下对应关系：  
X = x/w  
Y = y/w   
因此，齐次坐标中的点(1, 2, 1)在笛卡尔坐标中就代表(1, 2)；齐次坐标中的点(1, 2, 0)就代表无穷远处的点。  
齐次坐标具有缩放不变性，如下例：   
![r_homogeneous02](https://user-images.githubusercontent.com/74122331/137577874-a30b5a00-3e86-47b1-802e-c156e0aea771.png)  
c是前面将点聚集成的组的数量。基本上模型中的每个点![image](https://user-images.githubusercontent.com/74122331/137577788-41087c3d-a5e3-4feb-996a-3792b834526e.png)
都受到变换![image](https://user-images.githubusercontent.com/74122331/137577857-34081d2a-1272-482a-ac37-193c47bb2466.png)的影响。通常在可变形模型的假设中，c < m的准则为真，c的选择取决于模型的复杂性。当c=m时，问题变得非常棘手，而非常低的c值对于变换估计可能是无效的。在这项工作中，我们提出了一种分层的方法来找到不同层次上的转换参数![image](https://user-images.githubusercontent.com/74122331/137577857-34081d2a-1272-482a-ac37-193c47bb2466.png)，我们通过假设上述的简化变形模型来估计转换。然后这些变形的点被“缝合”在一个更高的层次上，这个过程不断重复，直到没有足够的点更新。该方法通过将问题简化为不同的层次来逼近变形。我们在实验结果部分证明，方法只考虑一个单一的几何水平，不能以稳健的方式近似大量的变形。  
我们假设在模型和数据(可以通过特征匹配算法得到)之间的关键点(η1，···，ηk)之间有k个初始对应数(initial correspondence)。我们把点集P的表面表征(surface representation)或三角测量(triangulation)定义为D，点集Q的表面表征定义为S。然后，根据每个顶点到相应关键点的测地线距离，从顶点的接近度开始计算,我们得到一组表面补丁(surface patches)![image](https://user-images.githubusercontent.com/74122331/137662365-1aac026f-e796-49a3-a485-dc2d95054078.png)以及![image](https://user-images.githubusercontent.com/74122331/137662405-58b10f66-eb86-4b45-b0f2-a7cd5f180905.png)。每个![image](https://user-images.githubusercontent.com/74122331/137662677-208b92dc-4369-4e54-83a1-99fc545c4c07.png)和![image](https://user-images.githubusercontent.com/74122331/137662703-319b5766-a55a-4e79-8e98-fdd8afc276b4.png)进一步划分r个环邻域(定义在下一节)，表示为![image](https://user-images.githubusercontent.com/74122331/137662737-db400ed0-9332-4ff4-bf46-186b65d2eb50.png)![image](https://user-images.githubusercontent.com/74122331/137662757-c87a9bc3-c35d-4ba3-9979-d681c9c64d98.png)和![image](https://user-images.githubusercontent.com/74122331/137662915-ef14d797-d66a-466c-b84d-c05febd6785e.png)





