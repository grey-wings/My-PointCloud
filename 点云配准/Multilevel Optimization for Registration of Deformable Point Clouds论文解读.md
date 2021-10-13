# Multilevel Optimization for Registration of Deformable Point Clouds论文解读  

## 三、问题公式化  
考虑两个三维点集![image](https://user-images.githubusercontent.com/74122331/137153979-59ea2f6f-3bf8-45ab-960f-576508eed560.png)以及![image](https://user-images.githubusercontent.com/74122331/137154041-20983e98-0d30-4e7a-a08d-5f1de8c796c0.png)![image](https://user-images.githubusercontent.com/74122331/137154113-e1c9bb77-1b01-4005-9c50-8c2fa9da5704.png)，P为模型（model），Q为数据（data）。任务是对齐Q和P，使每个顶点和ground truth（可以理解为正确答案的意思，解释可以看[这里](https://www.zhihu.com/question/22464082)）的对应误差（correspondence error）最小。在一般情况下，我们要寻找一个任意的全局变换函数T，使![image](https://user-images.githubusercontent.com/74122331/137156609-4c228ced-3d93-4ee1-826d-cc1766aef4f5.png)成立。  
但对于可变形点云模型，全局变换函数不能正确恢复点对应关系，需要对每个点的变换函数进行逼近。将每个点的变换函数记为![image](https://user-images.githubusercontent.com/74122331/137156983-2accd77a-f040-476d-a415-60bc28878288.png)，那么变换形式就是![image](https://user-images.githubusercontent.com/74122331/137157065-2817ca49-769b-4e55-ac08-c25062f03085.png)
。然而，估计云中的每个点的![image](https://user-images.githubusercontent.com/74122331/137156983-2accd77a-f040-476d-a415-60bc28878288.png)是一项极其困难的任务，通常使用简化变形模型的概念以一种易于处理的方式来解决这个问题。该模型的思想是将点聚集成少量的组，其中邻近的点可能经历类似类型的转换。

