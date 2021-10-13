# Multilevel Optimization for Registration of Deformable Point Clouds论文解读  

## 三、问题公式化  
考虑两个三维点集![image](https://user-images.githubusercontent.com/74122331/137153979-59ea2f6f-3bf8-45ab-960f-576508eed560.png)以及![image](https://user-images.githubusercontent.com/74122331/137154041-20983e98-0d30-4e7a-a08d-5f1de8c796c0.png)![image](https://user-images.githubusercontent.com/74122331/137154113-e1c9bb77-1b01-4005-9c50-8c2fa9da5704.png)，P为模型（model），Q为数据（data）。任务是对齐Q和P，使每个顶点和ground truth（可以理解为正确答案的意思，解释可以看[这里](https://www.zhihu.com/question/22464082)）的对应误差（correspondence error）最小。在一般情况下，我们要寻找一个任意的全局变换函数T，使![image](https://user-images.githubusercontent.com/74122331/137156609-4c228ced-3d93-4ee1-826d-cc1766aef4f5.png)成立。  

