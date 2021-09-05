# Focal loss  

## 1.交叉熵损失（Cross Entropy，CE）  
交叉熵损失在二分类时定义如下：  
![image](https://user-images.githubusercontent.com/74122331/132095695-5eea4159-e29e-4149-bb05-704709b1c45e.png)  
其中y表示该样本真实的分类，p表示预测的该样本分类为y=1的概率。  
为了简化表示，我们作如下约定：  
![image](https://user-images.githubusercontent.com/74122331/132095745-f1f49ce1-895e-48d4-b5cc-b08b56c56d3f.png)  
并重写CE的表达式为：![image](https://user-images.githubusercontent.com/74122331/132095770-342c0ebb-9576-4491-a6f8-c0ae582f86eb.png)  
在下图中，CE是最顶端的蓝色曲线。即使是容易分类的示例（pt >> 5），损失值也不小。  
![image](https://user-images.githubusercontent.com/74122331/132095808-860f975d-5563-4a6f-9693-76df1c6889cf.png)  

为了解决类不平衡(class imbalance)问题，定义Balanced Cross Entropy，引入一个在0~1之间的权重因子α，与y=1的损失项相乘；1-α与y=0的损失项相乘。那么可以用和上面定义pt的方式来定义αt，这是CE损失的表达式为：![image](https://user-images.githubusercontent.com/74122331/132096131-f4f51952-a1e1-4cfa-9264-f0bb696390bc.png)  

## 2.Focal loss  
向CE中加入调节因子![image](https://user-images.githubusercontent.com/74122331/132115362-c48b1c6d-c90f-4e5d-8beb-c0332ebe71c3.png)，其中可调节的聚焦参数(focusing parameter)γ >= 0.  
定义Focal Loss为：![image](https://user-images.githubusercontent.com/74122331/132115415-69795374-621e-4487-9083-fad736b02696.png)  
