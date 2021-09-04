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
