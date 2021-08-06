# 2021.8.6 PointNet2第一次试运行

# 一、文件夹结构
## 1.[data_utils](./data_utils)：数据处理工具  
[ModelNetDataLoader.py](./data_utils/ModelNetDataLoader.py)：用于加载ModelNet10或40数据集的程序。
## 2.[PointNet](./PointNet)：PointNet相关程序  
[pointnet_utils.py](./PointNet/pointnet_utils.py)：PointNet相关函数和部分网络架构。  
[pointnet_cls.py](./PointNet/pointnet_cls.py)：使用PointNet进行分类。  
## 3.[PointNet2](./PointNet2)：PointNet++相关程序  
[pointnet2_utils.py](./PointNet2/pointnet2_utils.py)：PointNet++相关函数和部分网络架构。 
[pointnet2_cls_msg.py](./PointNet2/pointnet2_cls_msg.py)：使用MSG（多尺度分组）方法的分类网络。
## 4.[Visualizer](./Visualizer)：可视化工具，这次没有用到。  

# 二、训练参数选择
```use_cpu:False(默认)``` 是否使用CPU  
```gpu:'0'(默认)``` GPU编号  
```batch_size:24(默认)``` 每个mini-batch的大小  
```model:'pointnet2_cls_msg'``` 所使用的网络结构  
```num_category:10``` 使用ModelNet10还是40  
```epoch:200(默认)``` 训练多少个epoch  
```learning_rate:0.001(默认)``` 学习率  
```num_point:1024(默认)``` 每个样本中选取的点的个数  
```optimizer:Adam(默认)``` 优化方法  
```log_dir:None(默认)``` 日志路径（不知道干什么用的）  
```decay_rate:1e-4(默认)``` 学习率衰减  
```use_normals:True``` 使用法向量数据  
```process_data:False(默认)```保存用python处理过的数据集  
```use_uniform_sample:False(默认)``` 是否在采num_point个点时使用fps采样  

