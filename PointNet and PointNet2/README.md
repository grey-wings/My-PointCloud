# Pytorch版本的PointNet和PointNet++实践
原仓库：[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)  


# 下面是README文档的翻译  
# PointNet和PointNet++的Python实现  

这个仓库是[PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)和[PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf)的PyTorch实现。  

## 更新
**2021/03/27:** 

(1) 发布用于语义分割的预训练的模型，在这里PointNet++可以达到**53.5\%** 的mIoU.

(2) 在`log/`中发布用于分类和零件分割的预训练的模型。

**2021/03/20:** 更新用于分类的代码，包括：

(1) 增加用于训练**ModelNet10**的代码。 Using setting of ``--num_category 10``. 

(2) 增加只在CPU上运行的代码。 Using setting of ``--use_cpu``. 

(3) 添加离线数据预处理的代码以加快训练。 Using setting of ``--process_data``. 

(4)增加均匀抽样的训练代码。 Using setting of ``--use_uniform_sample``. 

**2019/11/26:**

(1) 修正了以前代码中的一些错误，并增加了数据增强的技巧。 Now classification by only 1024 points can achieve **92.8\%**! 

(2) 增加了测试代码，包括分类和分割，以及具有可视化的语义分割。

(3)将所有模型组织到`./models`文件中，以方便使用。

## Install
最新的代码在Ubuntu 16.04、CUDA10.1、PyTorch 1.6和Python 3.7上测试：
```shell
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
```

## Classification (ModelNet10/40)
### 数据准备
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

### Run
你可以用以下代码运行不同的模式。
* 如果你想使用离线处理数据，你可以在第一次运行时使用`--process_data`。你可以在[这里](https://drive.google.com/drive/folders/1_fBYbDO3XSdRt3DSbEBe41r5l9YpIGWF?usp=sharing)下载预处理数据并将它保存在`data/modelnet40_normal_resampled/`.
* 如果你想在ModelNet10上训练，你可以使用 `--num_category 10`.
```shell
# ModelNet40
## Select different models in ./models 

## e.g., pointnet2_ssg without normal features
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
python test_classification.py --log_dir pointnet2_cls_ssg

## e.g., pointnet2_ssg with normal features
python train_classification.py --model pointnet2_cls_ssg --use_normals --log_dir pointnet2_cls_ssg_normal
python test_classification.py --use_normals --log_dir pointnet2_cls_ssg_normal

## e.g., pointnet2_ssg with uniform sampling
python train_classification.py --model pointnet2_cls_ssg --use_uniform_sample --log_dir pointnet2_cls_ssg_fps
python test_classification.py --use_uniform_sample --log_dir pointnet2_cls_ssg_fps

# ModelNet10
## Similar setting like ModelNet40, just using --num_category 10

## e.g., pointnet2_ssg without normal features
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg --num_category 10
python test_classification.py --log_dir pointnet2_cls_ssg --num_category 10
```

### Performance
| Model | Accuracy |
|--|--|
| PointNet (Official) |  89.2|
| PointNet2 (Official) | 91.9 |
| PointNet (Pytorch without normal) |  90.6|
| PointNet (Pytorch with normal) |  91.4|
| PointNet2_SSG (Pytorch without normal) |  92.2|
| PointNet2_SSG (Pytorch with normal) |  92.4|
| PointNet2_MSG (Pytorch with normal) |  **92.8**|

## Part Segmentation (ShapeNet)
### Data Preparation
在[这里](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)下载对齐的ShapeNet并保存在`data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`.
### Run
```
## Check model in ./models 
## e.g., pointnet2_msg
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
python test_partseg.py --normal --log_dir pointnet2_part_seg_msg
```
### Performance
| Model | Inctance avg IoU| Class avg IoU 
|--|--|--|
|PointNet (Official)	|83.7|80.4	
|PointNet2 (Official)|85.1	|81.9	
|PointNet (Pytorch)|	84.3	|81.1|	
|PointNet2_SSG (Pytorch)|	84.9|	81.8	
|PointNet2_MSG (Pytorch)|	**85.4**|	**82.5**	


## Semantic Segmentation (S3DIS)
### Data Preparation
在[这里](http://buildingparser.stanford.edu/dataset.html)下载3D室内解析数据集（S3DIS）并保存在`data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/`.
```
cd data_utils
python collect_indoor3d_data.py
```
处理后的数据将保存在 `data/s3dis/stanford_indoor3d/`.
### Run
```
## Check model in ./models 
## e.g., pointnet2_ssg
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5 --visual
```
可视化结果会保存在`log/sem_seg/pointnet2_sem_seg/visual/` ，你可以通过[MeshLab](http://www.meshlab.net/)对这些.obj文件进行可视化

### Performance
|Model  | Overall Acc |Class avg IoU | Checkpoint 
|--|--|--|--|
| PointNet (Pytorch) | 78.9 | 43.7| [40.7MB](log/sem_seg/pointnet_sem_seg) |
| PointNet2_ssg (Pytorch) | **83.0** | **53.5**| [11.2MB](log/sem_seg/pointnet2_sem_seg) |

## Visualization
### Using show3d_balls.py
```
## build C++ code for visualization
cd visualizer
bash build.sh 
## run one example 
python show3d_balls.py
```
![](/visualizer/pic.png)
### Using MeshLab
![](/visualizer/pic2.png)


## Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)


## 引用
如果你觉得这个repo对你的研究有用，请考虑引用它和我们的其他作品。
```
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
```
```
@InProceedings{yan2020pointasnl,
  title={PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling},
  author={Yan, Xu and Zheng, Chaoda and Li, Zhen and Wang, Sheng and Cui, Shuguang},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
```
@InProceedings{yan2021sparse,
  title={Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion},
  author={Yan, Xu and Gao, Jiantao and Li, Jie and Zhang, Ruimao, and Li, Zhen and Huang, Rui and Cui, Shuguang},
  journal={AAAI Conference on Artificial Intelligence ({AAAI})},
  year={2021}
}
```

## 使用此代码库的选定项目
* [PointConv: 3D点云上的深度卷积网络, CVPR'19](https://github.com/Young98CN/pointconv_pytorch)
* [深度三维点云模型在对抗性攻击下的等值稳健性, CVPR'20](https://github.com/skywalker6174/3d-isometry-robust)
* [使用近似凸面分解在点云上进行标签高效学习, ECCV'20](https://github.com/matheusgadelha/PointCloudLearningACD)
* [PCT：点云转换器](https://github.com/MenghaoGuo/PCT)
* [Point Sampling Net:用于点云深度学习的快速子采样和局部分组](https://github.com/psn-anonymous/PointSamplingNet)

