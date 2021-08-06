"""
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
"""
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    # 考虑在ModelNetDataLoader中的应用，pc为npoints×3的向量
    centroid = np.mean(pc, axis=0)

    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    """
    对应modelnet40_normal_resampled的数据集。
    pytorch关于map style dataset的说明：
    表示从键到数据样本的映射的所有数据集都应该对其进行子类化。所有子类都应该
    覆盖（overwrite）__getitem__()，支持获取给定键的数据样本。子类还可以选择性地覆盖__len__()，
    在许多采样器实现和DataLoader的默认选项中，这个东西被期望返回数据集的大小。
    """

    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        # 将每一类物体的名字整理成列表
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        # 给每一类附加一个数字标签

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        # shape_ids存储训练集或测试集所包含的样本文件名（不含后缀） 形如"bathtub_0001"

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # 去除样本的编号 之所以用join是为了防止tv_stand这种名字中的下划线被误分割
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i],
                                                       shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        # 生成一个由元组组成的列表，每个元组第1个元素是样本名(如bathtub)，第二个元素是样本路径
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' %
                                          (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' %
                                          (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
                    # N×6矩阵

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    # 存储每个数据文件中的点集数据
                    self.list_of_labels[index] = cls
                    # 存储这个点集样本的标签（数字形式）

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
                    # pickle提供了一个简单的持久化功能。可以将对象以文件的形式存放在磁盘上
                    # 序列化后的文件没有可读性
                    # dump将数据流写入文件中

            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)
                    # load将文件中的数据解析为一个Python对象

    def __len__(self):
        """
        :return: 数据集的大小。
        """
        return len(self.datapath)

    def _get_item(self, index):
        """

        :param index:
        :return:
        """
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            # 存储了第index个文件的类别名和路径的元组
            cls = self.classes[self.datapath[index][0]]
            # 该类别的数字标签
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            # N×6，N为这个样本中点的个数
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                # 直接选取前npoints个点，npoints×6

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
