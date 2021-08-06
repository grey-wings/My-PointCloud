import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# 本文件在仓库中的目录：Pointnet_Pointnet2_pytorch/models/pointnet_utils.py

class STN3d(nn.Module):
    """
    T-Net(input_transform_net)
    针对点云的旋转不变性，引入该T-Net来学习点云的旋转，将物体校准。
    在input transform中，校准nX3的点云数据（n个样本，xyz三个维度）
    因此只需要学习一个3x3的矩阵。

    点云由N个D维的点组成。令B为这个mini-batch中的样本数目。
    """
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        # 这里应该是一个一维卷积，把一个3维数据映射到64维上
        # 第一个参数表示输入通道数，第二个参数表示输出通道数，第三个参数表示kernel的大小
        # 关于一维卷积的理解可以参考https://blog.csdn.net/qq_36323559/article/details/102937606
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        # 全连接层，将1024个特征的样本转为512个特征的样本，不管样本的数量。
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        # 归一化层，见吴恩达课程笔记，参数是来自期望输入的特征数
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        根据前面的一些定义，x是一个BxDxN的tensor，这里D=3.
        """
        batchsize = x.size()[0]
        # 返回值是tuple的子类，这里取的是x的第一个维度
        # eg.x是(a, b, c)维度的张量，则这里取的a
        x = F.relu(self.bn1(self.conv1(x)))   # Bx64xN
        x = F.relu(self.bn2(self.conv2(x)))   # Bx128xN
        x = F.relu(self.bn3(self.conv3(x)))   # Bx1024xN
        x = torch.max(x, 2, keepdim=True)[0]  # Bx1024x1 (?猜的，需要继续考证)
        # dim代表取最大值的维度。0是每列的最大值，1是每行的最大值。
        # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。
        # 在x的第三个维度上取最大值（就是保留上例中的a, b两个维度）
        # 点云具有置换不变性，即点的排序不影响物体的性质。需要用对称函数来处理置换不变性。
        # max是一个对称函数，具体的处理方法见https://blog.csdn.net/weixin_39373480/article/details/88878629
        # 大概是相当于取xyz三个坐标中的最大值
        x = x.view(-1, 1024)   # Bx1024(?猜的)

        x = F.relu(self.bn4(self.fc1(x)))   # Bx512(?猜的)
        x = F.relu(self.bn5(self.fc2(x)))   # Bx256(?猜的)
        x = self.fc3(x)   # Bx9(?猜的)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)  # 这段不知道什么意思
        return x


class STNkd(nn.Module):
    """
    feature_transform_net
    """
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        # np.eye()生成一个对角矩阵，flatten令该矩阵变为一维，相当于input transform net中对应代码的一般化
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)    # 学习input transform的旋转矩阵
        x = x.transpose(2, 1)  # 转置这个张量的第2维和第1维，即x的size变为(B, N, D)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)  # 进行批矩阵乘操作。(BxNx3)与(Bx3x3)相乘，得到(BxNx3)的矩阵
        if D > 3:
            x = torch.cat([x, feature], dim=2)  # 仅对xyz三个坐标进行旋转，其他特征不变（这个结论自己猜的）
        x = x.transpose(2, 1)  # 把x转置回(BxDxN)
        x = F.relu(self.bn1(self.conv1(x)))  # x: Bx64xN
        # 这一步应该对应的mlp(64, 64)这个部分，但是它只有一层，或许前面这个64代表输入？（待考证）

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))  # x: Bx128xN
        x = self.bn3(self.conv3(x))          # x: Bx1024xN
        # 这两步代表mlp(64, 128, 1024)这个部分，同样不知道64对应什么？
        x = torch.max(x, 2, keepdim=True)[0]   # max pooling, x变为Bx1024x1
        x = x.view(-1, 1024)
        if self.global_feat:
        # 这里的x是global feature, 如果是分类问题，只需要x
        # 如果是分割问题，要把局部特征和全局特征连接起来
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    """
    论文中提到的在softmax训练损失中添加的正则化项，它使特征转换
    矩阵(feature transformation matrix)限制在一个正交矩阵的范围内。
    """
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    # torch.norm返回p范数，p默认为2
    # （但是不知道为什么这里要取平均值）
    return loss
