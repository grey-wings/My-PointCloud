import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):  # 没看懂这是干什么的？
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    函数输入是两组点，N为第一组点的个数，M为第二组点的个数，C为输入点的通道数（如果是xyz时C=3），
    返回的是两组点之间两两的欧几里德距离，即N × M的矩阵。
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # permute将dst的第2个维度和第1个维度换位，得到BxCxM
    # 这时dist是一个B×N×M的tensor，每个位置存储
    # -2xn*xm-2yn*ym-2zn*zm
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    # 加上xn*xn + yn*yn + zn*zn
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    # 加上xm*xm + ym*ym + zm*zm，即为距离公式
    return dist


def index_points(points, idx):
    """
    按照输入的点云数据和索引返回由索引的点云数据。例如points为B×2048×3
    的点云，idx为[1, 333, 1000, 2000]，则返回B个样本中每个样本的
    第1、333、1000、2000个点组成的B×4×3的点云集。
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    # view_shape = [B, S]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    # [B, 1]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # 这里没看懂？
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    最远点采样（FPS）函数。
    采集npoint个点，满足点与点之间的距离尽可能远。
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # 存储所需要的npoint个点的索引
    distance = torch.ones(B, N).to(device) * 1e10
    # 利用distance矩阵记录某个样本中所有点到某一个点的距离
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # randint生成一个整数tensor，取值在第一个参数和第二个参数之间，前闭后开。
    # 第三个参数是一个元组，表示生成tensor的形状。
    # fathest表示当前最远点，先随机生成。
    # （这应该是一个Bx1的tensor？）
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # 生成一个长度为，取值从0到B-1的一个一维tensor
    for i in range(npoint):
        centroids[:, i] = farthest  # 更新当前最远点
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # 取出所选中点的xyz坐标
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # 这是一个BxN的tensor，对于每个样本，N个元素储存的是
        # N个点中每一个点到选定的最远点的距离，假设选定的最远点坐标为(x1,y1,z1)
        # 也即(x-x1)^2+(y-y1)^2+(z-z1)^2

        mask = dist < distance
        distance[mask] = dist[mask]
        # 更新一下distance，在N个点中的每一个点i，
        # distance里面存储点i到所有npoint个最远点的距离中，
        # 最近的那一个距离。这个距离就作为某个点到整个npoint点集的距离。
        farthest = torch.max(distance, -1)[1]
        # 返回距离最大者的索引
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    寻找球形领域中的点。
    Input:
        radius: local region radius（球形领域的半径）
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
        （S个球形邻域的中心，即由前面fps得到的中心采样点）
    Return:
        group_idx: grouped points index, [B, S, nsample]
        每个样本有S个中心点，每个中心点有nsample个值，每个值
        对应一个属于该中心的点的索引。
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # group_idx:B×S×N，为xyz点集中的每个点编号
    sqrdists = square_distance(new_xyz, xyz)
    # sqrdists: [B, S, N]记录中心点与所有点之间的欧几里德距离
    group_idx[sqrdists > radius ** 2] = N
    # 直接把距离大于两倍半径的点的索引标成N（最大）
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # sort返回一个元组，[0]表示排序后的tensor，[1]表示原始输入中的下标。
    # 在距离不大于2倍半径的点中选取nsample个点。
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # 满足条件的点可能不到nsample个，即选中的点中也有可能索引为N
    # 这些点需要去掉，这里采用的方法就是全部用第一个点替换。
    # 这里把第一个点复制nsample遍
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    # index为N的点就用第一个点替换
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    将整个点云分散成局部的group，对每一个group都可以用PointNet单独的提取局部的全局特征。
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]

    本函数主要步骤如下：
    1. 先用farthest_point_sample函数实现最远点采样FPS得到采样点的索引，
    再通过index_points将这些点的从原始点中挑出来，作为new_xyz
    2. 利用query_ball_point和index_points将原始点云以new_xyz作为中心
    分为npoint个球形区域，其中每个区域有nsample个采样点。
    3. 每个区域的点减去区域的中心值。
    4. 如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征。
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    # 采样点索引
    new_xyz = index_points(xyz, fps_idx)  # 采样点集，[B, npoint, C]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    # 得到每个中心所包含的点的索引，[B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    # 每个中心所包含的点的坐标信息
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    # 这里什么意思？

    if points is not None:
        # 如果点云集除了xyz坐标以外还有别的特征：
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        # 这一段也没看懂？
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    把所有点作为同一组，实际上只是增加一个为1的维度并拼接特征和xyz坐标。
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """
    set abstraction层由Sampling layer,
    Grouping layer和PointNet layer组成。
    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        """

        :param npoint:Number of point for FPS sampling
        :param radius:Radius for ball query
        :param nsample:Number of point for each ball query
        :param in_channel:the dimention of channel
        :param mlp:A list for mlp input-output channel, such as [64, 64, 128]
        :param group_all:bool type for group_all or not
        """
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        # moduleList是一个包含多个module的列表
        # 支持索引等列表操作
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            # 根据输入mlp设计卷积层和归一化层
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # B×N×C
        if points is not None:
            points = points.permute(0, 2, 1)  # B×N×D

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            # i是在mlp_convs中元素的下标，conv是对应的元素。
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]  # max pooling
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


# 密度自适应层：解决点云数据密度不均匀造成的问题
class PointNetSetAbstractionMsg(nn.Module):
    """
    Multi-Scale Grouping，多尺度分组
    对于同一个中心点，如果使用3个不同尺度的话，就分别找围
    绕每个中心点画3个区域，每个区域的半径及里面的点的个数不同。
    对于同一个中心点来说，不同尺度的区域送入不同的PointNet进行特征提取，
    之后concat，作为这个中心点的特征。也就是说MSG实际上相当于并联了多个
    hierarchical structure，每个结构中心点不变，但是区域范围不同。
    """
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            # 根据radius_list中的半径数据做多次ball query
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        # 将不同半径下产生的结果拼接起来
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    """
    对于点云分割任务，需要取得原始点的特征。
    这里用的方法是特征从子采样点(subsampled points)传播到原始点上。

    FeaturePropagation层的实现主要通过线性差值与MLP堆叠完成。当采样点的
    个数只有一个的时候，采用repeat直接复制成N个点；当点的个数大于一个的
    时候，采用线性差值的方式进行上采样，再对上采样后的每一个点都做一个MLP，
    同时拼接上下采样前相同点个数的SA层的特征。
    """
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)  # [B, N, C]
        xyz2 = xyz2.permute(0, 2, 1)  # [B, S, C]

        points2 = points2.permute(0, 2, 1)  # [B, S, D]
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
            # 插值点
        else:
            dists = square_distance(xyz1, xyz2)  # [B, N, S]
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
