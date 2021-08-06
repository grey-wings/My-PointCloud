import argparse
import os
import time
from tqdm import tqdm


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


# root = "E:\\SME\\SME-数据\\大创-点云\\点云数据集\\modelnet40_normal_resampled"
# catfile = os.path.join(root, 'modelnet10_shape_names.txt')
# shape_ids = {}
# split = "train"
# shape_ids['train'] = [line.rstrip() for line in open(os.path.join(root, 'modelnet10_train.txt'))]
# assert (split == 'train' or split == 'test')
# shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
# datapath = [(shape_names[i], os.path.join(root, shape_names[i], shape_ids[split][i]) + '.txt') for i
#             in range(len(shape_ids[split]))]
# file = datapath[0]
# print(file)

# arg = parse_args()
# print(arg)
