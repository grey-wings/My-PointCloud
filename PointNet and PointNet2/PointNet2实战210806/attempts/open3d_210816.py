import open3d as o3d
import numpy as np

if __name__ == "__main__":
    nparr = np.loadtxt('../data/modelnet40_normal_resampled/airplane/airplane_0001.txt',
                       delimiter=',')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nparr[:, 0:3])
    pcd.normals = o3d.utility.Vector3dVector(nparr[:, 3:])
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([pcd], width=1080, height=608)

