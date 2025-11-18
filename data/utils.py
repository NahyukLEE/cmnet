import numpy as np
import open3d as o3d
import torch

def to_o3d_pcd(pts):
    '''
    From numpy array, make point cloud in open3d format
    :param pts: point cloud (nx3) in numpy array
    :return: pcd: point cloud in open3d format
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def get_correspondences(src_pcd, tgt_pcd, search_voxel_size, K=None):
    '''
    Give source & target point clouds as well as the relative transformation between them, calculate correspondences according to give threshold
    :param src_pcd: source point cloud
    :param tgt_pcd: target point cloud
    :param search_voxel_size: given threshold
    :param K: if k is not none, select top k nearest neighbors from candidate set after radius search
    :return: (m, 2) torch tensor, consisting of m correspondences
    '''

    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences

def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor

def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.pipelines.registration.Feature()
    feats.data = to_array(embedding).T
    return feats