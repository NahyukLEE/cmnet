r""" Helper functions """
import random

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def save_pc(filename: str, pcd_tensors: list):
    colors = [
        [1, 0.996, 0.804],
        [0.804, 0.98, 1],
        [1, 0.376, 0],
        [0, 0.055, 1]
    ]
    
    pcds = []
    for i, tensor_ in enumerate(pcd_tensors):
        if tensor_.size()[0] == 1:
            tensor_ = tensor_.squeeze(0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tensor_.cpu().numpy())
        pcd.paint_uniform_color(colors[i % len(colors)])  # Assign color based on index
        pcds.append(pcd)
    
    combined_cloud = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined_cloud += pcd
    
    o3d.io.write_point_cloud(filename, combined_cloud)
    
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, dict):
            # continue
            for k, v in value.items():
                if isinstance(v[0], torch.Tensor):
                    value[k] = [v_.cuda() for v_ in v]
        elif isinstance(value[0], torch.Tensor):
            batch[key] = [v.cuda() for v in value]
    batch['filepath'] = batch['filepath'][0]
    batch['obj_class'] = batch['obj_class'][0]
    batch['gt_correspondence'] = batch['gt_correspondence'][0]

    if batch.get('n_frac') is not None: batch['n_frac'] = batch['n_frac'][0]
    if batch.get('order') is not None: batch['order'] = batch['order'][0]
    if batch.get('anchor_idx') is not None: batch['anchor_idx'] = batch['anchor_idx'][0]

    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()