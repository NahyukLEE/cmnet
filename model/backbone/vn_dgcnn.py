import torch
import torch.nn as nn
from model.backbone.vn_layers import knn, get_graph_feature, mean_pool
from model.backbone.vn_layers import VNLinearLeakyReLU, VNMaxPool
from model.backbone.vn_layers import VNLinear

from lib.pointops.functions import pointops


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=4):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.mlp = VNLinearLeakyReLU(in_planes, out_planes)
        else:
            self.mlp = VNLinearLeakyReLU(in_planes, out_planes)
        
    def forward(self, p, x):
        o = torch.Tensor([p.size(0)]).to(torch.int32).cuda()
        n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
        n_o = torch.cuda.IntTensor(n_o)

        # FPS
        idx = pointops.furthestsampling(p, o, n_o)  # (m)
        n_p = p[idx.long(), :]  # (m, 3)

        # kNN-MLP
        x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=False)
        x = self.mlp(x.permute(2,3,0,1).unsqueeze(0))

        # Mean Pooling
        x = x.mean(dim=-1)  # (1, c, 3, m)
        return n_p, x, n_o

class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.mlp1 = VNLinearLeakyReLU(out_planes, out_planes, dim=4)
        self.mlp2 = VNLinearLeakyReLU(in_planes, out_planes, dim=4)
        
    def forward(self, pxo1, pxo2):
        p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
        self.mlp1(x1) 
        x = self.mlp1(x1) + pointops.interpolation(p2, p1, self.mlp2(x2).squeeze(0).contiguous(), o2, o1)
        return x


class EQCNN_UNet(nn.Module):

    def __init__(self, feat_dim, pooling='mean'):
        super(EQCNN_UNet, self).__init__()
        self.k = 20

        if pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(128//3)
            self.pool4 = VNMaxPool(256//3)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool
            self.pool5 = mean_pool
            self.pool6 = mean_pool
            self.pool7 = mean_pool
            self.pool8 = mean_pool
        
        # Encoder
        self.conv1 = VNLinearLeakyReLU(2, 64//3)

        self.downsample1 = TransitionDown(64//3, 64//3, stride=2, nsample=16)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3)

        self.downsample2 = TransitionDown(128//3, 128//3, stride=2, nsample=16)
        self.conv3 = VNLinearLeakyReLU(128//3*2, 256//3)

        self.downsample3 = TransitionDown(256//3, 256//3, stride=2, nsample=16)
        self.conv4 = VNLinearLeakyReLU(256//3*2, 512//3)

        self.conv5 = VNLinearLeakyReLU(512//3*2, 512//3)
    
        # Decoder
        self.upsample1 = TransitionUp(512//3, 256//3)
        self.conv6 = VNLinearLeakyReLU(256//3*2, 256//3)

        self.upsample2 = TransitionUp(256//3, 128//3)
        self.conv7 = VNLinearLeakyReLU(127//3*2, 128//3)

        self.upsample3 = TransitionUp(128//3, 64//3)
        self.conv8 = VNLinearLeakyReLU(64//3*2, 64//3)

        # Proj
        self.conv9 = VNLinearLeakyReLU(64//3, feat_dim//3, dim=4, share_nonlinearity=True)

    def forward(self, x):
        x = x.transpose(2, 1) # (batch_size, 3, num_points)
        batch_size = x.size(0)
        num_points = x.size(2)

        p1 = x.transpose(1,2).squeeze(0)
        x1 = x.unsqueeze(1) # (batch_size, 1, 3, num_points)
        o1 = torch.Tensor([p1.size(0)]).to(torch.int32).cuda()
        
        ### ENCODER 1
        x1 = get_graph_feature(x1, k=self.k) # torch.Size([1, 2, 3, 2556, 20])
        x1 = self.conv1(x1)
        x1 = self.pool1(x1) # (1, 21, 3, N)

        ### ENCODER 2
        p2, x2, o2 = self.downsample1(p1, x1.squeeze(0))
        x2 = get_graph_feature(x2, k=self.k)
        x2 = self.conv2(x2)
        x2 = self.pool2(x2) # (1, 42, 3, N/2)

        ### ENCODER 3
        p3, x3, o3 = self.downsample2(p2, x2.squeeze(0))
        x3 = get_graph_feature(x3, k=self.k)
        x3 = self.conv3(x3)
        x3 = self.pool3(x3) # (1, 85, 3, N/4)
        
        ### ENCODER 4
        p4, x4, o4 = self.downsample3(p3, x3.squeeze(0))
        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv4(x4)
        x4 = self.pool4(x4) # (1, 170, 3, N/8)

        x4 = get_graph_feature(x4, k=self.k)
        x4 = self.conv5(x4)
        x4 = self.pool5(x4) # (1, 170, 3, N/8)

        ### DECODER 1
        x5 = self.upsample1((p3, x3, o3), (p4, x4, o4)) 
        x5 = get_graph_feature(x5.contiguous(), k=self.k)
        x5 = self.conv6(x5)
        x5 = self.pool6(x5) # (1, 85, 3, N/4)

        ### DECODER 2
        x6 = self.upsample2((p2, x2, o2), (p3, x5, o3))
        x6 = get_graph_feature(x6.contiguous(), k=self.k)
        x6 = self.conv7(x6)
        x6 = self.pool7(x6) # (1, 42, 3, N/2)

        ### DECODER 3
        x7 = self.upsample3((p1, x1, o1), (p2, x6, o2))
        x7 = get_graph_feature(x7.contiguous(), k=self.k)
        x7 = self.conv8(x7)
        x7 = self.pool8(x7) # (1, 21, 3, N)

        equi_feat = self.conv9(x7)

        return equi_feat

class EQCNN_equi(nn.Module):

    def __init__(self, feat_dim, pooling='mean'):
        super(EQCNN_equi, self).__init__()
        self.k = 20

        if pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(128//3)
            self.pool4 = VNMaxPool(256//3)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 128//3)
        self.conv4 = VNLinearLeakyReLU(128//3*2, 256//3)

        self.conv5 = VNLinearLeakyReLU(256//3+128//3+64//3*2, feat_dim//3, dim=4, share_nonlinearity=True)

    def forward(self, x):
        x = x.transpose(2, 1) # (batch_size, 3, num_points)
        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # (batch_size, 1, 3, num_points)

        x = get_graph_feature(x, k=self.k) # (1, 2, 3, num_points, k)
        x = self.conv1(x)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = self.pool3(x)
        
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = self.pool4(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        equi_feat = self.conv5(x) # (batch_size, feat_dim//3, num_points)
        
        return equi_feat