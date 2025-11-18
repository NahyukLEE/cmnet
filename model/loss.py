import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class CircleLoss(nn.Module):

    def __init__(self, log_scale=16, pos_optimal=0.1, neg_optimal=1.4):
        super(CircleLoss,self).__init__()
        self.log_scale = 24
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal

        self.pos_margin = 0.1
        self.neg_margin = 1.4
        
        self.pos_radius = 0.018
        self.safe_radius = 0.03

        self.max_points = 128

    def get_circle_loss(self, coords_dist, feats_dist):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        """

        pos_mask = coords_dist < self.pos_radius
        neg_mask = coords_dist > self.safe_radius

        # get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)).detach()
        col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)).detach()

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive 
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach() 

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight).detach()

        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-1)
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-2)

        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-1)
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-2)

        loss_row = F.softplus(lse_pos_row + lse_neg_row)/self.log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col)/self.log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

        return circle_loss


    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence):
        if len(correspondence) == 0:
            return torch.tensor(0.).to(src_feats.device)

        c_dist = torch.norm(src_pcd[correspondence[:,0]] - tgt_pcd[correspondence[:,1]], dim = 1)
        c_select = c_dist < self.pos_radius - 0.001
        correspondence = correspondence[c_select]
        
        if correspondence.size(0) > self.max_points:
            choice = np.random.permutation(correspondence.size(0))[:self.max_points]
            correspondence = correspondence[choice]

        # Use only correspondence points
        src_idx = correspondence[:,0]
        tgt_idx = correspondence[:,1]
        src_pcd, tgt_pcd = src_pcd[src_idx], tgt_pcd[tgt_idx]
        src_feats, tgt_feats = src_feats[:, src_idx, :], tgt_feats[:, tgt_idx, :]

        src_feats = F.normalize(src_feats.squeeze(0), p=2, dim=-1)
        tgt_feats = F.normalize(tgt_feats.squeeze(0), p=2, dim=-1)

        # Get coordinate distance
        coords_dist = torch.sqrt(torch.sum((src_pcd[:, None, :] - tgt_pcd[None, :, :]) ** 2, dim=-1))
        feats_dist = (2.0 - 2.0 * torch.einsum('x d, y d -> x y', src_feats, tgt_feats)).pow(0.5)
        
        circle_loss = self.get_circle_loss(coords_dist, feats_dist)
        
        if circle_loss != circle_loss:
            circle_loss = torch.tensor(0.).to(src_feats.device)
            
        return circle_loss

class PointMatchingLoss(nn.Module):
    def __init__(self):
        super(PointMatchingLoss, self).__init__()
        self.positive_radius = 0.018

    def forward(self, matching_scores, src_pcd, trg_pcd):
        coords_dist = torch.sqrt(torch.sum((src_pcd[:, None, :] - trg_pcd[None, :, :]) ** 2, dim=-1))
        gt_corr_map = coords_dist < self.positive_radius

        # Initialize labels for the loss calculation
        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        
        # Handle slack rows and columns
        slack_row_labels = torch.sum(gt_corr_map[:, :-1], dim=1) == 0
        slack_col_labels = torch.sum(gt_corr_map[:-1, :], dim=0) == 0

        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels
        
        pmat_loss = -matching_scores[labels].mean()

        return pmat_loss

class OrientationLoss(nn.Module):
    def __init__(self):
        super(OrientationLoss, self).__init__()

    def forward(self, src_ori, trg_ori, correspondence, gt_rot):
        if len(correspondence) == 0:
            return torch.tensor(0.).to(src_ori.device)

        src_gt_rot, trg_gt_rot = gt_rot
        src_ori = src_ori[:, correspondence[:,0]]
        trg_ori = trg_ori[:, correspondence[:,1]]

        src_ori = torch.matmul(src_ori, src_gt_rot)
        trg_ori = torch.matmul(trg_ori, trg_gt_rot)

        diff = src_ori - trg_ori
        f_norm = torch.norm(diff, p='fro', dim=(2, 3))
        ori_loss = torch.mean(f_norm)

        return ori_loss