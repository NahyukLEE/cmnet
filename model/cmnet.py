import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from scipy.spatial.transform import Rotation

from common.rotation import ortho2rotation
from chamfer_distance import ChamferDistance as chamfer_dist

from model.backbone.vn_dgcnn import EQCNN_UNet
from model.soft_attention import ChannelAttention

from model.backbone.vn_layers import VNLinear
from model.loss import CircleLoss, PointMatchingLoss, OrientationLoss
from model.learnable_sinkhorn import LearnableLogOptimalTransport
from model.local_global_registration import LocalGlobalRegistration

from einops import rearrange

from common.evaluator import Evaluator

class CMNet(pl.LightningModule):
    def __init__(self, lr, visualize=False):
        super(CMNet, self).__init__()

        self.lr = lr
        self.feat_dim = 1024

        # Feature Extractor
        self.backbone = EQCNN_UNet(feat_dim=self.feat_dim, pooling="mean")

        # Basis Vector
        self.proj = VNLinear(self.feat_dim//3, 2)

        # Channel Attention
        self.c_attn = ChannelAttention(1023, 1024, ratio=4)
        
        # Shape Descriptor
        self.shape_mlp = nn.Sequential(nn.Conv1d(1023, 512, kernel_size=1, bias=False),
                                nn.InstanceNorm1d(512),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                nn.InstanceNorm1d(512),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                nn.InstanceNorm1d(512),
                                nn.LeakyReLU(negative_slope=0.2),
                                )
        
        # Occupancy Descriptor
        self.occ_mlp = nn.Sequential(nn.Conv1d(1023, 512, kernel_size=1, bias=False),
                                nn.InstanceNorm1d(512),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                nn.InstanceNorm1d(512),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                nn.InstanceNorm1d(512),
                                nn.Tanh()
                                )
        
        # Optimal Transport
        self.optimal_transport = LearnableLogOptimalTransport(num_iterations=100)

        # Local to Global Registration
        self.fine_matching = LocalGlobalRegistration(
            k=3,
            acceptance_radius=0.1,
            mutual=True,
            confidence_threshold=0.05,
            use_dustbin=False,
            use_global_score=False,
            correspondence_threshold=3,
            correspondence_limit=None,
            num_refinement_steps=5,
        )
        
        # Objectives
        self.shape_loss = CircleLoss()
        self.occupancy_loss = CircleLoss()
        self.matching_loss = PointMatchingLoss()
        self.orientation_loss = OrientationLoss()

        # Weights for losses
        self.shp_loss_weight = 0.5 
        self.occ_loss_weight = 0.5
        self.mat_loss_weight = 1.0
        self.ori_loss_weight = 0.1

        self.visualize = visualize

        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        self.pairwise_evaluator = Evaluator()

    def configure_optimizers(self):
        """Build optimizer and lr scheduler."""
        lr = self.lr
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16919, eta_min=1e-3)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, in_dict, batch_idx):
        _, loss_dict = self.forward_pass(in_dict, mode='train')
        if loss_dict['loss']==0.: return None
        return loss_dict['loss']
    
    def validation_step(self, in_dict, batch_idx):
        _, loss_dict = self.forward_pass(in_dict, mode='val')
        self.validation_step_outputs.append(loss_dict)
        return loss_dict

    def on_validation_epoch_end(self):    
        # avg_loss among all data
        losses = {
            f'val/{k}': torch.stack([output[k] for output in self.validation_step_outputs])
            for k in self.validation_step_outputs[0].keys()
        }
        avg_loss = {k: (v).sum() / v.size(0) for k, v in losses.items()}

        self.log_dict(avg_loss, sync_dist=True, batch_size=1)
        self.validation_step_outputs.clear()

    def test_step(self, in_dict, batch_idx):
        _, loss_dict = self.forward_pass(in_dict, mode='test')
        self.test_step_outputs.append(loss_dict)
        return loss_dict

    def on_test_epoch_end(self):    
        # avg_loss among all data
        losses = {
            f'val/{k}': torch.stack([output[k] for output in self.test_step_outputs])
            for k in self.test_step_outputs[0].keys()
        }
        avg_loss = {k: (v).sum() / v.size(0) for k, v in losses.items()}
        # print('; '.join([f'{k}: {v.item():.6f}' for k, v in avg_loss.items()]))

        # Compute and print final results
        print('====PAIRWISE ASSEMBLY RESULTS====')
        print(f"CRD:     {avg_loss['val/crd'].item():.4f}")
        print(f"CD:      {avg_loss['val/cd'].item():.4f}")
        print(f"RMSE (R):   {avg_loss['val/rrmse'].item():.4f}")
        print(f"RMSE (T):   {avg_loss['val/trmse'].item():.4f}")
        self.test_results = avg_loss
        self.test_step_outputs.clear()

    def forward(self, in_dict, mode):

        out_dict = {}
        src_pcd = in_dict['pcd_t'][0] # (1, N ,3)
        trg_pcd = in_dict['pcd_t'][1] # (1, M ,3)
        
        # 1. SO(3)-Equivariant Feature Extractor
        src_equi_feats = self.backbone(src_pcd) # (1, 341, 3, N)
        trg_equi_feats = self.backbone(trg_pcd) # (1, 341, 3, M)

        # 2. Basis Vector Projection 
        src_vecs = self.proj(src_equi_feats).permute(0, 3, 1, 2) # (1, 341, 3, N) -> (1, 2, 3, N) -> (1, N, 2, 3)
        trg_vecs = self.proj(trg_equi_feats).permute(0, 3, 1, 2) # (1, 341, 3, M) -> (1, 2, 3, M) -> (1, M, 2, 3)

        # 3. Gram Schmidt & Cross-product
        src_ori = ortho2rotation(src_vecs) # (1, N, 2, 3) -> (1, N, 3, 3)
        trg_ori = ortho2rotation(trg_vecs) # (1, M, 2, 3) -> (1, M, 3, 3)

        # 4. Invariant Features
        src_inv_feats = torch.matmul(src_equi_feats.permute(0, 3, 1, 2), src_ori.transpose(-2,-1)) # (1, N, 341, 3) x (1, N, 3, 3) -> (1, N, 341, 3)
        trg_inv_feats = torch.matmul(trg_equi_feats.permute(0, 3, 1, 2), trg_ori.transpose(-2,-1)) # (1, M, 341, 3) x (1, M, 3, 3) -> (1, M, 341, 3)
        src_inv_feats = rearrange(src_inv_feats, 'b n c r -> b (c r) n') # (1, N, 341, 3) -> (1, 1023, N)
        trg_inv_feats = rearrange(trg_inv_feats, 'b n c r -> b (c r) n') # (1, M, 341, 3) -> (1, 1023, M)
        
        # 5. Chaneel Attention Map
        inv_feats = torch.cat([src_inv_feats, trg_inv_feats], dim=-1)  # (1, 1023, N+M)
        attention = self.c_attn(inv_feats) # (1, 1023, N+M) -> (1, 1023, N+M)
        shape_attention, occ_attention = attention[:, :512], attention[:, 512:]
        
        # 6. Shape descriptor
        src_shape_feats = self.shape_mlp(src_inv_feats) # (1, 1023, M) -> (1, 512, M)
        src_shape_feats = src_shape_feats * shape_attention
        trg_shape_feats = self.shape_mlp(trg_inv_feats) # (1, 1023, M) -> (1, 512, M)
        trg_shape_feats = trg_shape_feats * shape_attention

        # 7. Occupancy descriptor
        src_occ_feats = self.occ_mlp(src_inv_feats) # (1, 1023, N) -> (1, 512, N)
        src_occ_feats = src_occ_feats * occ_attention
        trg_occ_feats = self.occ_mlp(trg_inv_feats) # (1, 1023, M) -> (1, 512, M)
        trg_occ_feats = trg_occ_feats * occ_attention
        
        # 8. Optimal Transport
        shape_matching_scores = torch.einsum('b c n , b c m -> b n m', src_shape_feats, trg_shape_feats) # (1, N, M)
        shape_matching_scores = shape_matching_scores / src_shape_feats.shape[1] ** 0.5

        occ_matching_scores = -torch.einsum('b c n , b c m -> b n m', src_occ_feats, trg_occ_feats) # (1, N, M)
        occ_matching_scores = occ_matching_scores / src_occ_feats.shape[1] ** 0.5

        matching_scores = self.optimal_transport(shape_matching_scores + occ_matching_scores) # (1, N, M) -> (1, N+1, M+1)
        matching_scores_drop = matching_scores[:,:-1,:-1] # drop dustbin

        out_dict['src_ori'] = src_ori
        out_dict['trg_ori'] = trg_ori
        out_dict['src_shape_feats'] = src_shape_feats
        out_dict['trg_shape_feats'] = trg_shape_feats
        out_dict['src_occ_feats'] = src_occ_feats
        out_dict['trg_occ_feats'] = trg_occ_feats
        out_dict['matching_scores'] = matching_scores

        # 9. Transformation estimation with weighted SVD
        if mode in ['val', 'test']:
            with torch.no_grad():
                src_corr_pts, trg_corr_pts, corr_scores, estimated_transform, pred_corr = self.fine_matching(
                    src_pcd, trg_pcd, matching_scores_drop, k=128)

            out_dict['estimated_rotat'] = estimated_transform[:3, :3].T
            out_dict['estimated_trans'] = -(estimated_transform[:3, :3].inverse() @ -estimated_transform[:3, 3])
            out_dict['corr_scores'] = corr_scores
            out_dict['matching_scores_drop'] = matching_scores_drop
        
        return out_dict


    def forward_pass(self, in_dict, mode):

        loss = {}
        out_dict = self.forward(in_dict, mode)

        # Calculate Loss
        src_pcd_raw = in_dict['pcd'][0].squeeze(0)
        trg_pcd_raw = in_dict['pcd'][1].squeeze(0)
        gt_corr = in_dict['gt_correspondence'].squeeze(0)
        
        loss['shp_loss'] = self.shape_loss(src_pcd_raw, trg_pcd_raw, out_dict['src_shape_feats'].transpose(-2,-1), out_dict['trg_shape_feats'].transpose(-2,-1), gt_corr)
        loss['occ_loss'] = self.occupancy_loss(src_pcd_raw, trg_pcd_raw, out_dict['src_occ_feats'].transpose(-2,-1), -out_dict['trg_occ_feats'].transpose(-2,-1), gt_corr)
        loss['mat_loss'] = self.matching_loss(out_dict['matching_scores'], src_pcd_raw, trg_pcd_raw)
        loss['ori_loss'] = self.orientation_loss(out_dict['src_ori'], out_dict['trg_ori'], gt_corr, in_dict['gt_rotat'])

        loss['loss'] = self.shp_loss_weight * loss['shp_loss'] + self.occ_loss_weight * loss['occ_loss'] + self.mat_loss_weight * loss['mat_loss'] + self.ori_loss_weight * loss['ori_loss']
        out_dict.update(loss)
        
        # 10. Evaluation
        if mode in ['val', 'test']:
            eval_dict = self.evaluate_prediction(in_dict, out_dict, gt_corr)
            loss.update(eval_dict)

        # in training we log for every step
        if mode == 'train':
            log_dict = {f'{mode}/{k}': v.item() for k, v in loss.items()}
            self.log_dict(log_dict, logger=True, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True, batch_size=1)
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', lr, prog_bar=True, logger=True)

        return out_dict, loss

    @torch.no_grad()
    def evaluate_prediction(self, in_dict, out_dict, gt_corr):

        # Init return buffer
        eval_result = {}
        
        pred_relative_trsfm = out_dict['estimated_rotat'], out_dict['estimated_trans'] 
        grtr_relative_trsfm = [x.squeeze(0) for x in in_dict['relative_trsfm']['0-1']]
        src_pcd, trg_pcd = [x.squeeze(0) for x in in_dict['pcd_t']]

        # Assemble using prediction, pseudo-gt, and ground-truth
        assm_pred, pcds_pred = self.pairwise_evaluator.pairwise_mating(src_pcd, trg_pcd, pred_relative_trsfm[0], pred_relative_trsfm[1])
        assm_grtr, pcds_grtr = self.pairwise_evaluator.pairwise_mating(src_pcd, trg_pcd, grtr_relative_trsfm[0], grtr_relative_trsfm[1])
        
        # (a) Compute CD between prediction & ground-truth
        eval_result['cd'] = self.pairwise_evaluator.compute_cd(assm_pred, assm_grtr)

        # (b) Compute MSE between prediction & ground-truth for rotation (in degree) and translation
        eval_result['rrmse'], eval_result['trmse'] = self.pairwise_evaluator.compute_trsfm_error(pred_relative_trsfm, grtr_relative_trsfm)

        # (c) Compute CoRrespondence Distance (CRD) betwween prediction & ground-truth
        eval_result['crd'] = self.pairwise_evaluator.compute_crd(assm_pred, assm_grtr)

        if self.visualize:
            pass

        return eval_result