import torch
from scipy.spatial.transform import Rotation
from chamfer_distance import ChamferDistance as chamfer_dist

class Evaluator:

    def __init__(self, mpa=False):
        self.mpa = mpa

    def is_trg_larger(self, src_pcd, trg_pcd):
        src_volume = (src_pcd.max(dim=0)[0] - src_pcd.min(dim=0)[0]).prod(dim=0)
        trg_volume = (trg_pcd.max(dim=0)[0] - trg_pcd.min(dim=0)[0]).prod(dim=0)

        return src_volume < trg_volume

    def transform(self, pcd, rotat=None, trans=None, rotate_first=True):
        if rotat == None: rotat = torch.eye(3, 3)
        if trans == None: trans = torch.zeros(3)

        rotat = rotat.to(pcd.device)
        trans = trans.to(pcd.device)

        if rotate_first:
            return torch.einsum('x y, n y -> n x', rotat, pcd) + trans
        else:
            return torch.einsum('x y, n y -> n x', rotat, pcd + trans)
    
    def pairwise_mating(self, src_pcd, trg_pcd, rotat, trans):
        pcd_t = []
        is_trg_larger = self.is_trg_larger(src_pcd, trg_pcd)

        if is_trg_larger:
            src_pcd_t = self.transform(src_pcd.squeeze(0), rotat, -trans, True)
            pcd_t = [src_pcd_t, trg_pcd.squeeze(0)]
        else:
            trg_pcd_t = self.transform(trg_pcd.squeeze(0), rotat.T, trans, False)
            pcd_t = [src_pcd.squeeze(0), trg_pcd_t]
        return torch.cat(pcd_t, dim=0), pcd_t

    def multi_part_assemble(self, pcds, rotat, trans):
        pcd_t = []
        for pcd, R, t in zip(pcds, rotat, trans):
            pcd_t.append(self.transform(pcd.squeeze(0), R.inverse(), t, False))
        return torch.cat(pcd_t, dim=0), pcd_t

    def compute_cd(self, assm1, assm2, scaling=1000):
        chd = chamfer_dist()
        dist1, dist2, idx1, idx2 = chd(assm1.unsqueeze(0), assm2.unsqueeze(0))
        cd = (dist1.mean(dim=-1) + dist2.mean(dim=-1)) * scaling
        return cd
    
    def compute_crd(self, assm1, assm2, scaling=100):
        corr_dist = (assm1 - assm2).norm(dim=-1).mean(dim=-1) * scaling
        return corr_dist
    
    def compute_trsfm_error(self, trnsf1, trnsf2, trmse_scaling=100):
        if self.mpa:
            rotat1, trans1 = trnsf1
            rotat2, trans2 = trnsf2
        else:
            rotat1, trans1 = [trnsf1[0]], [trnsf1[1]]
            rotat2, trans2 = [trnsf2[0]], [trnsf2[1]]
        
        rrmse, trmse = 0., 0.
        for r1, r2, t1, t2 in zip(rotat1, rotat2, trans1, trans2):
            r1_deg = torch.tensor(Rotation.from_matrix(r1.cpu()).as_euler('xyz', degrees=True))
            r2_deg = torch.tensor(Rotation.from_matrix(r2.cpu()).as_euler('xyz', degrees=True))
            diff1 = (r1_deg - r2_deg).abs()
            diff2 = 360. - (r1_deg - r2_deg).abs()
            diff = torch.minimum(diff1, diff2)
            rrmse += diff.pow(2).mean().pow(0.5)
            trmse += (t1 - t2).pow(2).mean().pow(0.5) * trmse_scaling
        div = len(rotat1) if self.mpa else 1
        return rrmse / div, trmse / div
    
    def compute_pa(self, assm_pts1, assm_pts2, scaling=100):
        success = 0
        for pred_pts, gt_pts in zip(assm_pts1, assm_pts2):
            part_cd = self.compute_cd(pred_pts, gt_pts, 1)
            if part_cd < 0.01: success += 1
        return success / len(assm_pts1) * scaling
    
    def compute_pa_crd(self, assm_pts1, assm_pts2, scaling=100):
        success = 0
        for pred_pts, gt_pts in zip(assm_pts1, assm_pts2):
            part_crd = self.compute_crd(pred_pts, gt_pts, 1)
            if part_crd < 0.1: success += 1
        return success / len(assm_pts1) * scaling
    