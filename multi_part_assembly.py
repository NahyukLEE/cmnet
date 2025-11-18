import os
import sys
import argparse
import gc
import itertools

import torch
import numpy as np
import trimesh

import gtsam
from scipy.spatial.transform import Rotation

from model.cmnet import CMNet
from data.dataset import GADataset

from common import utils
from common.evaluator import Evaluator

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars", category=RuntimeWarning)

def estimate_poses_given_rot(
        factors: gtsam.BetweenFactorPose3s,
        rotations: gtsam.Values,
        uncertainty,
        anchor_idx
):
    """Estimate Poses from measurements, given rotations. From SfmProblem in shonan.
    Source: https://github.com/Jiaxin-Lu/Jigsaw/blob/41713e196f1b6b294913665945cd22240127c80b/utils/global_alignment/shonan_averaging.py#L6
    Arguments:
        factors -- data structure with many BetweenFactorPose3 factors
        rotations {Values} -- Estimated rotations
    Returns:
        Values -- Estimated Poses
    """

    graph = gtsam.GaussianFactorGraph()
    model = gtsam.noiseModel.Unit.Create(3)

    # Add a factor anchoring t_anchor
    graph.add(anchor_idx, np.eye(3), np.zeros((3,)), model)

    # Add a factor saying t_j - t_i = Ri*t_ij for all edges (i,j)
    for idx in range(len(factors)):
        factor = factors[idx]
        keys = factor.keys()
        i, j, Tij = keys[0], keys[1], factor.measured()
        if i == j:
            continue
        model = gtsam.noiseModel.Diagonal.Variances(
            uncertainty[idx] * (1e-2) * np.ones(3)
        )
        measured = rotations.atRot3(j).inverse().rotate(Tij.translation())
        graph.add(j, np.eye(3), i, -np.eye(3), measured, model)

    # Solve linear system
    translations = graph.optimize()
    # Convert to Values.
    result = gtsam.Values()
    for j in range(rotations.size()):
        tj = translations.at(j)
        result.insert(j, gtsam.Pose3(rotations.atRot3(j), tj))

    return result

def test(args):
    # Model initialization
    utils.fix_randseed(0)
    model = CMNet(args.lr, visualize=args.visualize)

    model.to(torch.device('cuda:0'))
    model.eval()

    # Load checkpoint
    if args.load:
        print(f"Loading checkpoint from {args.load}")
        checkpoint = torch.load(args.load, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("No checkpoint specified. Exiting program.")
        sys.exit(1)
    
    # Dataset initialization
    GADataset.initialize(args.datapath, args.data_category, args.sub_category, args.n_pts, mpa=True)
    dataloader_test = GADataset.build_dataloader(args.batch_size, args.n_worker, 'test')
    
    evaluator = Evaluator(mpa=True)
    total = len(dataloader_test)
    
    # Organize metrics in a dictionary
    metrics = {
        'crd': [], 'cd': [], 'rrmse': [], 'trmse': [], 
        'pa': [], 'pa_crd': []
    }

    for idx, in_dict in enumerate(dataloader_test):
        # 1. Network forward pass: Pairwise matching & assembly
        in_dict = utils.to_cuda(in_dict)
        pair_indices = list(itertools.permutations([i for i in range(in_dict['n_frac'])], 2))
        out_dict = {}

        with torch.no_grad():
            for pair_idx in pair_indices:
                # Initialize pairwise input
                pair_idx0, pair_idx1 = pair_idx
                in_dict_pair = {
                    'filepath': in_dict['filepath'], 'obj_class': in_dict['obj_class'],
                    'pcd_t': [in_dict['pcd_t'][pair_idx0], in_dict['pcd_t'][pair_idx1]],
                    'pcd': [in_dict['pcd'][pair_idx0], in_dict['pcd'][pair_idx1]],
                    'n_frac': 2, # Pairwise matching
                    'gt_trans': [in_dict['gt_trans'][pair_idx0], in_dict['gt_trans'][pair_idx1]],
                    'gt_rotat': [in_dict['gt_rotat'][pair_idx0], in_dict['gt_rotat'][pair_idx1]],
                    'gt_rotat_inv': [in_dict['gt_rotat_inv'][pair_idx0], in_dict['gt_rotat_inv'][pair_idx1]],
                    'gt_trans_inv': [in_dict['gt_trans_inv'][pair_idx0], in_dict['gt_trans_inv'][pair_idx1]],
                    'relative_trsfm': {f'{pair_idx0}-{pair_idx1}': in_dict['relative_trsfm'][f'{pair_idx0}-{pair_idx1}']},
                }
                # Forward pass
                out_dict[f'{pair_idx0}-{pair_idx1}'] = model(in_dict_pair, mode='test')
                torch.cuda.empty_cache(); gc.collect()
        
        # 2. Pose Graph Optimization
        params = gtsam.ShonanAveragingParameters3(gtsam.LevenbergMarquardtParams.CeresDefaults())
        factors = gtsam.BetweenFactorPose3s()
        uncertainty = []

        
        # # 2-1. Add factors(relative transformations)
        for i in range(in_dict['n_frac'].item()):
            # search highest score
            max_score = 0
            for j in range(in_dict['n_frac'].item()):
                if i == j: continue
                value = torch.exp(out_dict[f'{j}-{i}']['matching_scores_drop']).sum()
                if max_score < value:
                    max_score = value
                    max_idx = j
            relative_rotat = Rotation.from_matrix(out_dict[f'{i}-{max_idx}']['estimated_rotat'].cpu().numpy()).as_quat()
            relative_trans = out_dict[f'{i}-{max_idx}']['estimated_trans'].cpu().numpy()

            max_score = max_score.detach().cpu()

            # add factor
            pose = gtsam.Pose3(gtsam.Rot3.Quaternion(relative_rotat[3], relative_rotat[0], relative_rotat[1], relative_rotat[2]), gtsam.Point3(relative_trans))
            factors.append(gtsam.BetweenFactorPose3(i, max_idx, pose, gtsam.noiseModel.Diagonal.Information((1/max_score) * np.eye(6))))
            uncertainty.append(1/max_score)
        
        # 2-3. Run shonan averaging
        sa3 = gtsam.ShonanAveraging3(factors, params)
        initial = sa3.initializeRandomly()
        pMax = 20
        shonan_fail = False
        while True:
            pMax += 20
            if pMax == 60:
                shonan_fail = True; print("shonan failed")
                break
            try: 
                abs_rotat, _ = sa3.run(initial, 3, pMax)
                break
            except RuntimeError as e:
                print(f"An error occurred during Shonan::run: with pMax {pMax}")
                continue

        # Align predicted rotation to anchor fracture
        anchor_idx = in_dict['anchor_idx']
        if not shonan_fail:
            aligned_pred_rotat, aligned_pred_trans = [], []

            aligned_pred_rotat2 = []
            abs_anchor_R2 = abs_rotat.atRot3(anchor_idx)
            
            for j in range(abs_rotat.size()):
                aligned_pred_rotat2.append(torch.tensor(abs_anchor_R2.between(abs_rotat.atRot3(j)).matrix()).to(torch.float32).cuda())

            rel_rotat = gtsam.Values()
            abs_anchor_R = abs_rotat.atRot3(anchor_idx)
            for i in range(abs_rotat.size()):
                if i == anchor_idx:
                    rel_rotat.insert(i, gtsam.Rot3(np.eye(3)))
                else:
                    rel_rotat.insert(i, abs_anchor_R.inverse().compose(abs_rotat.atRot3(i)))

            poses = estimate_poses_given_rot(
                factors, rel_rotat, np.array(uncertainty), anchor_idx
            )
            abs_anchor_T = poses.atPose3(anchor_idx).translation()

            for j in range(poses.size()):
                aligned_pred_rotat.append(
                    torch.tensor(abs_anchor_R.between(abs_rotat.atRot3(j)).matrix()).to(torch.float32).cuda()
                )
                pred_trans = poses.atPose3(j).rotation().rotate(abs_anchor_T + poses.atPose3(j).translation())
                aligned_pred_trans.append(torch.tensor(pred_trans).to(torch.float32).cuda())
        else:
            aligned_pred_rotat, aligned_pred_trans = [], []
            for i in range(0, in_dict['n_frac']):
                if i == anchor_idx: 
                    aligned_pred_rotat.append(torch.eye(3).to(torch.float32).cuda())
                    aligned_pred_trans.append(torch.tensor([0,0,0]).to(torch.float32).cuda())
                else: 
                    aligned_pred_rotat.append(out_dict[f'{anchor_idx}-{i}']['estimated_rotat'].squeeze(0))
                    aligned_pred_trans.append(out_dict[f'{anchor_idx}-{i}']['estimated_trans'])
        
        # Align GT transformation to anchor fracture
        aligned_gt_rotat, aligned_gt_trans = [], []
        for i in range(0, in_dict['n_frac']):
            if i == anchor_idx:
                aligned_gt_rotat.append(torch.eye(3).to(torch.float32).cuda())
                aligned_gt_trans.append(torch.tensor([0,0,0]).to(torch.float32).cuda())
            else:
                aligned_gt_rotat.append(in_dict['relative_trsfm'][f'{anchor_idx}-{i}'][0].squeeze(0))
                aligned_gt_trans.append(in_dict['relative_trsfm'][f'{anchor_idx}-{i}'][1].squeeze(0))

        # Save aligned transformations
        in_dict['aligned_gt_trans'] = aligned_gt_trans
        in_dict['aligned_gt_rotat'] = aligned_gt_rotat
        out_dict['aligned_pred_trans'] = aligned_pred_trans
        out_dict['aligned_pred_rotat'] = aligned_pred_rotat

        assm_pred, pcds_pred = evaluator.multi_part_assemble(in_dict['pcd_t'], aligned_pred_rotat, aligned_pred_trans)
        assm_grtr, pcds_grtr = evaluator.multi_part_assemble(in_dict['pcd_t'], aligned_gt_rotat, aligned_gt_trans)

        cd = evaluator.compute_cd(assm_pred, assm_grtr).item()
        crd = evaluator.compute_crd(assm_pred, assm_grtr).item()
        rrmse, trmse = evaluator.compute_trsfm_error((aligned_pred_rotat, aligned_pred_trans), (aligned_gt_rotat, aligned_gt_trans))
        rrmse, trmse = rrmse.item(), trmse.item()
        pa = evaluator.compute_pa(pcds_pred, pcds_grtr)
        pa_crd = evaluator.compute_pa_crd(pcds_pred, pcds_grtr)

        if args.visualize:
            pass
        
        # Accumulate metrics
        metrics['crd'].append(crd)
        metrics['cd'].append(cd)
        metrics['rrmse'].append(rrmse)
        metrics['trmse'].append(trmse)
        metrics['pa'].append(pa)
        metrics['pa_crd'].append(pa_crd)

        result_str = f'{idx+1}/{total} | #-Part: {len(pcds_pred)} | CRD: {round(crd,2)} | CD: {round(cd,2)} | RRMSE: {round(rrmse,2)} | TRMSE: {round(trmse,2)} | PA(cd): {round(pa,2)} | PA(crd): {round(pa_crd,2)}'
        print(result_str)

    # Compute and print final results
    print('====MULTI PART ASSEMBLY RESULTS====')
    print(f"CRD:     {np.mean(metrics['crd']):.2f}")
    print(f"CD:      {np.mean(metrics['cd']):.2f}")
    print(f"RMSE (R):   {np.mean(metrics['rrmse']):.2f}")
    print(f"RMSE (T):   {np.mean(metrics['trmse']):.2f}")
    print(f"PA(CRD): {np.mean(metrics['pa_crd']):.1f}")
    print(f"PA(CD):  {np.mean(metrics['pa']):.1f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combinative Matching Network (CMNet) Pytorch Lightning Implementation')
    parser.add_argument('--datapath', type=str, default='../../data/bbad_v2')
    parser.add_argument('--data_category', type=str, default='everyday', choices=['everyday', 'artifact'])
    parser.add_argument('--sub_category', type=str, default='all')
    parser.add_argument('--n_pts', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)

    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--load', type=str, default='')

    parser.add_argument('--visualize', action='store_true')
      
    args = parser.parse_args()

    test(args)