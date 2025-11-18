import argparse
import torch
import pytorch_lightning as pl

from model.cmnet import CMNet
from data.dataset import GADataset
from common import utils

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars", category=RuntimeWarning)

@torch.no_grad()
def test(args):
    
    # Model initialization
    utils.fix_randseed(0)
    model = CMNet(args.lr, visualize=args.visualize)
    model.to(torch.device('cuda:0'))
    model.eval()
    
    # Dataset initialization
    GADataset.initialize(args.datapath, args.data_category, args.sub_category, args.n_pts, mpa=False)
    dataloader_test = GADataset.build_dataloader(args.batch_size, args.n_worker, 'test')

    # Testing
    trainer = pl.Trainer(accelerator='gpu', devices=[0])
    trainer.test(model, dataloader_test, ckpt_path=args.load)
    results = model.test_results
    results = {k[5:]: v.detach().cpu().numpy() for k, v in results.items()}
    print('Done testing...')

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