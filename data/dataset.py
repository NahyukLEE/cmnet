import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data.breakingbad import DatasetBreakingBad

class GADataset:

    @classmethod
    def initialize(cls, datapath, data_category, sub_category, n_pts, mpa=False):
        cls.datapath = datapath
        cls.data_category = data_category
        cls.sub_category = sub_category
        cls.n_pts = n_pts
        cls.mpa = mpa

    @classmethod
    def build_dataloader(cls, batch_size, nworker, split, visualize=False):
        training = split == 'train'
        shuffle = training

        dataset = DatasetBreakingBad(cls.datapath, cls.data_category, cls.sub_category, cls.n_pts, cls.mpa, split, visualize)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=nworker, pin_memory=False)
        return dataloader