#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset

class DataSet_GS(Dataset):
    """
    Load in 3D medical image, treate image as a stack of 2D images with given dimension
    """
    def __init__(self, input_image, groundtruth, transform = None):        
        super(DataSet_GS, self).__init__()
        self.input_image = input_image
        self.groundtruth = groundtruth
        
        
    def __len__(self):
        return np.shape(self.input_image)[0]
    
    def __getitem__(self, idx):
        im_sample = self.input_image[idx, :,...]
        gt_sample = self.groundtruth[idx]
        sample = {'input_image': im_sample, 'gt': gt_sample}
        return sample



