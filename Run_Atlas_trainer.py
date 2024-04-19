from os import PathLike
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import os, glob
import json
import subprocess
import sys
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from easydict import EasyDict as edict
import random
import yaml
from losses import NCC, MSE, Grad
from networks import UnetDense  
from SitkDataSet import SitkDataset as SData
from uEpdiff import Epdiff
from networks import *


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

################Parameter Loading#######################
def read_yaml(path):
    try:
        with open(path, 'r') as f:
            file = edict(yaml.load(f, Loader=yaml.FullLoader))
        return file
    except:
        print('NO FILE READ!')
        return None


##################Data Loading##########################
def load_and_preprocess_data(data_dir, json_file, keyword):
    readfilename = f'{data_dir}/{json_file}.json'
    try:
        with open(readfilename, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f'Error loading JSON data: {e}')
        return None
    outputs = []
    temp_scan = sitk.GetArrayFromImage(sitk.ReadImage(f'{data_dir}/{data[keyword][0]["image"]}'))
    xDim, yDim, zDim = temp_scan.shape
    return xDim, yDim, zDim


def initialize_network_optimizer(xDim, yDim, zDim, para, dev):
    net = UnetDense(inshape=(xDim, yDim, zDim),
                      nb_unet_features=[[16, 32,32], [ 32, 32, 32, 16, 16]], #[16, 32,32], [ 32, 32, 32, 16, 16]
                      nb_unet_conv_per_level=1,
                      int_steps=7,
                      int_downsize=2,
                      src_feats=1,
                      trg_feats=1,
                      unet_half_res=True)
    net = net.to(dev)

    if para.model.loss == 'L2':
        criterion = nn.MSELoss()
    elif para.model.loss == 'L1':
        criterion = nn.L1Loss()
    if para.model.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=para.solver.lr)
    elif para.model.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=para.solver.lr, momentum=0.9)

    return net, criterion, optimizer


def train_network(trainloader, aveloader, net, para, criterion, optimizer, DistType, RegularityType, weight_dist, weight_reg,  reduced_xDim, reduced_yDim, reduced_zDim, xDim, yDim, zDim, dev):
    """
    Train the network.

    Parameters:
    trainloader (DataLoader): DataLoader for training data.
    aveloader (DataLoader): DataLoader for average scans or random scans to intial atlas.
    net: The neural network model.
    para: Parameters object.
    criterion: The loss criterion.
    optimizer: The optimizer.
    DistType: Type of distance.
    RegularityType: Type of regularity.
    weight_dist: Weight for distance loss.
    weight_reg: Weight for regularity loss.
    reduced_xDim, reduced_yDim, reduced_zDim: Dimensions of reduced space.
    xDim, yDim, zDim: Dimensions of the input image.
    dev: Device to run on.
    """
    running_loss = 0
    total = 0

    # Get an initialization of the atlas
    for ave_scan in trainloader:
        atlas, temp = ave_scan
    atlas.requires_grad=True
    opt = optim.Adam([atlas], lr=para.solver.atlas_lr) 

    for epoch in range(para.solver.epochs):
        net.train()
        print('epoch:', epoch)
        for j, tar_bch in enumerate(trainloader):
            b, c, w, h, l = tar_bch[0].shape
            optimizer.zero_grad()
            phiinv_bch = torch.zeros(b, w, h, l, 3).to(dev)
            reg_save = torch.zeros(b, w, h, l, 3).to(dev)
            
            # Pretrain the atlas building network
            if epoch <= para.model.pretrain_epoch:
                perm_indices = torch.randperm(b)
                atlas_bch = tar_bch[0][perm_indices]
            else:
                atlas_bch = torch.cat(b*[atlas]).reshape(b, c, w, h, l)

            atlas_bch = atlas_bch.to(dev).float() 
            tar_bch_img = tar_bch[0].to(dev).float() 
            momentum, latent_feat = net(atlas_bch, tar_bch_img, registration=True) 
            momentum = momentum.permute(0, 4, 3, 2, 1)
            identity = get_grid2(xDim, dev).permute([0, 4, 3, 2, 1])  
            epd = Epdiff(dev, (reduced_xDim, reduced_yDim, reduced_zDim), (xDim, yDim, zDim), para.solver.Alpha, para.solver.Gamma, para.solver.Lpow)

            for b_id in range(b):
                v_fourier = epd.spatial2fourier(momentum[b_id,...].reshape(w, h , l, 3))
                velocity = epd.fourier2spatial(epd.Kcoeff * v_fourier).reshape(w, h , l, 3)  
                reg_temp = epd.fourier2spatial(epd.Lcoeff * v_fourier * v_fourier)
                num_steps = para.solver.Euler_steps
                v_seq, displacement = epd.forward_shooting_v_and_phiinv(velocity, num_steps)  
                phiinv = displacement.unsqueeze(0) + identity
                phiinv_bch[b_id,...] = phiinv 
                reg_save[b_id,...] = reg_temp

            dfm = Torchinterp(atlas_bch,phiinv_bch) 
            Dist = criterion(dfm, tar_bch_img)
            Reg_loss =  reg_save.sum()
            loss_total =  Dist + weight_reg * Reg_loss
            loss_total.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss_total.item()
            total += running_loss
            running_loss = 0.0

        if epoch > para.model.pretrain_epoch:
            opt.step()
            opt.zero_grad()

        print('Total training loss:', total)

def main():

    dev = get_device()
    para = read_yaml('./parameters.yml')
    data_dir = '.'
    json_file = 'train_json'
    keyword = 'train'
    xDim, yDim, zDim= load_and_preprocess_data(data_dir, json_file, keyword)
    print (xDim, yDim, zDim)
    dataset = SData('./train_json.json', "train")
    ave_data = SData('./train_json.json', 'train')
    trainloader = DataLoader(dataset, batch_size= para.solver.batch_size, shuffle=True)
    aveloader = DataLoader(ave_data, batch_size= 1 , shuffle = False)
    combined_loader = zip(trainloader, aveloader )
    net, criterion, optimizer = initialize_network_optimizer(xDim, yDim, zDim, para, dev)
    print (xDim, yDim, zDim)
    train_network(trainloader, aveloader, net, para, criterion, optimizer, NCC, 'l2', 10, 0.001, 16,16,16, xDim, yDim, zDim, dev)

if __name__ == "__main__":
    main()








       
    
 
        


