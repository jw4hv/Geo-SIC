from os import PathLike
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import os
import glob
import json
import subprocess
import sys
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
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
from classifiers import *
import lagomorph as lm 


def get_device():
    """Returns the device available (cuda or cpu)"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def read_yaml(path):
    """Reads a YAML file and returns its contents as a dictionary."""
    try:
        with open(path, 'r') as f:
            file = edict(yaml.load(f, Loader=yaml.FullLoader))
        return file
    except:
        print('NO FILE READ!')
        return None


def load_and_preprocess_data(data_dir, json_file, keyword):
    """
    Loads and preprocesses data from a specified directory and JSON file.
    Returns the dimensions of the loaded data.
    """
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
    """
    Initializes the atlas building neural network, classifier, loss functions, optimizer, and scheduler.
    Returns the initialized objects.
    """
    # Initialize the atlas building network (UnetDense)
    net = UnetDense(inshape=(xDim, yDim, zDim),
                      nb_unet_features=[[16, 32,32], [ 32, 32, 32, 16, 16]], #[16, 32,32], [ 32, 32, 32, 16, 16]
                      nb_unet_conv_per_level=1,
                      int_steps=7,
                      int_downsize=2,
                      src_feats=1,
                      trg_feats=1,
                      unet_half_res=True)
    net = net.to(dev)

    # Initialize the image classifier (Flexi3DCNN)
    in_channels = 1
    conv_channels = [8, 16, 16, 32, 32]  # Number of channels for each convolutional layer
    conv_kernel_sizes = [3, 3, 3,3, 3]  # Kernel sizes for each convolutional layer
    activation = 'ReLU'  # Activation function
    num_classes = 2 # Number of classes
    clfer = Flexi3DCNN(in_channels, conv_channels, conv_kernel_sizes, num_classes, activation)
    clfer = clfer.to(dev)

    # Combine parameters for optimization
    params = list(net.parameters()) + list(clfer.parameters())

    # Initialize loss functions
    criterion_clf = nn.CrossEntropyLoss()
    if para.model.loss == 'L2':
        criterion = nn.MSELoss()
    elif para.model.loss == 'L1':
        criterion = nn.L1Loss()

    # Initialize optimizer
    if para.model.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=para.solver.lr)
    elif para.model.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=para.solver.lr, momentum=0.9)

    # Initialize scheduler (CosineAnnealingLR)
    scheduler = CosineAnnealingLR(optimizer, T_max=para.solver.epochs)

    return net, clfer, criterion, criterion_clf, num_classes, optimizer, scheduler


def train_network(trainloader, aveloader, net, clfer, para, criterion, criterion_clf, num_classes, optimizer, scheduler, DistType, RegularityType, weight_dist, weight_reg, weight_latent, reduced_xDim, reduced_yDim, reduced_zDim, xDim, yDim, zDim, dev, flag):
    """
    Trains the atlas building neural network and classifier.
    """
    running_loss = 0
    total = 0
    ''' Define fluid paramerts if using vector-momenta to shoot forward'''
    fluid_params = [1.0, 0.1, 0.05]
    lddmm_metirc = lm.FluidMetric(fluid_params)
    # Get an initialization of the atlas
    for ave_scan in trainloader:
        atlas, temp = ave_scan
    atlas.requires_grad=True
    opt = optim.Adam([atlas], lr=para.solver.atlas_lr) 

    for epoch in range(para.solver.epochs):
        net.train()
        clfer.train()
        print('epoch:', epoch)
        for j, tar_bch in enumerate(trainloader):
            b, c, w, h, l = tar_bch[0].shape
            optimizer.zero_grad()
            phiinv_bch = torch.zeros(b, w, h, l, 3).to(dev)
            reg_save = torch.zeros(b, w, h, l, 3).to(dev)
            
            # Shuffle the pairs then pretrain the atlas building network
            if epoch <= para.model.pretrain_epoch:
                perm_indices = torch.randperm(b)
                atlas_bch = tar_bch[0][perm_indices]
            else:
                atlas_bch = torch.cat(b*[atlas]).reshape(b, c, w, h, l)

            atlas_bch = atlas_bch.to(dev).float() 
            tar_bch_img = tar_bch[0].to(dev).float() 
            
            # Train atlas building with extracted latent features
            pred = net(atlas_bch, tar_bch_img, registration=True, shooting = flag) 

            # Train image classifier with feature fusion strategy using a specified weighting parameter, this network will not be updated unless the atlas building is pretrained
            cl_pred = clfer (tar_bch_img ,pred[2], weight_latent)

            # Create a tensor from the ground truth label, one-hot for multi-classes
            tar_bch_lbl = F.one_hot(torch.tensor(int(tar_bch[1][0])), num_classes).to(dev).float()
            clf_loss = criterion_clf(cl_pred[0], tar_bch_lbl)
            
            # Characterize the geometric shape information using different methods after obtaining the momentum from the atlas building network
            if (flag == "FLDDMM"): # LDDMM to perform geodesic shooting 
                momentum = pred[0].permute(0, 4, 3, 2, 1)
                identity = get_grid2(xDim, dev).permute([0, 4, 3, 2, 1])  
                epd = Epdiff(dev, (reduced_xDim, reduced_yDim, reduced_zDim), (xDim, yDim, zDim), para.solver.Alpha, para.solver.Gamma, para.solver.Lpow)

                for b_id in range(b):
                    v_fourier = epd.spatial2fourier(momentum[b_id,...].reshape(w, h , l, 3))
                    velocity = epd.fourier2spatial(epd.Kcoeff * v_fourier).reshape(w, h , l, 3)  
                    # sitk.WriteImage(sitk.GetImageFromArray(velocity.detach().cpu().numpy()), "./Velocity0.nii.gz")
                    reg_temp = epd.fourier2spatial(epd.Lcoeff * v_fourier * v_fourier)
                    num_steps = para.solver.Euler_steps
                    v_seq, displacement = epd.forward_shooting_v_and_phiinv(velocity, num_steps)  
                    phiinv = displacement.unsqueeze(0) + identity
                    phiinv_bch[b_id,...] = phiinv 
                    reg_save[b_id,...] = reg_temp

                dfm = Torchinterp(atlas_bch,phiinv_bch) 
                Dist = criterion(dfm, tar_bch_img)
                Reg_loss =  reg_save.sum()
                if epoch <= para.model.pretrain_epoch:
                    loss_total =  Dist + weight_reg * Reg_loss
                else:
                    loss_total =  Dist + weight_reg * Reg_loss + clf_loss

            elif (flag == "SVF"): # Stationary velocity fields to shoot forward 
                print (pred[1].shape)
                Dist = NCC().loss(pred[0], tar_bch_img)   
                Reg = Grad( penalty= RegularityType)
                Reg_loss  = Reg.loss(pred[1])
                if epoch <= para.model.pretrain_epoch:
                    loss_total = weight_dist * Dist + weight_reg * Reg_loss 
                else:
                    loss_total = weight_dist * Dist + weight_reg * Reg_loss + clf_loss

            elif (flag == "VecMome"): # A spatial version of LDDMM on CUDA to perform geodesic shooting 
                h = lm.expmap(lddmm_metirc, pred[1], num_steps= para.solver.Euler_steps)
                Idef = lm.interp(atlas_bch, h)
                v = lddmm_metirc.sharp(pred[1])
                reg_term = (v*pred[1]).mean()
                
                if epoch <= para.model.pretrain_epoch:
                    loss_total= (1/(para.solver.Sigma*para.solver.Sigma))*NCC().loss(Idef, tar_bch_img) + reg_term
                else:
                    loss_total= (1/(para.solver.Sigma*para.solver.Sigma))*NCC().loss(Idef, tar_bch_img) + reg_term + clf_loss

            loss_total.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss_total.item()
            total += running_loss
            running_loss = 0.0

        scheduler.step()  # Update learning rate

        '''Using Adam to update the atlas'''
        if epoch > para.model.pretrain_epoch:
            opt.step()
            opt.zero_grad()

        print('Total training loss:', total)


def main():
    """
    Main function to run the training process.
    """
    dev = get_device()
    para = read_yaml('./parameters.yml')
    data_dir = '.'
    json_file = 'train_json'
    keyword = 'train'
    xDim, yDim, zDim = load_and_preprocess_data(data_dir, json_file, keyword)
    dataset = SData('./train_json.json', "train")
    ave_data = SData('./train_json.json', 'train')
    trainloader = DataLoader(dataset, batch_size=para.solver.batch_size, shuffle=True)
    aveloader = DataLoader(ave_data, batch_size=1, shuffle=False)
    combined_loader = zip(trainloader, aveloader)
    net, clfer, criterion, criterion_clf, num_classes, optimizer, scheduler = initialize_network_optimizer(xDim, yDim, zDim, para, dev)

    train_network(trainloader, aveloader, net, clfer, para, criterion, criterion_clf, num_classes, optimizer, scheduler, NCC, 'l2', 0.5, 0.5, 0.2, 16, 16, 16, xDim, yDim, zDim, dev, "VecMome")


if __name__ == "__main__":
    main()
