from __future__ import division, print_function
from typing import Dict, SupportsRound, Tuple, Any
from os import PathLike
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch,gc
from torch.autograd import grad
from torch.autograd import Variable
import torch.fft ############### Pytorch >= 1.8.0
import torch.nn.functional as F
import SimpleITK as sitk
import os, glob
import json
import subprocess
import sys
from PIL import Image
import torch.nn.functional as nnf
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from easydict import EasyDict as edict
from sklearn.metrics import accuracy_score
import random
import yaml
from DataSet import DataSet_GS
from Tools import *
from uEpdiff import *
     


################ Device Seting #######################
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

################ Parameter Loading #######################
def read_yaml(path):
    try:
        with open(path, 'r') as f:
            file = edict(yaml.load(f, Loader=yaml.FullLoader))
        return file
    except:
        print('NO FILE READ!')
        return None
para = read_yaml('./parameters.yml')
xDim = para.data.x 
yDim = para.data.y
zDim = para.data.z

##################EPDiff setting up####################
imagesize = para.data.x
truncate = 16
gradv_batch = torch.cuda.FloatTensor(para.solver.batch_size, 3, para.data.x, para.data.y,para.data.z).fill_(0).contiguous()
defIm_batch = torch.cuda.FloatTensor(para.solver.batch_size, 1, para.data.x, para.data.y,para.data.z).fill_(0).contiguous()
temp = torch.cuda.FloatTensor(para.solver.batch_size, 3, para.data.x, para.data.y,para.data.z).fill_(0).contiguous()
transformations = torch.cuda.FloatTensor(para.solver.batch_size, 3,  para.data.x, para.data.y,para.data.z).fill_(0).contiguous() 
def loss_Reg(y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0
        return grad

##################Data Loading##########################
readfilename = './3DBrain/data' + '.json'
datapath = './3DBrain/'
data = json.load(open(readfilename, 'r'))

################## Training Data Loading##########################
outputs = []
keyword = 'train'
ave_scan = np.zeros((1,xDim,yDim,zDim))
source_scan = np.zeros((1,xDim,yDim,zDim))
for i in range (0,len(data[keyword])):
    filename_src = datapath + data[keyword][i]['image']
    itkimage_src = sitk.ReadImage(filename_src)
    source_scan = sitk.GetArrayFromImage(itkimage_src)
    ave_scan = ave_scan + source_scan
    outputs.append(source_scan)
ave_scan = torch.FloatTensor((ave_scan/len(data[keyword])).reshape(1,1,xDim,yDim,zDim))
rand_scan= torch.FloatTensor(source_scan.reshape(1,1,xDim,yDim,zDim))
train = torch.FloatTensor(outputs)
train = train.reshape(len(data[keyword]),1,xDim,yDim,zDim)
print (train.shape)


class_label = []
for i in range (0,len(data[keyword])):
    templabel = int(data[keyword][i]['label'])
    class_label.append(templabel)

train_label = torch.tensor(class_label, dtype=torch.long)
print (train_label.shape)
training = DataSet_GS (input_image = train, groundtruth = train_label )

################## Validation Data Loading##########################
'''Creat a json file with the same format with training data'''
outputs = []
keyword = 'val'
# outputs = np.array(outputs)

for i in range (0,len(data[keyword])):
    filename_src = datapath + data[keyword][i]['image']
    itkimage_src = sitk.ReadImage(filename_src)
    source_scan = sitk.GetArrayFromImage(itkimage_src)
    ave_scan = ave_scan + source_scan
    outputs.append(source_scan)
ave_scan = torch.FloatTensor((ave_scan/len(data[keyword])).reshape(1,1,xDim,yDim,zDim))
rand_scan= torch.FloatTensor(source_scan.reshape(1,1,xDim,yDim,zDim))
val = torch.FloatTensor(outputs)
val= val.reshape (len(data[keyword]),1,xDim,yDim,zDim)
print (val.shape)

class_label = []
for i in range (0,len(data[keyword])):
    templabel = int(data[keyword][i]['label'])
    class_label.append(templabel)

val_label = torch.tensor(class_label, dtype=torch.long)
print (val_label.shape)
validation = DataSet_GS (input_image = val, groundtruth = val_label )

################# Network Setting########################
'''Loading a simple Unet and a CNN'''
from classifiers import simpleeCNN
from networks import VxmDense  
from losses import MSE, Grad
net = []
for i in range(3):
    temp = VxmDense(inshape = (xDim,yDim,zDim),
				 nb_unet_features= [[16, 32],[ 32, 32, 16, 16]],
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res= False)
    net.append(temp)
net = net[0].to(dev)
'''Checking network'''
# print (net)

net_SIC = simpleeCNN()
net_SIC = net_SIC.to(dev)
trainloader = torch.utils.data.DataLoader(training, batch_size = para.solver.batch_size, shuffle=True, num_workers=1)
valloader = torch.utils.data.DataLoader(validation, batch_size = para.solver.batch_size, shuffle=True, num_workers=1)

running_loss = 0 
running_loss_val = 0
template_loss = 0
printfreq = 1
sigma = 0.02
repara_trick = 0.0
loss_array = torch.FloatTensor(para.solver.epochs,1).fill_(0)
loss_array_val = torch.FloatTensor(para.solver.epochs,1).fill_(0)
# # loss_array = loss_array.to(device)
atlas = torch.cuda.FloatTensor(1, 1, xDim, yDim).fill_(0).contiguous()
atlas.requires_grad=True
ave_scan = ave_scan.to(dev)
rand_scan = rand_scan.to(dev)
ave_scan.requires_grad=True


deform_size = [1, xDim, yDim]
params = list(net.parameters()) + list(net_SIC.parameters())
if(para.model.loss == 'L2'):
    criterion = nn.MSELoss()
elif (para.model.loss == 'L1'):
    criterion = nn.L1Loss()
if(para.model.optimizer == 'Adam'):
    optimizer = optim.Adam(net.parameters(), lr= para.solver.lr)
elif (para.model.optimizer == 'SGD'):
    optimizer = optim.SGD(net.parameters(), lr= para.solver.lr, momentum=0.9)
if (para.model.scheduler == 'CosAn'):
    scheduler = CosineAnnealingLR(optimizer, T_max=len(valloader), eta_min=0)
criterion_SIC = nn.CrossEntropyLoss() 
optimizer_SIC = optim.Adam(net_SIC.parameters(), lr= para.solver.lr)
opt = optim.Adam(params, lr= para.solver.lr)
optimizer_template = optim.Adam(net.parameters(), lr= para.solver.lr)
scheduler_template = CosineAnnealingLR(optimizer_template, T_max=len(valloader), eta_min=0)

# ##################Training###################################        
for epoch in range(para.solver.epochs):
    total= 0; 
    total_val = 0; 
    total_template = 0; 
    net.train()
    net_SIC.train()
    print('epoch:', epoch)
    for j, atlas_data in enumerate(trainloader):
        inputs = atlas_data['input_image'].to(dev)
        batch_labels = atlas_data['gt'].to(dev)
        b, c, w, h, len = inputs.shape
        opt.zero_grad()
        atlas = ave_scan
        source_b = torch.cat(b*[atlas]).reshape(b , c, w , h, len)
        pred = net(source_b, inputs, registration = True)   #prediction size (b, 3, 128,128,128)
        m0 = pred[0]
        latent_feature = pred[1]
        ############################ Using Fourier EPDiff in C++ ###########################

        for batch_id in range (0, b):
            src = atlas[batch_id,0,:,:,:].reshape(w,h,len).detach().cpu().numpy()
            '''Check sources'''
            # if ((epoch%printfreq==0) and (batch_id == 0)):
            #     im= sitk.GetImageFromArray(src, isVector=False)
            #     save_path = './saved_files_train/source' + str(epoch) +'_'+str(j) + '_'+ str(batch_id)  + '.mhd'
            #     sitk.WriteImage(sitk.GetImageFromArray(src, isVector=False), save_path,False)
            '''Saving sources for FLASH'''
            sitk.WriteImage(sitk.GetImageFromArray(src, isVector=False), './source.mhd',False)
            tar = inputs[batch_id,0,:,:,:].reshape(w,h,len).detach().cpu().numpy()
            '''Check targets'''
            # if ((epoch%printfreq==0)and (batch_id == 0)):
            #     im = sitk.GetImageFromArray(tar, isVector=False)
            #     save_path = './saved_files_train/target' + str(epoch) +'_'+ str(j) + '_'+ str(batch_id)  + '.mhd'
            #     sitk.WriteImage(sitk.GetImageFromArray(tar, isVector=False), save_path,False)
            '''Saving targets for FLASH'''
            sitk.WriteImage(sitk.GetImageFromArray(tar, isVector=False), './target.mhd',False)

            '''Saving momentum fields for FLASH'''
            velo = m0[batch_id,:,:,:,:].reshape(c*3,w,h,len)
            velo = velo.permute(1,2,3,0).detach().cpu().numpy()
            sitk.WriteImage(sitk.GetImageFromArray(velo,isVector=True), './v0Spatial.mhd',False)

            ''' Run a bash file from FLASH to generate gradient and deformed images'''
            command = subprocess.run(["sh runImageMatching.sh"],  shell=True)
            gradv = sitk.GetArrayFromImage(sitk.ReadImage('./gradSpatial.mhd'))
            gradv = torch.tensor(gradv, dtype=torch.float)
            gradv = gradv.permute(0, 3, 1, 2)
            # print (gradv.shape)
            gradv_batch[batch_id,:,:,:,:] = gradv
            gradv_save = gradv.reshape(c,w,h,len)
            gradv_save = gradv_save.permute(1,2,3,0).detach().cpu().numpy()

            ''' Checking gradients'''
            if ((epoch%printfreq==0) and (batch_id == 0) ):
                save_path = './saved_files_train/gradv' + str(epoch) +'_'+ str(j) + '_' + str(batch_id) + '.mhd'
                im= sitk.GetImageFromArray(gradv_save, isVector=True)
                sitk.WriteImage(im, save_path,False)

        
            deformIm = sitk.GetArrayFromImage(sitk.ReadImage('./deformIm.mhd'))
            '''Check deformed images'''
            if ((epoch%printfreq==0) and (batch_id == 0)):
                save_path = './saved_files_train/deform' + str(epoch) +'_'+ str(j) + '_' + str(batch_id) + '.mhd'
                im= sitk.GetImageFromArray(deformIm, isVector=False)
                sitk.WriteImage(im, save_path,False)

            deformIm= torch.tensor(deformIm) 
            defIm_batch[batch_id,:,:,:,:] = deformIm
            command = subprocess.run(["sh clean.sh"],  shell=True)
            '''Delete the one-time computed tensors to save some space on CUDA'''
            del src, tar, velo, gradv, deformIm
        '''Compute the atlas building loss'''
        loss_1 = criterion(defIm_batch, inputs) # Dissimilarity term
        loss_2 = criterion(m0, temp) # Regularity term on v_0
        loss = loss_1/(sigma*sigma)  + loss_2

        pred2 = net_SIC(inputs, pred[2]) # Classification loss
        loss_SIC = criterion_SIC(pred, batch_labels) 
        
        #Only train SIC loss when Geo network is pre-trained
        if (epoch > para.solver.pre_train ):
            loss_total = .5*loss_SIC + loss
        else: 
             loss_total = loss
        loss_total.backward(retain_graph=True)
        opt.step()
        running_loss += loss_total.item()
        print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss ))
        total += running_loss
        running_loss = 0.0
        '''Update atlas'''
        if (epoch > para.solver.pre_train):
            with torch.no_grad():
                atlas.data = atlas.data - para.solver.lrofatlas*atlas.grad

    # Validation
    acc_ave = 0
    for t, val_data in enumerate(valloader):
        inputs = val_data['input_image'].to(dev)
        batch_labels = val_data['gt'].to(dev)
        b, c, w, h = inputs.shape
        pred = net(source_b, inputs, registration = True)  
        pred = net_SIC(inputs, pred[2])
        pred_labels = torch.cuda.FloatTensor(b).fill_(0).contiguous()
        loss_SIC = criterion_SIC(pred, batch_labels)
        #Only train SIC loss when Geo network is pre-trained
        loss_total_val = loss_SIC
        running_loss_val += loss_total_val.item()
        total_val += running_loss_val
        running_loss_val = 0.0
        for h in range (0, b):
            if (pred[h,0]>pred[h,1]):
                pred_labels[h] = 0
            else:
                pred_labels[h] = 1
        # print (batch_labels.detach().cpu().numpy(), pred_labels.detach().cpu().numpy())
        acc= accuracy_score(batch_labels.detach().cpu().numpy(), pred_labels.detach().cpu().numpy())
        acc_ave += acc 
    loss_array_val[epoch] = acc_ave/(len(val)/para.solver.batch_size)
    print ('total training loss:', total)
    print ('total validation loss:', total_val)
np.save ('./accuracy_no',loss_array_val.detach().cpu().numpy())


       
    
 
        


