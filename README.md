*************************************** Geo-SIC ****************************************
This repository contains source code and data of a Deep Image Classifiers with Learning Deformable Geometric Shape representations.


*************************************** Disclaimer ****************************************
This software is published for academic and non-commercial use only.

The implementation includes network training, testing for 2D and 3D image classification & atlas building. We request you to cite our research paper if you use it:

Geo-SIC: Learning Deformable Geometric Shapes in Deep Image Classifiers
Jian Wang, Miaomiao Zhang. Conference on Neural Information Processing Systems (NeurIPS), 2022.

*************************************** Setup ****************************************
* [PyTorch 3.7](http://pytorch.org/)
* [FLASH] (https://bitbucket.org/FlashC/flashc/src/master/) 
* [CUDA 10.0](https://developer.nvidia.com/cuda-downloads)
* [Anaconda 4.3.1](https://anaconda.org)

*************************************** Usage ****************************************
Below is a *QuickStart* guide on how to use Geo-SIC for network training and testing.

    +++========================= Data organization ========================+++
    All data are mangaed by ".json " script. There are two keys in "json" as 'train' and 'validation' or 'testing'. Please organized the data under those two keys by any script and put the .json file with the same directory with the data.
    
    +++========================= Parameter setting ========================+++
    We use .yaml file to manage the paramters used in this model, an example of parameters is given as follows, 

                    study:
                            name: 'prototype' (namming the model)

                    data:
                        predict: '2D' (choose 2D or 3D)
                        x: 128 (input data size here)
                        y: 128 (input data size here)
                        z: 128 (if it is 2D input 1)

                    model:
                        network_type: 'CNN' (select network type here)
                        pooling: 'AverageROI' (select pooling type here)
                        num_outputs: 3 (input the number of classes)
                        num_blocks: 2 (input the number of CNN blocks if choosing CNN)
                        optimizer: 'Adam' (select optimizer here)
                        scheduler: 'CosAn' (select learning rate schduler here)
                        loss: 'L2'    ##'L2' or 'L1' (select loss type here)
                        augmentation: False (select if augumentation is needed, now only False provided)
                        image_grad: False (select if image grad is needed, now only False provided)
                        pure_atlas_bulding: True (select this if you only need to build atlas).

                    solver:
                        lr: 0.001 (specify model learning rate)
                        epochs: 1000 (specify model learning epochs)
                        batch_size: 4 (specify model batch size)
                        weight_decay: 0.0001 (specify weight decay parameter)
                        pre_train: 20 (specify pretrain epochs)
    +++========================= Training & Testing ========================+++
                    Steps: 
                    cd Geo-SIC/
                    python Geo-SIC.py
                        
