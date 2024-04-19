Geo-SIC Repository Overview

This repository includes the implementation of "Geo-SIC: Learning Deformable Geometric Shapes in Deep Image Classifiers", as described in the paper (https://arxiv.org/abs/2210.13704).

Updates in Progress

We have recently updated the geometric shape learning based on Fourier representations (https://link.springer.com/article/10.1007/s11263-018-1099-x) of transformation fields for atlas building, which is now up to date. You can find the offcial implementation at FlashC (https://bitbucket.org/FlashC/flashc).

Upcoming Additions

Very soon, we plan to provide additional ways to learn geometric shapes using different techniques in classifers as follows, stay tuned.

    Stationary Velocity Field in Voxelmorph (Paper: https://arxiv.org/abs/1809.05231, Code: https://github.com/voxelmorph/voxelmorph).
    Vector-Momenta Shooting Method in Lagomorph (Paper: https://openreview.net/pdf?id=Hkg0j9sA1V, Code: https://github.com/jacobhinkle/lagomorph).

```Usage```

To use the repository, simply execute the script Run_Atlas_Trainer.py.

```Network Parameters```



    optimizer: Specifies the optimizer used for training the model (e.g., 'Adam').
    scheduler: Specifies the learning rate scheduler (e.g., 'CosAn' for Cosine Annealing).
    loss: Specifies the loss function used for training (e.g., 'L2' for Mean Squared Error).
    augmentation: Boolean indicating whether data augmentation is enabled.
    image_grad: Boolean indicating whether to use image gradients.
    kernel_K: Boolean indicating whether to use a kernel K.
    pure_atlas_building: Boolean indicating whether to perform pure atlas building.
    reduced_dim: Specifies the reduced dimensions.
    pretrain_epoch: Specifies the number of epochs for pretraining.

```Solver Specifications```

    lr: Learning rate for the main model.
    atlas_lr: Learning rate for the atlas.
    epochs: Total number of epochs for training.
    batch_size: Batch size for training.
    weight_decay: Weight decay parameter.
    pre_train: Number of epochs for pre-training.
    Euler_steps: Number of steps for Euler integration.
    Alpha: Alpha in shoorting.
    Gamma: Gamma in shoorting.
    Lpow: the power of laplacian operator in shoorting.
