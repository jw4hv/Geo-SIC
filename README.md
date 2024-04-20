
# Geo-SIC: Learning Deformable Geometric Shapes in Deep Image Classifiers

Geo-SIC is an image classifier enhanced by joint geometric shape learning in atlas building. Read our paper on [arXiv](https://arxiv.org/abs/2210.13704).

## Upcoming Additions

We've updated the most recent version of joint learning of Geo-SIC using PyTorch. Additionally, an integration between C++ and Python will be available soon. We're also expanding our testing procedure to include more diverse classifiers.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Parameters](#parameters)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)

## Introduction

Geo-SIC operates as follows:

1. **Atlas Building**: Input images into an atlas building network to extract geometric features in the latent space of transformation fields. An atlas is learned during this process.
2. **Image Classifier**: Input images to a backbone classifier. Fuse both backbone features and the features from the atlas to predict a class label.

## Installation

Ensure that your Python version is greater than 3.12. If you wish to use the spatial version of LDDMM (Lagomorph) to produce geometric features, make sure inlude the source code we provided and also install the `lagomorph` package by:

```bash
pip install lagomorph
```

PyTorch version >= 0.4.0 is also required.

## Parameters

### Entire Model

- **optimizer**: Specifies the optimizer used for training the model (e.g., 'Adam').
- **scheduler**: Specifies the learning rate scheduler (e.g., 'CosAn' for Cosine Annealing).
- **loss**: Specifies the loss function used for training (e.g., 'L2' for Mean Squared Error, NCC for normalized cross-correlation).
- **augmentation**: Boolean indicating whether data augmentation is enabled.
- **reduced_dim**: Specifies the reduced dimensions (e.g., 16, 16, 16).
- **pretrain_epoch**: Specifies the number of epochs for pretraining (e.g., 300).
- **lr**: Learning rate for both models (e.g., 1e-4).
- **epochs**: Total number of epochs for training (e.g., 1000).
- **batch_size**: Batch size for training (e.g., 1, 4, 8).
- **weight_decay**: Weight decay parameter (e.g., 1e-5).

### Geometric Shape Learning

- **Euler_steps**: Number of steps for Euler integration (e.g., 5, 10).
- **Alpha**: Alpha in shooting (e.g., 1.0, 2.0).
- **Gamma**: Gamma in shooting (e.g., 1.0).
- **Lpow**: The power of the Laplacian operator in shooting (e.g., 4.0, 6.0).
- **Sigma**: The noise variance on the image matching term in LDDMM (e.g., 0.02, 0.03).

### Image Classifier

- `in_channels`: Number of input channels (should be set to 1 for grayscale images).
- `conv_channels`: List containing the number of channels for each convolutional layer (e.g., [8, 16, 16, 32, 32]).
- `conv_kernel_sizes`: List containing the kernel sizes for each convolutional layer (e.g., [3, 3, 3, 3, 3]).
- `activation`: Activation function used in the convolutional layers (e.g., 'ReLU', 'PReLU').
- `num_classes`: Number of output classes for the classification task (e.g., 2).

## Usage

To build an atlas using our deep learning framework, run `Run_Atlas_trainer.py`. To run the entire training for classifiers, simply execute the script `Run_trainer.py`.

## Documentation

We've included different shooting models:

- [Stationary Velocity Field in Voxelmorph](https://arxiv.org/abs/1809.05231) (Code: [GitHub](https://github.com/voxelmorph/voxelmorph))
- [Vector-Momenta Shooting Method in Lagomorph](https://openreview.net/pdf?id=Hkg0j9sA1V) (Code: [GitHub](https://github.com/jacobhinkle/lagomorph))
- [LDDMM-based on Fourier representations](https://people.csail.mit.edu/polina/papers/Zhang-IPMI-2017.pdf) (Code: [Bitbucket](https://bitbucket.org/FlashC/flashc/src)).

## Contributing

If you have any questions, feel free to reach out to us by opening an issue.

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.


