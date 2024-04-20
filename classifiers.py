import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Flexi3DCNN(nn.Module):
    def __init__(self, in_channels, conv_channels, conv_kernel_sizes, num_classes, activation):
        super(Flexi3DCNN, self).__init__()
        self.num_conv_layers = len(conv_channels)

        # Define convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(self.num_conv_layers):
            if i == 0:
                conv_layer = nn.Conv3d(in_channels, conv_channels[i], kernel_size=conv_kernel_sizes[i], stride=1, padding=1)
            else:
                conv_layer = nn.Conv3d(conv_channels[i-1], conv_channels[i], kernel_size=conv_kernel_sizes[i], stride=1, padding=1)
            self.conv_layers.append(conv_layer)

        # Pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_channels[-1] * 4 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Activation function
        if activation == 'ReLU':
            self.act = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.act = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation function.")

    def forward(self, x, latent_f, weight_f):
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = self.act(self.pool(conv_layer(x)))
        x = weight_f*latent_f + x 
        x = torch.flatten(x, 1)
        # Fully connected layers
        x = self.act(self.fc1(x))
        x = self.fc2(x)

        return x

