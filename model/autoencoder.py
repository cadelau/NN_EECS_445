"""
EECS 445 - Introduction to Machine Learning
Winter 2019 - Project 2
Autoencoder
    Constructs a pytorch model for a neural autoencoder
    Autoencoder usage: from model.autoencoder import Autoencoder
    Autoencoder classifier usage:
        from model.autoencoder import AutoencoderClassifier
    Naive method usage: from model.autoencoder import NaiveRecon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Autoencoder(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        self.repr_dim = repr_dim

        ## Solution: define each layer
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 20736)
        ##

        self.deconv = nn.ConvTranspose2d(repr_dim, 3, 5, stride=2, padding=2)
        self.init_weights()

    def init_weights(self):
        # TODO: initialize the parameters for
        #       [self.fc1, self.fc2, self.fc3, self.deconv]
        for fc in [self.fc1, self.fc2, self.fc3]:
            C_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 0.1 / sqrt(C_in))
            nn.init.constant_(fc.bias, 0.01)
        #
        for dc in [self.deconv]:
            nn.init.normal_(dc.weight, 0.0, 0.01)
            nn.init.constant_(dc.bias, 0.0)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encoder(self, x):
        # TODO: encoder
        N, C, H, W = x.shape
        encoded = self.pool(x)
        encoded = encoded.view(-1, self.num_flat_features(encoded))
        encoded = F.elu(self.fc1(encoded))
        encoded = F.elu(self.fc2(encoded))
        #

        return encoded
    
    def decoder(self, encoded):
        # TODO: decoder
        z = F.elu(self.fc3(encoded))
        #

        decoded = self._grow_and_crop(z)
        decoded = _normalize(decoded)
        return decoded

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def _grow_and_crop(self, x, input_width=18, crop_size=32, scale=2):
        decoded = x.view(-1, self.repr_dim, input_width, input_width)
        decoded = self.deconv(decoded)
        
        magnified_length = input_width * scale
        crop_offset = (magnified_length - crop_size) // 2
        L, R = crop_offset, (magnified_length-crop_offset)
        decoded = decoded[:, :, L:R, L:R]
        return decoded


class AutoencoderClassifier(nn.Module):
    # skip connections
    def __init__(self, repr_dim, d_out, n_neurons=32):
        super().__init__()
        self.repr_dim = repr_dim

        # TODO: define each layer
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 64)
        #

        self.fc_1 = nn.Linear(repr_dim, n_neurons)
        self.fc_2 = nn.Linear(n_neurons, n_neurons)
        self.fc_3 = nn.Linear(n_neurons, n_neurons)
        self.fc_last = nn.Linear(n_neurons, d_out)
    
    def forward(self, x):
        encoded = self.encoder(x)

        z1 = F.elu(self.fc_1(encoded))
        z2 = F.elu(self.fc_2(z1))
        z3 = F.elu(self.fc_3(z2))
        z = F.elu(self.fc_last(z1 + z3))
        return z

    def encoder(self, x):
        # TODO: encoder
        N, C, H, W = x.shape
        encoded = self.pool(x)
        encoded = encoded.view(-1, self.num_flat_features(encoded))
        encoded = F.elu(self.fc1(encoded))
        encoded = F.elu(self.fc2(encoded))
        #
        
        return encoded

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class NaiveRecon(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        compressed = F.avg_pool2d(x, self.scale, stride=self.scale)
        grow = F.interpolate(compressed, size=(32, 32),
            mode='bilinear', align_corners=False)
        reconstructed = _normalize(grow)
        return compressed, reconstructed

def _normalize(x):
    """
    Per-image channelwise normalization
    """
    mean = x.mean(2, True).mean(3, True).mean(0, True)
    std = torch.sqrt((x - mean).pow(2).mean(2, True).mean(3, True).mean(0, True))
    z = (x - mean) / std
    return z
