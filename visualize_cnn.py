"""
EECS 445 - Introduction to Machine Learning
Winter 2019 - Project 2
Visualize CNN
    This will produce visualizations of the activations of the
    first convolutional layer and save them to file.

    Usage: python visualize_cnn.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import utils
from dataset import get_train_val_test_loaders
from model.cnn import CNN
from train_common import *
from utils import config
from scipy.misc import imread
import utils
import copy
import math
import pandas as pd

from matplotlib import pyplot as plt

def visualize_layer1_activations(i):
    xi, yi = tr_loader.dataset[i]
    xi = xi.view((1,3,config('image_dim'),config('image_dim')))
    zi = F.relu(model.conv1(xi))
    zi = zi.detach().numpy()[0]
    sort_mask = np.argsort(model.conv1.weight.detach().numpy().mean(axis=(1,2,3)))
    zi = zi[sort_mask]
    fig, axes = plt.subplots(4, 4, figsize=(10,10))
    for i, ax in enumerate(axes.ravel()):
        ax.axis('off')
        im = ax.imshow(zi[i], cmap='gray')
    fig.suptitle('Layer 1 activations, y={}'.format(yi))
    fig.savefig('CNN_viz1_{}.png'.format(yi), dpi=200, bbox_inches='tight')

if __name__ == '__main__':
    # Attempts to restore from checkpoint
    print('Loading cnn...')
    model = CNN()
    model, start_epoch, _ = restore_checkpoint(model, config('cnn.checkpoint'),
        force=True)

    tr_loader, _, _, _ = get_train_val_test_loaders(
        num_classes=config('cnn.num_classes'))

    # Saving input images in original resolution
    metadata = pd.read_csv(config('csv_file'))
    for idx in [20, 13, 14]:
        filename = os.path.join(
            config('image_path'), metadata.loc[idx, 'filename'])
        plt.imshow(imread(filename))
        plt.axis('off')
        plt.savefig('CNN_viz0_{}.png'.format(int(
            metadata.loc[idx, 'numeric_label'])),
            dpi=200, bbox_inches='tight')

    # Saving layer activations
    for i in [8, 9, 10]:
        visualize_layer1_activations(i)

