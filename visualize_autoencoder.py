"""
EECS 445 - Introduction to Machine Learning
Winter 2019 - Project 2
Visualize Autoencoder
    For each class, shows side-by-side random validation images, naive
    reconstructions obtained by downsampling-then-upsampling, and the
    autoencoder reconstructions. Exit by causing a KeyboardInterrupt
    (press CTRL-c).
    Usage: python visualize_autoencoder.py
"""
import os
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
from dataset import get_train_val_test_loaders
from model.autoencoder import Autoencoder, NaiveRecon
from train_common import *
from utils import config

random.seed(0)
torch.manual_seed(0)

def get_data_by_label(dataset):
    data = []
    for i, (X, y) in enumerate(dataset):
        for c in range(config('autoencoder.num_classes')):
            batch = X[(y == c).nonzero().squeeze(1)]
            if len(data) <= c:
                data.append(batch)
            else:
                data[c] = torch.cat((data[c], batch))
    return data

def visualize_autoencoder(dataset, get_semantic_label, ae, naive):
    num_classes = config('autoencoder.num_classes')
    cols = ['Orig', 'Naive recon', 'Autoencoder recon']
    rows = []
    fig, axes = plt.subplots(nrows=num_classes, ncols=3, figsize=(10,16))

    for c in range(num_classes):
        X = dataset[c]
        y = get_semantic_label(c)
        rows.append(y)

        image = utils.denormalize_image(
            X[0].data.numpy().transpose((1,2,0)))
        naive_img = utils.denormalize_image(
            naive(X)[1][0].data.numpy().transpose((1,2,0)))
        ae_img = utils.denormalize_image(
            ae(X)[1][0].data.numpy().transpose((1,2,0)))

        axes[c,0].imshow(image,     interpolation='bicubic')
        axes[c,1].imshow(naive_img, interpolation='bicubic')
        axes[c,2].imshow(ae_img,    interpolation='bicubic')

    # Add column and row headers
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(110, 0),
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='large', ha='right', va='center')

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax in axes.ravel():
        ax.axis('off')

    plt.savefig('ae_recon_comparison.png', dpi=200, bbox_inches='tight')
#     plt.show()

def main():
    # data loaders
    _, va_loader, _, get_semantic_label = get_train_val_test_loaders(
        num_classes=config('autoencoder.num_classes'))
    dataset = get_data_by_label(va_loader)

    ae = Autoencoder(config('autoencoder.ae_repr_dim'))
    naive = NaiveRecon(config('autoencoder.naive_scale'))

    # Restore the latest checkpoint of autoencoder
    print('Loading autoencoder...')
    ae, _, _ = restore_checkpoint(ae, config('autoencoder.checkpoint'),
        force=True)

    # Visualize
    visualize_autoencoder(dataset, get_semantic_label, ae, naive)

if __name__ == '__main__':
    main()
