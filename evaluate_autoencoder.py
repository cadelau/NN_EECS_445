"""
EECS 445 - Introduction to Machine Learning
Winter 2019 - Project 2
Evaluate Autoencoder
    Runs inference on an autoencoder to get the per-class performance on the
    validation data.
    Usage: python evaluate_autoencoder.py
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
from dataset import get_train_val_test_loaders
from model.autoencoder import Autoencoder
from train_common import *
from utils import config

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

def evaluate_autoencoder(dataset, get_semantic_label, model, criterion):
    """
    Runs inference on an autoencoder model to evaluate the mse loss on the
    validation sets. Reports per-class performance to terminal.
    """
    num_classes = config('autoencoder.num_classes')
    batch_size = config('autoencoder.batch_size')
    performance = np.zeros(num_classes)
    for c in range(num_classes):
        len_ = 0
        X = dataset[c]
        _, recon = model(X)
        batch_mse = criterion(recon, X).item()
        performance[c] = batch_mse
    for c, p in enumerate(performance):
        print('Class {}: {} mean squared error'
            .format(get_semantic_label(c), p))

def report_validation_performance(dataset, get_semantic_label, model, criterion):
    cols = ['Orig', 'Autoencoder recon', 'Orig', 'Autoencoder recon', 'Orig', 'Autoencoder recon']
    rows = []
    fig, axes = plt.subplots(nrows=config('autoencoder.num_classes'),
        ncols=6, figsize=(16,16))
    for i in range(config('autoencoder.num_classes')):
        X = dataset[i]
        rows.append(get_semantic_label(i))
        _, recon = model(X)
        error = criterion(recon, X).item()
        print('label {}, test error is {}'.format(i, error))
        losses = ((recon - X) ** 2).mean(3).mean(2).mean(1).data.numpy()
        best, worst = np.argmin(losses), np.argmax(losses)
        typical = np.argsort(losses)[len(losses)//2]
        print('  best case:', losses[best])
        print(' worst case:', losses[worst])
        print('    typical:', losses[typical])
        axes[i,0].imshow(utils.denormalize_image(
            np.transpose(X[best].numpy(), (1, 2, 0))),
            plt.get_cmap('gray'), interpolation='bicubic', clim=(-1.0, +1.0))
        axes[i,1].imshow(utils.denormalize_image(
            np.transpose(recon[best].data.numpy(), (1, 2, 0))),
            plt.get_cmap('gray'), interpolation='bicubic', clim=(-1.0, +1.0))
        axes[i,2].imshow(utils.denormalize_image(
            np.transpose(X[worst].numpy(), (1, 2, 0))),
            plt.get_cmap('gray'), interpolation='bicubic', clim=(-1.0, +1.0))
        axes[i,3].imshow(utils.denormalize_image(
            np.transpose(recon[worst].data.numpy(), (1, 2, 0))),
            plt.get_cmap('gray'), interpolation='bicubic', clim=(-1.0, +1.0))
        axes[i,4].imshow(utils.denormalize_image(
            np.transpose(X[typical].numpy(), (1, 2, 0))),
            plt.get_cmap('gray'), interpolation='bicubic', clim=(-1.0, +1.0))
        axes[i,5].imshow(utils.denormalize_image(
            np.transpose(recon[typical].data.numpy(), (1, 2, 0))),
            plt.get_cmap('gray'), interpolation='bicubic', clim=(-1.0, +1.0))

    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(110, 0),
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='large', ha='right', va='center')

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax in axes.ravel():
        ax.axis('off')

    plt.suptitle('Autoencoder reconstruction\n Best, Worst, Typical', size=20)
    plt.savefig("ae_per_class_perf.png", dpi=200, bbox_inches='tight')

def main():
    # data loaders
    _, va_loader, _, get_semantic_label = get_train_val_test_loaders(
        num_classes=config('autoencoder.num_classes'))
    dataset = get_data_by_label(va_loader)

    model = Autoencoder(config('autoencoder.ae_repr_dim'))
    criterion = torch.nn.MSELoss()

    # Attempts to restore the latest checkpoint if exists
    print('Loading autoencoder...')
    model, start_epoch, _ = restore_checkpoint(model,
        config('autoencoder.checkpoint'))

    # Evaluate model
    evaluate_autoencoder(dataset, get_semantic_label, model, criterion)

    # Report performance
    report_validation_performance(dataset, get_semantic_label, model, criterion)

if __name__ == '__main__':
    main()