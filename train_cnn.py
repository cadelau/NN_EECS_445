"""
EECS 445 - Introduction to Machine Learning
Winter 2019 - Project 2
Train CNN
    Trains a convolutional neural network to classify images
    Periodically outputs training information, and saves model checkpoints
    Usage: python train_cnn.py
"""
import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
from model.cnn import CNN
from train_common import *
from utils import config
import utils

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    for i, (X, y) in enumerate(data_loader):
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def _evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch,
    stats):
    """
    Evaluates the `model` on the train and validation set.
    """
    y_true, y_pred = [], []
    correct, total = 0, 0
    running_loss = []
    for X, y in tr_loader:
        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
    train_loss = np.mean(running_loss)
    train_acc = correct / total
    y_true, y_pred = [], []
    correct, total = 0, 0
    running_loss = []
    for X, y in val_loader:
        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
    val_loss = np.mean(running_loss)
    val_acc = correct / total
    stats.append([val_acc, val_loss, train_acc, train_loss])
    utils.log_cnn_training(epoch, stats)
    utils.update_cnn_training_plot(axes, epoch, stats)

def main():
    # Data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        num_classes=config('cnn.num_classes'))

    # Model
    model = CNN()

    # TODO: define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    #

    print('Number of float-valued parameters:', count_parameters(model))

    # Attempts to restore the latest checkpoint if exists
    print('Loading cnn...')
    model, start_epoch, stats = restore_checkpoint(model,
        config('cnn.checkpoint'))

    axes = utils.make_cnn_training_plot()

    # Evaluate the randomly initialized model
    _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, start_epoch,
        stats)
    
    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config('cnn.num_epochs')):
        # Train model
        _train_epoch(tr_loader, model, criterion, optimizer)
        
        # Evaluate model
        _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, epoch+1,
            stats)

        # Save model parameters
        save_checkpoint(model, epoch+1, config('cnn.checkpoint'), stats)

    print('Finished Training')

    # Save figure and keep plot open
    utils.save_cnn_training_plot()
    utils.hold_training_plot()

if __name__ == '__main__':
    main()