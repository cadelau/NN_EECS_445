"""
EECS 445 - Introduction to Machine Learning
Winter 2019 - Project 2
Utility functions
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def config(attr):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.
    """
    if not hasattr(config, 'config'):
        with open('config.json') as f:
            config.config = eval(f.read())
    node = config.config
    for part in attr.split('.'):
        node = node[part]
    return node

def denormalize_image(image):
    """ Rescale the image's color space from (min, max) to (0, 1) """
    ptp = np.max(image, axis=(0,1)) - np.min(image, axis=(0,1))
    return (image - np.min(image, axis=(0,1))) / ptp

def hold_training_plot():
    """
    Keep the program alive to display the training plot
    """
    plt.ioff()
    plt.show()

def log_cnn_training(epoch, stats):
    """
    Logs the validation accuracy and loss to the terminal
    """
    valid_acc, valid_loss, train_acc, train_loss = stats[-1]
    print('Epoch {}'.format(epoch))
    print('\tValidation Loss: {}'.format(valid_loss))
    print('\tValidation Accuracy: {}'.format(valid_acc))
    print('\tTrain Loss: {}'.format(train_loss))
    print('\tTrain Accuracy: {}'.format(train_acc))

def make_cnn_training_plot(name='CNN'):
    """
    Runs the setup for an interactive matplotlib graph that logs the loss and
    accuracy
    """
    plt.ion()
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    plt.suptitle(name + ' Training')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')

    return axes

def update_cnn_training_plot(axes, epoch, stats):
    """
    Updates the training plot with a new data point for loss and accuracy
    """
    valid_acc = [s[0] for s in stats]
    valid_loss = [s[1] for s in stats]
    train_acc = [s[2] for s in stats]
    train_loss = [s[3] for s in stats]
    axes[0].plot(range(epoch - len(stats) + 1, epoch + 1), valid_acc,
        linestyle='--', marker='o', color='b')
    axes[0].plot(range(epoch - len(stats) + 1, epoch + 1), train_acc,
        linestyle='--', marker='o', color='r')
    axes[0].legend(['Validation', 'Train'])
    axes[1].plot(range(epoch - len(stats) + 1, epoch + 1), valid_loss,
        linestyle='--', marker='o', color='b')
    axes[1].plot(range(epoch - len(stats) + 1, epoch + 1), train_loss,
        linestyle='--', marker='o', color='r')
    axes[1].legend(['Validation', 'Train'])
    plt.pause(0.00001)

def save_cnn_training_plot(name='cnn'):
    """
    Saves the training plot to a file
    """
    plt.savefig(name + '_training_plot.png', dpi=200)

def log_ae_training(epoch, stats):
    """
    Logs the validation loss to the terminal
    """
    valid_loss, train_loss = stats[-1]
    print('Epoch {}'.format(epoch))
    print('\tValidation Mean squared error loss: {}'.format(valid_loss))
    print('\tTrain Mean squared error loss: {}'.format(train_loss))

def make_ae_training_plot():
    """
    Runs the setup for an interactive matplotlib graph that logs the loss
    """
    plt.ion()
    fig, axes = plt.subplots(1,1, figsize=(5,5))
    plt.suptitle('Autoencoder Training')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('MSE')

    return axes

def update_ae_training_plot(axes, epoch, stats):
    """
    Updates the training plot with a new data point for loss
    """
    valid_loss = [s[0] for s in stats]
    train_loss = [s[1] for s in stats]
    axes.plot(range(epoch - len(stats) + 1, epoch + 1), valid_loss,
        linestyle='--', marker='o', color='b')
    axes.plot(range(epoch - len(stats) + 1, epoch + 1), train_loss,
        linestyle='--', marker='o', color='r')
    axes.legend(['Validation', 'Train'])
    plt.pause(0.00001)

def save_ae_training_plot():
    """
    Saves the training plot to a file
    """
    plt.savefig('ae_training_plot.png', dpi=200)
