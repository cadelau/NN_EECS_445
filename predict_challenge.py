'''
EECS 445 - Introduction to Machine Learning
Winter 2019 - Project 2
Predict Challenge
    Runs the challenge model inference on the test dataset and saves the
    predictions to disk
    Usage: python predict_challenge.py --uniqname=<uniqname>
'''
import argparse
import torch
import numpy as np
import pandas as pd
import utils
from dataset import get_train_val_test_loaders
from model.challenge import Challenge
from train_common import *
from utils import config

def predict_challenge(data_loader, model):
    """
    Runs the model inference on the test set and outputs the predictions
    """
    model_pred = np.array([])
    for i, (X, y) in enumerate(data_loader):
        output = model(X)
        predicted = predictions(output.data)
        predicted = predicted.numpy()
        model_pred = np.concatenate([model_pred, predicted])
    return model_pred

def main(uniqname):
    # data loaders
    _, _, te_loader, get_semantic_label = get_train_val_test_loaders(
        num_classes=config('challenge.num_classes'))

    model = Challenge()

    # Attempts to restore the latest checkpoint if exists
    model, _, _ = restore_checkpoint(model, config('challenge.checkpoint'))

    # Evaluate model
    model_pred = predict_challenge(te_loader, model)

    print('saving challenge predictions...\n')
    model_pred = [get_semantic_label(p) for p in model_pred]
    pd_writer = pd.DataFrame(model_pred, columns=['predictions'])
    pd_writer.to_csv(uniqname + '.csv', index=False, header=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--uniqname', required=True)
    args = parser.parse_args()
    main(args.uniqname)