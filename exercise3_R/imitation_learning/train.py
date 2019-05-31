# from __future__ import print_function

import sys
sys.path.append("../") 

import pickle
import numpy as np
import os
import gzip
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import metrics
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation


def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray()
    # from utils.py, the output shape (96, 96, 1) 2. you can train your model with discrete actions (as you get them
    # from read_data) by discretizing the action space using action_to_id() from utils.py.
    print('.... preprocessing')

    X_train = np.array([rgb2gray(x) for x in X_train])
    X_valid = np.array([rgb2gray(x) for x in X_valid])

    y_train = np.array([action_to_id(y) for y in y_train])
    y_valid = np.array([action_to_id(y) for y in y_valid])

    # Removing 'score' information - it is probably noise??
    X_valid[:, 85:, :15] = 0.0
    X_train[:, 85:, :15] = 0.0

    # TODO History:
    # At first you should only use the current image as input to your network to learn the next action.
    # Then the input states have shape (96, 96, 1). Later, add a history of the last N images to your state so that
    # a state has shape (96, 96, N).
    if history_length >= 1:

        X_train = np.array([X_train[(i - history_length):i]
                            for i in range(history_length, X_train.shape[0] + 1)])
        X_valid = np.array([X_valid[(i - history_length):i]
                            for i in range(history_length, X_valid.shape[0] + 1)])
        # X_train = np.array([np.stack((X_train[(i - history_length):i]), axis=2)
        #                     for i in range(history_length, X_train.shape[0] + 1)])
        # X_valid = np.array([np.stack((X_valid[(i - history_length):i]), axis=2)
        #                     for i in range(history_length, X_valid.shape[0] + 1)])
        # filtering labels to the corresponding X values
        y_train = y_train[history_length-1:]
        y_valid = y_valid[history_length-1:]
    else:
        raise ValueError('History cannot be 0!')

    return X_train, y_train, X_valid, y_valid


def get_data_loader(X, y, batch_size, shuffle=True):
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr,
                history_length=1, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
 
    print("... train model")

    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    agent = BCAgent(device='cuda', history_length=history_length, lr=lr, n_classes=5)

    idx = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_eval = Evaluation(os.path.join(tensorboard_dir, "Imitation"), name='Car Racing', idx=idx,
                                  stats=['loss', 'train_acc', 'valid_acc'])

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard.
    # You can watch the progress of your training *during* the training in your web browser

    train_loader = get_data_loader(X_train, y_train, batch_size=batch_size, shuffle=False)
    valid_loader = get_data_loader(X_valid, y_valid, batch_size=batch_size, shuffle=False)

    # training loop
    for i in range(n_minibatches):
        # sample 1 batch from dataset
        ids = np.random.choice(list(range(X_train.shape[0])), size=batch_size)
        X_batch = torch.tensor(X_train[ids])
        y_batch = torch.tensor(y_train[ids])
        # run update
        loss = agent.update(X_batch, y_batch)

        if (i+1) % 10 == 0:
            # compute training/ validation accuracy and write it to tensorboard
            # predict on all data
            train_pred = []
            valid_pred = []
            for X, y in train_loader:
                pred = agent.predict(X)
                train_pred = np.append(train_pred, pred)
            for X, y in valid_loader:
                pred = agent.predict(X)
                valid_pred = np.append(valid_pred, pred)

            # accuracy
            train_acc = metrics.accuracy_score(y_true=y_train, y_pred=train_pred)
            valid_acc = metrics.accuracy_score(y_true=y_valid, y_pred=valid_pred)

            print('Epoch [%d/%d] loss: %.4f  train & valid accuracies: %.4f  %.4f' %
                  (i+1, n_minibatches, loss.item(), train_acc, valid_acc))
            tensorboard_eval.write_episode_data(i, {'loss': loss.item(), 'train_acc': train_acc,
                                                    'valid_acc': valid_acc})

    # TODO: save your agent
    model_dir = model_dir+"agent_"+idx+".pt"
    agent.save(model_dir)
    print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, help="history length in model", required=True)

    args = parser.parse_args()


    print(args)

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    history_length = args.history

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid,
                                                       history_length=history_length)

    # train model (you can change the parameters!)
    model_dir = './models/'
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=1000, batch_size=64, lr=1e-4,
                history_length=history_length, model_dir=model_dir, tensorboard_dir='../tensorboard')
