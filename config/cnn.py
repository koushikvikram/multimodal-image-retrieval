'''Configurations for training the CNN'''
import os

import torch
import torch.nn as nn


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.01
CRITERION = nn.BCEWithLogitsLoss().to(DEVICE)
OPTIMIZER = torch.optim.Adam(model.parameters(), LEARNING_RATE)
BATCH_SIZE = 96
EPOCHS = 15
NUM_WORKERS = 2
PIN_MEMORY = True
MODEL_FILE_PATH = os.path.abspath("../model/val_0.006941943121035761.pth.tar")
