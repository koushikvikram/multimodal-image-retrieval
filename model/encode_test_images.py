'''Pass Test Images through trained CNN and get embeddings'''
import pickle

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import config.dataset as cfg_ds
import config.word2vec as cfg_w2v
import config.cnn as cfg_cnn
from src.dataset import TestImagesDataset
from src.model import Encoder
from src.train import encode


# create test dataset
test_ds = TestImagesDataset(
    cfg_ds.SAMPLES_PATH+"test_embeddings.pkl",
    cfg_ds.IMAGES_PATH,
    )

# get device
device = cfg_cnn.DEVICE

# create data loader for testing
test_dataloader = DataLoader(
    test_ds,
    batch_size=cfg_cnn.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg_cnn.NUM_WORKERS,
    pin_memory=cfg_cnn.PIN_MEMORY    
)

# create model
state_dict = torch.load(cfg_cnn.MODEL_FILE_PATH)
test_model = Encoder(embedding_dim=cfg_w2v.SIZE)
if device == "cuda:0":
    test_model = nn.DataParallel(test_model)
test_model.load_state_dict(state_dict, strict=True)

# generate embeddings
test_embeddings = encode(test_dataloader, test_model)

# dump embeddings
with open("../application-files/predicted_embeddings.pkl", "wb") as file:
    pickle.dump(test_embeddings, file)
