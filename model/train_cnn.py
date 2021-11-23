'''Training code for CNN'''
from torch.utils.data import DataLoader
import torch.nn as nn

from src.dataset import CaptionEmbeddingsDataset
from src.train import train, validate, save_checkpoint
from src.model import EmbeddingLearner
import config.cnn as cfg_cnn
import config.dataset as cfg_ds
import config.word2vec as cfg_w2v


# create train and val datasets
train_ds = CaptionEmbeddingsDataset(
    cfg_ds.SAMPLES_PATH+"train_embeddings.pkl", 
    cfg_ds.IMAGES_PATH,
    )
val_ds = CaptionEmbeddingsDataset(
    cfg_ds.SAMPLES_PATH+"val_embeddings.pkl", 
    cfg_ds.IMAGES_PATH,
    )

# get device type - needed for model, train and validate
device = cfg_cnn.DEVICE

# create DataLoaders
train_dataloader = DataLoader(
    train_ds,
    batch_size=cfg_cnn.BATCH_SIZE,
    shuffle=True,
    num_workers=cfg_cnn.NUM_WORKERS,
    pin_memory=cfg_cnn.PIN_MEMORY    
)
val_dataloader = DataLoader(
    val_ds,
    batch_size=cfg_cnn.BATCH_SIZE,
    shuffle=True,
    num_workers=cfg_cnn.NUM_WORKERS,
    pin_memory=cfg_cnn.PIN_MEMORY    
)

# define the CNN model
model = EmbeddingLearner(cfg_w2v.SIZE)
if device == "cuda:0":
    model = nn.DataParallel(model)

# training and validation loop
best_loss = 1000 # random large number
for epoch in range(1, cfg_cnn.EPOCHS+1):
    train_loss = train(train_dataloader, model, cfg_cnn.CRITERION, cfg_cnn.OPTIMIZER)
    val_loss = validate(val_dataloader, model, cfg_cnn.CRITERION)
    print(f"\n\nEpoch: {epoch}, Train loss: {train_loss}, Val loss: {val_loss}\n\n")
    if val_loss < best_loss:
        save_checkpoint(model, f"val_{val_loss}")
        best_loss = val_loss
