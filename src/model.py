'''Vision Models'''
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as F


class EmbeddingLearner(nn.Module):
    '''ResNet50 for training'''
    def __init__(self, embedding_dim):
        super(EmbeddingLearner, self).__init__()
        self.model = resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, embedding_dim) # Not using Sigmoid because of Cross-Entropy loss
    def forward(self, image):
        embeddings = self.model(image)
        return embeddings


class Encoder(nn.Module):
    '''Prediction Model for Test Data'''
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.model = resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, embedding_dim)
    def forward(self, image):
        embeddings = self.model(image)
        embeddings = F.sigmoid(embeddings)
        return embeddings
