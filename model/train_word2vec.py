'''Training script for Word2Vec on the dataset's captions'''
from gensim.models import Word2Vec


ALL_CLEANED_CAPTIONS = None
SIZE = 100
MIN_COUNT = 5
N_CORES = 8
EPOCHS = 10
WINDOW = 8

model = gensim.models.Word2Vec(
    ALL_CLEANED_CAPTIONS, 
    size=SIZE, min_count=MIN_COUNT, 
    workers=N_CORES, 
    iter=EPOCHS, 
    window=WINDOW)
model.save("word2vec_instacities.model")
