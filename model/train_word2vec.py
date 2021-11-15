'''Training script for Word2Vec on the dataset's captions'''
from gensim.models import Word2Vec
import config.word2vec as cfg


ALL_CLEANED_CAPTIONS = None

model = Word2Vec(
    ALL_CLEANED_CAPTIONS,
    size=cfg.SIZE, 
    min_count=cfg.MIN_COUNT,
    workers=cfg.N_CORES,
    iter=cfg.EPOCHS,
    window=cfg.WINDOW)
model.save("word2vec_instacities.model")
