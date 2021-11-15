'''Training script for Word2Vec on the dataset's captions'''
from gensim.models import Word2Vec
import config.word2vec as cfg


WORD2VEC_DATASET = None

model = Word2Vec(
    WORD2VEC_DATASET,
    size=cfg.SIZE,
    min_count=cfg.MIN_COUNT,
    workers=cfg.N_CORES,
    iter=cfg.EPOCHS,
    window=cfg.WINDOW)
model.save("word2vec_instacities.model")
