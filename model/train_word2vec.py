'''Training script for Word2Vec on the dataset's captions'''
from src.dataset import Dataset
from gensim.models import Word2Vec
import config.word2vec as cfg


# # prepare dataset for word2vec
# dataset = Dataset(
#     captions_path=cfg.CAPTIONS_PATH,
#     images_path=""
#     )
# dataset.read_captions(clean=True)

# we've already cleaned the captions
WORD2VEC_DATASET = dataset.get_word2vec_dataset(min_count=cfg.MIN_COUNT)

# train word2vec model
model = Word2Vec(
    WORD2VEC_DATASET,
    size=cfg.SIZE,
    min_count=cfg.MIN_COUNT,
    workers=cfg.N_CORES,
    iter=cfg.EPOCHS,
    window=cfg.WINDOW)
model.save("word2vec_instacities.model")
