'''Training script for Word2Vec on the dataset's captions'''
from gensim.models import Word2Vec


ALL_CLEANED_CAPTIONS = None

model = Word2Vec(ALL_CLEANED_CAPTIONS, min_count=5)
model.save("word2vec_instacities_default.model")
