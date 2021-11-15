'''Training script for Word2Vec on the dataset's captions'''
import gensim
from gensim.models import Word2Vec


all_cleaned_captions = None

model = Word2Vec(all_cleaned_captions, min_count=5)
model.save("word2vec_instacities_default.model")
