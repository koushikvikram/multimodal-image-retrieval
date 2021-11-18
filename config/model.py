'''Model objects for testing'''

import os
from gensim.models import Word2Vec


WORD2VEC_PATH = os.path.abspath("../model/word2vec.model")
WORD2VEC_MODEL = Word2Vec.load(WORD2VEC_PATH)
