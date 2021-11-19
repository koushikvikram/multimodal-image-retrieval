'''Module for computing caption embeddings'''

import os
import pickle
from typing import List

import numpy as np


MODEL_PATH = os.environ.get('WORD2VEC_MODEL_PATH') + "word2vec.model"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


def compute_embedding(words: List[str]):
    '''computer and return caption embedding by taking the mean of words
    and normalizing it'''
    try:
        embedding = np.mean(model[words], axis=0)
    except KeyError:
        print("Hint: Check dataset clean and min_count. Check MODEL_PATH.")
        raise
    if min(embedding) < 0:
        embedding = embedding - min(embedding)
    if max(embedding) > 0:
        embedding = embedding/max(embedding)
    return embedding
