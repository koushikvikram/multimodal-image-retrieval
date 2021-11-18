'''behavioral and functional tests for the word2vec model'''

import os
import pytest
from gensim.models import Word2Vec

import config.word2vec as wv_cfg


MODEL_PATH = os.path.abspath("../model/word2vec.model")

@pytest.fixture
def model():
    '''trained word2vec model'''
    w2v = Word2Vec.load(MODEL_PATH)
    return w2v

def test_vector_size(model):
    '''verify size of individual word vector'''
    assert model.wv.vector_size == wv_cfg.SIZE

def test_window_size(model):
    '''verify window size used for training'''
    assert model.window == wv_cfg.WINDOW

def test_epochs(model):
    '''verify number of epochs model was trained for'''
    assert model.iter == wv_cfg.EPOCHS
