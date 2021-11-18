'''behavioral and functional tests for the word2vec model'''

import os
import pytest
from gensim.models import Word2Vec

import config.word2vec as wv_cfg
import config.model as model_cfg


@pytest.fixture
def model():
    '''returns an instance of the trained Word2Vec model'''
    return model_cfg.WORD2VEC_MODEL

def test_vector_size(model):
    '''verify size of individual word vector'''
    assert model.wv.vector_size == wv_cfg.SIZE

def test_window_size(model):
    '''verify window size used for training'''
    assert model.window == wv_cfg.WINDOW

def test_epochs(model):
    '''verify number of epochs model was trained for'''
    assert model.iter == wv_cfg.EPOCHS
