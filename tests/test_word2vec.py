'''behavioral and parameter tests for the word2vec model'''

import os
import pytest
from gensim.models import Word2Vec

import config.word2vec as wv_cfg
from tests.word2vec_case import NON_MATCHING_PAIRS, VOCAB_WORDS, STOP_WORDS, SIMILAR_WORDS


@pytest.fixture
def model():
    '''returns an instance of the trained Word2Vec model'''
    model_path = os.environ.get('WORD2VEC_MODEL_PATH') + "word2vec.model"
    w2v = Word2Vec.load(model_path)
    return w2v

# test model parameters
def test_vector_size(model):
    '''verify size of individual word vector'''
    assert model.wv.vector_size == wv_cfg.SIZE

def test_window_size(model):
    '''verify window size used for training'''
    assert model.window == wv_cfg.WINDOW

def test_epochs(model):
    '''verify number of epochs model was trained for'''
    assert model.epochs == wv_cfg.EPOCHS

# test model behavior
@pytest.mark.parametrize("present_word", VOCAB_WORDS)
def test_word_presence(model, present_word):
    '''verify the presence of high frequency words from our dataset'''
    assert present_word in model.wv.vocab

@pytest.mark.parametrize("stop_word", STOP_WORDS)
def test_word_absence(model, stop_word):
    '''verify the absence of stop words'''
    assert stop_word not in model.wv.vocab

@pytest.mark.parametrize("word1, word2", SIMILAR_WORDS)
def test_similarity(model, word1, word2):
    '''test if model scores similarity between words at > 0.5'''
    assert model.wv.similarity(word1, word2) > 0.5

@pytest.mark.parametrize("words_list, non_matching", NON_MATCHING_PAIRS)
def test_non_matching(model, words_list, non_matching):
    '''test if model correctly identifies the non-matching word'''
    assert model.wv.doesnt_match(words_list) == non_matching
