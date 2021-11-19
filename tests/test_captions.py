'''Testing the functionality of the Caption class'''

import os
import pytest

from src.dataset import Caption
from tests.caption_case import FILE_NAMES, EXCEPTION_FILES, RAW_CAPTIONS, CLEAN_CAPTIONS, FILE_ID


@pytest.fixture
def get_raw_captions(filepath):
    '''returns a list of uncleaned words'''
    dataset_path = os.environ.get(TESTING_CAPTIONS_DATASET_PATH)
    cap = Caption(dataset_path + filepath)
    cap.read()
    return cap.get_data()

@pytest.fixture
def get_clean_captions(filepath):
    '''returns a list of cleaned words'''
    dataset_path = os.environ.get(TESTING_CAPTIONS_DATASET_PATH)
    cap = Caption(dataset_path + filepath)
    cap.read(clean=True)
    return cap.get_data()

@pytest.fixture
def get_caption_id(filepath):
    '''returns the id of the caption file'''
    cap = Caption(filepath)
    return cap.get_id()

@pytest.mark.parametrize(
    "filepath, raw_captions",
    list(zip(FILE_NAMES, RAW_CAPTIONS)),
    )
def test_raw_captions(get_raw_captions, raw_captions):
    '''test if captions are read correctly without cleaning'''
    assert get_raw_captions == raw_captions
