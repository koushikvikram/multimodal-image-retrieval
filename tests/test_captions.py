'''Testing the functionality of the Caption class'''

import os
import pytest
from src.caption import IncorrectFileFormat

from src.dataset import Caption
from tests.caption_case import FILE_NAMES, INCORRECT_EXT_FILES, NON_EXISTING_FILES, RAW_CAPTIONS, CLEAN_CAPTIONS, FILE_ID


@pytest.fixture
def get_raw_captions(filepath):
    '''returns a list of uncleaned words'''
    dataset_path = os.environ.get('TESTING_CAPTIONS_DATASET_PATH')
    cap = Caption(dataset_path + filepath)
    cap.read()
    return cap.get_data()

@pytest.fixture
def get_clean_captions(filepath):
    '''returns a list of cleaned words'''
    dataset_path = os.environ.get('TESTING_CAPTIONS_DATASET_PATH')
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

@pytest.mark.parametrize(
    'filepath, clean_captions',
    list(zip(FILE_NAMES, CLEAN_CAPTIONS))
)
def test_clean_captions(get_clean_captions, clean_captions):
    '''test if captions are being cleaned'''
    assert get_clean_captions == clean_captions

@pytest.mark.parametrize(
    'filepath, caption_id',
    list(zip(FILE_NAMES, FILE_ID))
)
def test_caption_id(get_caption_id, caption_id):
    '''check if the caption file's ID is correctly extracted'''
    assert get_caption_id == caption_id

@pytest.mark.parametrize(
    'file_path',
    INCORRECT_EXT_FILES
)
def test_incorrect_file_format(file_path):
    '''check if exceptions are raised correctly on wrong file formats 
    and files without extensions'''
    dataset_path = os.environ.get('TESTING_CAPTIONS_DATASET_PATH')
    with pytest.raises(IncorrectFileFormat) as exceptioninfo:
        Caption(dataset_path + file_path)
    assert str(exceptioninfo.value) == "Please provide a file with .txt extension"

@pytest.mark.parametrize(
    'file_path',
    NON_EXISTING_FILES
)
def test_non_existing_file(file_path):
    '''check if exceptions are raised correctly on file not present'''
    dataset_path = os.environ.get('TESTING_CAPTIONS_DATASET_PATH')
    with pytest.raises(FileNotFoundError):
        Caption(dataset_path + file_path)
