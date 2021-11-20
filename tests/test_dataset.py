'''Tests for the Dataset class'''

import os
import pytest

from src.dataset import CaptionDataset, EmptyDataset
from tests.captiondataset_case import UNCLEAN_READ_RESULT, CLEAN_READ_RESULT
from tests.captiondataset_case import UNCLEAN_MIN_COUNT_3_RESULT
from tests.captiondataset_case import CLEAN_MIN_COUNT_2_RESULT
from tests.captiondataset_case import CLEAN_EMBEDDINGS_RESULT


@pytest.fixture
def read_caption_dataset_unclean():
    '''read and return unclean CaptionDataset'''
    dataset_path = os.environ.get('MIN_VIABLE_TESTING_DS_PATH')
    cap_ds = CaptionDataset(dataset_path)
    cap_ds.read_captions()
    return cap_ds


@pytest.fixture
def read_caption_dataset_clean():
    '''read return clean CaptionDataset'''
    dataset_path = os.environ.get('MIN_VIABLE_TESTING_DS_PATH')
    cap_ds = CaptionDataset(dataset_path)
    cap_ds.read_captions(clean=True)
    return cap_ds


@pytest.fixture
def read_caption_dataset_unclean_min_count_3():
    '''read and return unclean CaptionDataset with min_count 3'''
    dataset_path = os.environ.get('MIN_VIABLE_TESTING_DS_PATH')
    cap_ds = CaptionDataset(dataset_path)
    cap_ds.read_captions(min_count=3)
    return cap_ds


@pytest.fixture
def read_caption_dataset_clean_min_count_2():
    '''read and return clean CaptionDataset with min_count 2'''
    dataset_path = os.environ.get('MIN_VIABLE_TESTING_DS_PATH')
    cap_ds = CaptionDataset(dataset_path)
    cap_ds.read_captions(clean=True, min_count=2)
    return cap_ds


@pytest.fixture
def read_caption_dataset_clean_min_count_3():
    '''read and return clean CaptionDataset with min_count 3'''
    dataset_path = os.environ.get('MIN_VIABLE_TESTING_DS_PATH')
    cap_ds = CaptionDataset(dataset_path)
    cap_ds.read_captions(clean=True, min_count=3)
    return cap_ds


@pytest.fixture
def read_empty_dataset():
    '''read empty dataset'''
    dataset_path = os.environ.get('EMPTY_DS_PATH')
    cap_ds = CaptionDataset(dataset_path)
    return cap_ds
    

def test_caption_dataset_unclean_read(read_caption_dataset_unclean):
    '''test if dataset was read correctly'''
    assert read_caption_dataset_unclean.get_captions() == UNCLEAN_READ_RESULT


def test_caption_dataset_clean_read(read_caption_dataset_clean):
    '''test if dataset was read and cleaned'''
    assert read_caption_dataset_clean.get_captions() == CLEAN_READ_RESULT


def test_caption_dataset_unclean_min_count_read(read_caption_dataset_unclean_min_count_3):
    '''test if dataset with only words with count >= 3'''
    assert read_caption_dataset_unclean_min_count_3.get_captions() == UNCLEAN_MIN_COUNT_3_RESULT


def test_caption_dataset_clean_min_count_read(read_caption_dataset_clean_min_count_2):
    '''test if dataset is clean with only words with count >= 2'''
    assert read_caption_dataset_clean_min_count_2.get_captions() == CLEAN_MIN_COUNT_2_RESULT


def test_caption_dataset_clean_min_count_3_read(read_caption_dataset_clean_min_count_3):
    '''test if dataset is clean with only words with count >= 3'''
    assert read_caption_dataset_clean_min_count_3.get_captions() == {}


def test_is_clean_result(read_caption_dataset_clean):
    '''test if .is_clean() returns True when clean=True is set'''
    assert read_caption_dataset_clean.is_clean() == True


def test_read_empty_dataset(read_empty_dataset):
    '''test if EmptyDataset Exception is raised when calling .read_captions()'''
    with pytest.raises(EmptyDataset) as exceptioninfo:
        read_empty_dataset.read_captions()
    assert str(exceptioninfo.value) == "No .txt files found"


def test_caption_embeddings_success(read_caption_dataset_clean):
    '''test if expected caption embeddings are generated'''
    read_caption_dataset_clean.make_caption_embeddings()
    assert read_caption_dataset_clean.get_caption_embeddings() == CLEAN_EMBEDDINGS_RESULT