'''Tests for the Dataset class'''

import os
import pytest

from src.dataset import CaptionDataset
from tests.captiondataset_case import UNCLEAN_READ_RESULT, CLEAN_READ_RESULT
from tests.captiondataset_case import UNCLEAN_MIN_COUNT_3_RESULT
from tests.captiondataset_case import CLEAN_MIN_COUNT_2_RESULT


@pytest.fixture
def read_caption_dataset_unclean():
    '''read captions txt files and return unclean CaptionDataset'''
    dataset_path = os.environ.get('MIN_VIABLE_TESTING_DS_PATH')
    cap_ds = CaptionDataset(dataset_path)
    cap_ds.read_captions()
    return cap_ds


@pytest.fixture
def read_caption_dataset_clean():
    '''read captions txt files and return unclean CaptionDataset'''
    dataset_path = os.environ.get('MIN_VIABLE_TESTING_DS_PATH')
    cap_ds = CaptionDataset(dataset_path)
    cap_ds.read_captions(clean=True)
    return cap_ds


@pytest.fixture
def read_caption_dataset_unclean_min_count():
    '''read captions txt files and return unclean CaptionDataset'''
    dataset_path = os.environ.get('MIN_VIABLE_TESTING_DS_PATH')
    cap_ds = CaptionDataset(dataset_path)
    cap_ds.read_captions(min_count=3)
    return cap_ds


@pytest.fixture
def read_caption_dataset_clean_min_count():
    '''read captions txt files and return unclean CaptionDataset'''
    dataset_path = os.environ.get('MIN_VIABLE_TESTING_DS_PATH')
    cap_ds = CaptionDataset(dataset_path)
    cap_ds.read_captions(clean=True, min_count=2)
    return cap_ds


def test_caption_dataset_unclean_read():
    '''test if dataset was read correctly'''
    assert read_caption_dataset_unclean == UNCLEAN_READ_RESULT


def test_caption_dataset_clean_read():
    '''test if dataset was read and cleaned'''
    assert read_caption_dataset_clean == CLEAN_READ_RESULT


def test_caption_dataset_unclean_min_count_read():
    '''test if dataset only has words with count >= 3'''
    assert read_caption_dataset_unclean_min_count == UNCLEAN_MIN_COUNT_3_RESULT


def test_caption_dataset_clean_min_count_read():
    '''test if dataset is clean and only has words with count >= 2'''
    assert read_caption_dataset_clean_min_count == CLEAN_MIN_COUNT_2_RESULT