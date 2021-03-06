'''Tests for the Dataset class'''

import os
import pytest

from src.dataset import CaptionDataset, EmptyDataset
from src.caption import IncorrectFileFormat
from tests.captiondataset_case import UNCLEAN_READ_RESULT, CLEAN_READ_RESULT
from tests.captiondataset_case import UNCLEAN_MIN_COUNT_3_RESULT
from tests.captiondataset_case import CLEAN_EMBEDDINGS_RESULT
from tests.captiondataset_case import WORD2VEC_DATASET_RESULT
from tests.captiondataset_case import CLEAN_SPLIT_NO_SHUFFLE


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
    assert read_caption_dataset_clean_min_count_2.get_captions() == CLEAN_READ_RESULT


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
    embeddings_1 = read_caption_dataset_clean.get_caption_embeddings()['000001']
    expected_embeddings_1 = CLEAN_EMBEDDINGS_RESULT['000001']
    embeddings_2 = read_caption_dataset_clean.get_caption_embeddings()['000002']
    expected_embeddings_2 = CLEAN_EMBEDDINGS_RESULT['000002']
    embeddings_3 = read_caption_dataset_clean.get_caption_embeddings()['000003']
    expected_embeddings_3 = CLEAN_EMBEDDINGS_RESULT['000003']
    match_1 = (embeddings_1 == expected_embeddings_1).all()
    match_2 = (embeddings_2 == expected_embeddings_2).all()
    match_3 = (embeddings_3 == expected_embeddings_3).all()
    assert match_1 and match_2 and match_3


def test_caption_embeddings_empty(read_caption_dataset_clean_min_count_3):
    '''test if EmptyDataset exception is raised'''
    with pytest.raises(EmptyDataset) as exceptioninfo:
        read_caption_dataset_clean_min_count_3.make_caption_embeddings()
    assert str(exceptioninfo.value) == "Captions dataset is empty."


def test_caption_embeddings_key_error(read_caption_dataset_unclean):
    '''test if KeyError is raised when stop words are present'''
    with pytest.raises(KeyError):
        read_caption_dataset_unclean.make_caption_embeddings()


def test_make_empty_word2vec_dataset(read_caption_dataset_clean_min_count_3):
    '''test if EmptyDataset is raised'''
    with pytest.raises(EmptyDataset) as exceptioninfo:
        read_caption_dataset_clean_min_count_3.make_word2vec_dataset()
    assert str(exceptioninfo.value) == "Captions dataset is empty."


def test_get_empty_word2vec_dataset(read_caption_dataset_clean):
    '''test if EmptyDataset is raised if .get_word2vec_dataset() called before make'''
    with pytest.raises(EmptyDataset) as exceptioninfo:
        read_caption_dataset_clean.get_word2vec_dataset()
    assert str(exceptioninfo.value) == "Empty dataset. Try calling .make_word2vec_dataset() first."


def test_word2vec_dataset_success(read_caption_dataset_clean):
    '''test if word2vec dataset was made correctly'''
    read_caption_dataset_clean.make_word2vec_dataset()
    assert read_caption_dataset_clean.get_word2vec_dataset() == WORD2VEC_DATASET_RESULT


def test_split_greater_than_1(read_caption_dataset_clean):
    '''test if ValueError is raised when train+val+split > 1.0'''
    with pytest.raises(ValueError) as exceptioninfo:
        read_caption_dataset_clean.get_split(
            ds_type='captions',
            train=1.0,
            val=0.2,
            test=0.8
            )
    assert str(exceptioninfo.value) == "Specify train, val and test to add up to 1.0"


def test_split_lesser_than_1(read_caption_dataset_clean):
    '''test if ValueError is raised when train+val+split < 1.0'''
    with pytest.raises(ValueError) as exceptioninfo:
        read_caption_dataset_clean.get_split(
            ds_type='captions',
            train=0.1,
            val=0.2,
            test=0.3
            )
    assert str(exceptioninfo.value) == "Specify train, val and test to add up to 1.0"


def test_split_ds_type_valueerror(read_caption_dataset_clean):
    '''test if ValueError is raised when ds_type is invalid'''
    with pytest.raises(ValueError) as exceptioninfo:
        read_caption_dataset_clean.get_split(
            ds_type='word2vec',
            train=0.8,
            val=0.05,
            test=0.15
            )
    assert str(exceptioninfo.value) == "ds_type should either be 'captions' or 'embeddings'"


def test_split_empty_dataset(read_caption_dataset_clean_min_count_3):
    '''test if EmptyDataset is raised'''
    with pytest.raises(EmptyDataset) as exceptioninfo:
        read_caption_dataset_clean_min_count_3.get_split(
            ds_type="captions",
            train=0.33,
            val=0.33,
            test=0.34
            )
    assert str(exceptioninfo.value) == "captions dataset is empty."


def test_split_captions(read_caption_dataset_clean):
    '''test if captions data is correctly split'''
    assert read_caption_dataset_clean.get_split(
        ds_type="captions",
        train=0.33,
        val=0.33,
        test=0.34
        ) == CLEAN_SPLIT_NO_SHUFFLE


def test_split_captions_only_train(read_caption_dataset_clean):
    '''test if captions data is correctly split'''
    assert read_caption_dataset_clean.get_split(
        ds_type="captions",
        train=1.0,
        val=0.0,
        test=0.0
        )[0] == CLEAN_READ_RESULT


def test_split_embeddings(read_caption_dataset_clean):
    '''test if embeddings dataset is correctly split'''
    read_caption_dataset_clean.make_caption_embeddings()
    embeddings_ds = read_caption_dataset_clean.get_caption_embeddings()
    train_ds, val_ds, test_ds = read_caption_dataset_clean.get_split(
        ds_type="embeddings",
        train=0.33,
        val=0.33,
        test=0.34,
        )
    train_key = list(train_ds.keys())[0]
    val_key = list(val_ds.keys())[0]
    test_key = list(test_ds.keys())[0]
    match_train = (train_ds[train_key] == embeddings_ds[train_key]).all()
    match_val = (val_ds[val_key] == embeddings_ds[val_key]).all()
    match_test = (test_ds[test_key] == embeddings_ds[test_key]).all()
    assert match_train and match_val and match_test


def test_write_empty_captions(read_caption_dataset_clean_min_count_3):
    '''test if EmptyDataset is raised'''
    file_path = os.environ.get('CAPTION_CHECKPOINT_PATH')
    with pytest.raises(EmptyDataset) as exceptioninfo:
        read_caption_dataset_clean_min_count_3.write_captions(file_path)
    assert str(exceptioninfo.value) == "Captions dataset is empty."


def test_write_captions_incorrect(read_caption_dataset_clean):
    '''test if IncorrectFileFormat is raised'''
    file_name = 'captions.txt'
    file_path = os.environ.get('CHECKPOINT_READ_PATH')
    with pytest.raises(IncorrectFileFormat) as exceptioninfo:
        read_caption_dataset_clean.write_captions(file_path+file_name)
    assert str(exceptioninfo.value) == "checkpoint should end in .pkl or .pickle"


def test_write_embeddings_incorrect(read_caption_dataset_clean):
    '''test if IncorrectFileFormat is raised'''
    file_name = 'embeddings.txt'
    file_path = os.environ.get('CHECKPOINT_READ_PATH')
    read_caption_dataset_clean.make_caption_embeddings()
    with pytest.raises(IncorrectFileFormat) as exceptioninfo:
        read_caption_dataset_clean.write_caption_embeddings(file_path+file_name)
    assert str(exceptioninfo.value) == "checkpoint should end in .pkl or .pickle"


def test_write_embeddings_incorrect(read_caption_dataset_clean):
    '''test if IncorrectFileFormat is raised'''
    file_name = 'word2vec.txt'
    file_path = os.environ.get('CHECKPOINT_READ_PATH')
    read_caption_dataset_clean.make_word2vec_dataset()
    with pytest.raises(IncorrectFileFormat) as exceptioninfo:
        read_caption_dataset_clean.write_word2vec_dataset(file_path+file_name)
    assert str(exceptioninfo.value) == "checkpoint should end in .pkl or .pickle"


def test_read_captions_checkpoint(read_caption_dataset_unclean):
    '''test if captions checkpoint is correctly read'''
    file_name = "captions.pkl"
    file_path = os.environ.get('CHECKPOINT_READ_PATH')
    read_caption_dataset_unclean.read_captions_checkpoint(file_path+file_name)
    assert read_caption_dataset_unclean.get_captions() == CLEAN_READ_RESULT


def test_read_captions_incorrect_path(read_caption_dataset_clean):
    '''test if IncorrectFileFormat is raised for incorrect extension'''
    file_name = "captions.txt"
    file_path = os.environ.get('CHECKPOINT_READ_PATH')
    with pytest.raises(IncorrectFileFormat) as exceptioninfo:
        read_caption_dataset_clean.read_captions_checkpoint(file_path+file_name)
    assert str(exceptioninfo.value) == "Please specify the correct path to pickle file"


def test_read_captions_incorrect_embeddings(read_caption_dataset_clean):
    '''test if IncorrectFileFormat is raised for embeddings pickle file'''
    file_name = "embeddings.pkl"
    file_path = os.environ.get('CHECKPOINT_READ_PATH')
    with pytest.raises(IncorrectFileFormat) as exceptioninfo:
        read_caption_dataset_clean.read_captions_checkpoint(file_path+file_name)
    assert str(exceptioninfo.value) == "dict value not List[str]: Possibly incorrect pickle file"


def test_read_embeddings_checkpoint(read_caption_dataset_clean):
    '''test if embeddings checkpoint is correctly read'''
    file_name = "embeddings.pkl"
    file_path = os.environ.get('CHECKPOINT_READ_PATH')
    read_caption_dataset_clean.read_caption_embeddings_checkpoint(file_path+file_name)
    embeddings_1 = read_caption_dataset_clean.get_caption_embeddings()['000001']
    expected_embeddings_1 = CLEAN_EMBEDDINGS_RESULT['000001']
    embeddings_2 = read_caption_dataset_clean.get_caption_embeddings()['000002']
    expected_embeddings_2 = CLEAN_EMBEDDINGS_RESULT['000002']
    embeddings_3 = read_caption_dataset_clean.get_caption_embeddings()['000003']
    expected_embeddings_3 = CLEAN_EMBEDDINGS_RESULT['000003']
    match_1 = (embeddings_1 == expected_embeddings_1).all()
    match_2 = (embeddings_2 == expected_embeddings_2).all()
    match_3 = (embeddings_3 == expected_embeddings_3).all()
    assert match_1 and match_2 and match_3


def test_read_embeddings_incorrect_path(read_caption_dataset_clean):
    '''test if IncorrectFileFormat is raised for incorrect extension'''
    file_name = "embeddings.txt"
    file_path = os.environ.get('CHECKPOINT_READ_PATH')
    with pytest.raises(IncorrectFileFormat) as exceptioninfo:
        read_caption_dataset_clean.read_caption_embeddings_checkpoint(file_path+file_name)
    assert str(exceptioninfo.value) == "Please specify the correct path to pickle file"


def test_read_embeddings_incorrect_captions(read_caption_dataset_clean):
    '''test if IncorrectFileFormat is raised for captions pickle file'''
    file_name = "captions.pkl"
    file_path = os.environ.get('CHECKPOINT_READ_PATH')
    with pytest.raises(IncorrectFileFormat) as exceptioninfo:
        read_caption_dataset_clean.read_caption_embeddings_checkpoint(file_path+file_name)
    assert str(exceptioninfo.value) == "dict value not np.array[np.float32]: Possibly incorrect pickle file"
