'''test the configuration path'''

from config.dataset import CAPTIONS_PATH

def test_captions_path():
    assert CAPTIONS_PATH == "D:\projects\multimodal-image-retrieval\datasets\raw\InstaNY100K\captions\newyork"