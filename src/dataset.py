from typing import List
import re

import nltk
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')

class Dataset:
    def __init__(self, captions_path: str, images_path: str):
        self.CAPTIONS_PATH = captions_path
        self.IMAGES_PATH = images_path
    def read_dataset(self):
        raise NotImplementedError
    def clean_dataset(self):
        raise NotImplementedError


class Caption:
    def __init__(self, filepath: str):
        self._fpath = filepath
        self._data = self.read()
        self._id = self.get_id()
    def read(self):
        raw_text = self.__read_raw_text()
        words = raw_text.split(" ")
        self.__set_data(words)
    def clean(self):
        text = self.__read_raw_text()
        # remove non-ascii characters
        ascii_text = text.encode("ascii", "ignore").decode()
        # remove non-alphanumeric characters
        pattern = re.compile('[^a-zA-Z\d\s]')
        alnum_text = pattern.sub('', ascii_text)
        # lowercase the text
        lowercase_text = alnum_text.lower()
        # remove stop words
        no_stop_words = remove_stopwords(lowercase_text)
        # word tokenize
        cleaned_words = word_tokenize(no_stop_words) 
        self.__set_data(cleaned_words)
    def get_id(self):
        filename = self._fpath.split("/")[-1]
        id = filename.split(".")[0]
        return id
    def get_data(self):
        return self._data
    def __read_raw_text(self):
        with open(self._fpath, 'r') as f:
            text = f.read()
        return text
    def __set_data(self, data: List[str]):
        self._data = data