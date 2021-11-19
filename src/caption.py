'''Module for working with a single caption file'''

from typing import List

import re
import nltk
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import word_tokenize


nltk.download('punkt')

class IncorrectFileFormat(Exception):
    '''Raise when file with wrong extension is given as input'''


class Caption:
    '''Perform operations on individual caption textfiles'''
    def __init__(self, filepath: str):
        if filepath.split(".")[-1] != "txt":
            raise IncorrectFileFormat("Please provide a file with .txt extension")
        self._fpath = filepath
        self._data = []
        self._id = self.get_id()
    def read(self, clean=False):
        '''read textfile and make words from captions'''
        text = self.__read_raw_text()
        if clean:
            # remove non-ascii characters
            ascii_text = text.encode("ascii", "ignore").decode()
            # remove non-alphanumeric characters
            pattern = re.compile(r'[^a-zA-Z\d\s]')
            alnum_text = pattern.sub('', ascii_text)
            # lowercase the text
            lowercase_text = alnum_text.lower()
            # remove stop words
            no_stop_words = remove_stopwords(lowercase_text)
            # word tokenize
            cleaned_words = word_tokenize(no_stop_words)
            # remove custom stopwords
            custom_stopwords = [
                'http', 'https', 'photo', 'picture',
                'image', 'insta', 'instagram', 'post']
            clean_captions = []
            for word in cleaned_words:
                if word not in custom_stopwords:
                    clean_captions.append(word)
            self.__set_data(clean_captions)
        else:
            words = text.split(" ")
            self.__set_data(words)
    def get_id(self):
        '''get caption's ID, present in filename'''
        filename = self._fpath.split("/")[-1]
        caption_id = filename.split(".")[0]
        return caption_id
    def get_data(self):
        '''returns list of words'''
        return self._data
    def __read_raw_text(self):
        '''read txt file as string'''
        print(f"Reading caption file: {self._fpath}")
        with open(self._fpath, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    def __set_data(self, data: List[str]):
        '''assign list of words to method's _data variable'''
        self._data = data
