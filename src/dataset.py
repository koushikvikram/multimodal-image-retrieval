'''Module for creating and working with the InstaNY100K dataset'''
from typing import List
import glob
import re

import nltk
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')

class Dataset:
    '''Perform overall Dataset operations'''
    def __init__(self, captions_path: str, images_path: str):
        self.captions_path = captions_path
        self.images_path = images_path
        self.captions_dataset = {}
    def read_captions(self, clean=False):
        '''read all caption files'''
        filepaths = glob.glob(self.captions_path+"*.txt")
        all_captions = {}
        for path in filepaths:
            caption = Caption(path)
            caption.read(clean=clean)
            caption_id = caption.get_id()
            words = caption.get_data()
            all_captions[caption_id] = words
        self.__set_captions(all_captions)
    def make_caption_datasets(self, path, min_count=5, split_ratio=(0.8,0.05,0.15)):
        '''split captions into train, val and test sets and save them to path'''
        raise NotImplementedError
    def get_word2vec_dataset(self, min_count=5):
        '''make captions dataset for training word2vec'''
        raise NotImplementedError
    def __set_captions(self, captions):
        '''set self.captions_dataset to captions'''
        self.captions_dataset = captions
    def get_caption_embeddings(self):
        '''get a single vector representation from word2vec for each caption'''
        raise NotImplementedError


class Caption:
    '''Perform operations on individual caption textfiles'''
    def __init__(self, filepath: str):
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
    def get_embeddings(self):
        '''get word2vec embeddings of caption'''
        raise NotImplementedError
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
        with open(self._fpath, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    def __set_data(self, data: List[str]):
        '''assign list of words to method's _data variable'''
        self._data = data
