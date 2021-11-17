'''Module for creating and working with the InstaNY100K dataset'''
from typing import List
import glob
from collections import Counter
from tqdm import tqdm
import pickle

import re
import nltk
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import word_tokenize


nltk.download('punkt')


class EmptyDataset(Exception):
    pass

class IncorrectFileFormat(Exception):
    pass

class Dataset:
    '''Perform overall Dataset operations'''
    def __init__(self, captions_path: str, images_path: str):
        self.captions_path = captions_path
        self.images_path = images_path
        self.captions_dataset = {}
        self.high_freq_dataset = {}
        self.word2vec_dataset = []
        self.clean = False
        self.caption_freq_count = None
    def is_clean(self):
        '''return True if captions were cleaned, else False'''
        return self.clean
    def read_captions(self, clean=False):
        '''read all caption files'''
        self.clean = clean
        filepaths = glob.glob(self.captions_path+"*.txt")
        if len(filepaths) == 0:
            raise EmptyDataset("No .txt files found")
        all_captions = {}
        for path in tqdm(filepaths):
            caption = Caption(path)
            caption.read(clean=clean)
            caption_id = caption.get_id()
            words = caption.get_data()
            if len(words) > 0:
                all_captions[caption_id] = words
        self.__set_captions(all_captions)
    def read_captions_checkpoint(self, checkpoint):
        '''read previously stored .pkl captions (list of words) files from checkpoint path'''
        try:
            with open(checkpoint, 'rb') as file:
                all_captions = pickle.load(file)
        except:
            raise IncorrectFileFormat("Please specify the correct path to pickle file")
        if not isinstance(type(list(all_captions.values())[0][0]), str):
            raise IncorrectFileFormat("dict.values() is not List[str]: Possibly incorrect pickle file")
        self.__set_captions(all_captions)
    def read_caption_embeddings_checkpoint(self, checkpoint):
        '''read caption embeddings files'''
        raise NotImplementedError
    def make_high_frequency_captions(self, min_count=5):
        '''make captions dataset with only words having count >= min_count'''
        # get all captions
        all_captions = self.get_captions()
        if len(all_captions) == 0:
            raise EmptyDataset("Empty dict: Check if .read_captions() was called")
        # set self.min_count
        self.caption_freq_count = min_count
        # get all words from the captions
        all_words = []
        print("Getting all words ...")
        for words in tqdm(all_captions.values()):
            all_words.extend(list(words))
        # get counts for each word
        word_counts = Counter(all_words)
        # make a new dict and store captions with words having a count>=5
        high_freq_captions = {}
        print("Getting High-Frequency Captions ...")
        for id, caption in tqdm(all_captions):
            high_freq_words = []
            for word in caption:
                if word_counts[word] >= min_count:
                    high_freq_words.append(word)
            if len(high_freq_words) > 0:
                high_freq_captions[id] = high_freq_words
        self.high_freq_dataset = high_freq_captions
    def make_word2vec_dataset(self, min_count=5):
        '''make captions dataset for training word2vec'''
        # check if high frequency dataset has already been created
        # and if high frequency dataset was created with same min_count
        if self.caption_freq_count and self.caption_freq_count == min_count:
            high_freq_captions = self.get_high_frequency_captions()
        else:
            self.make_high_frequency_captions(min_count=min_count)
            high_freq_captions = self.get_high_frequency_captions()
        word2vec_dataset = []
        for words in high_freq_captions.values():
            word2vec_dataset.append(list(words))
        self.word2vec_dataset = word2vec_dataset
    def get_captions(self):
        '''get words list for each caption along with their id'''
        return self.captions_dataset
    def get_caption_embeddings(self, model):
        '''get a single vector representation from word2vec for each caption'''
        raise NotImplementedError
    def get_high_frequency_captions(self):
        '''get high frequency captions dataset'''
        return self.high_freq_dataset
    def get_word2vec_dataset(self):
        '''returns a word2vec dataset'''
        if len(self.word2vec_dataset) == 0:
            raise EmptyDataset("Empty dataset. Try calling .make_word2vec_dataset() first")
        return self.word2vec_dataset
    def split(self, ds_type, train, val, test):
        '''split dataset into train, val and test sets'''
        raise NotImplementedError
    def write_captions(self, checkpoint):
        '''write captions (list of words) to checkpoint path'''
        if checkpoint.split(".")[-1] not in ["pkl", "pickle"]:
            raise IncorrectFileFormat("checkpoint should end in .pkl or .pickle")
        all_captions = self.get_captions()
        if len(all_captions) == 0:
            raise EmptyDataset("Captions dataset is empty.")
        with open(checkpoint, "wb") as file:
            pickle.dump(all_captions, file)
    def write_caption_embeddings(self, checkpoint):
        '''write caption embeddings to checkpoint path'''
        raise NotImplementedError
    def write_high_frequency_captions(self, checkpoint):
        '''write high frequency captions to checkpoint path'''
        if checkpoint.split(".")[-1] not in ["pkl", "pickle"]:
            raise IncorrectFileFormat("checkpoint should end in .pkl or .pickle")
        high_freq_captions = self.get_high_frequency_captions()
        if len(high_freq_captions) == 0:
            raise EmptyDataset("High Frequency Captions dataset is empty.")
        with open(checkpoint, "wb") as file:
            pickle.dump(high_freq_captions, file)
    def write_word2vec_dataset(self, checkpoint):
        '''write Word2Vec dataset to checkpoint path'''
        if checkpoint.split(".")[-1] not in ["pkl", "pickle"]:
            raise IncorrectFileFormat("checkpoint should end in .pkl or .pickle")
        word2vec_dataset = self.get_word2vec_dataset()
        if len(word2vec_dataset) == 0:
            raise EmptyDataset("Word2Vec dataset is empty.")
        with open(checkpoint, "wb") as file:
            pickle.dump(word2vec_dataset, file)
    def __set_captions(self, captions):
        '''set self.captions_dataset to captions'''
        self.captions_dataset = captions

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
