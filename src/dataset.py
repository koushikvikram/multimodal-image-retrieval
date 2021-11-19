'''Module for creating and working with the InstaNY100K dataset'''

import glob
from collections import Counter
import pickle
import random
from math import ceil
from tqdm import tqdm

from src.embedding import compute_embedding
from src.caption import Caption
from src.caption import IncorrectFileFormat


class EmptyDataset(Exception):
    '''Raise when operation is called on a dataset before it has been created'''


class CaptionDataset:
    '''Perform operations on Captions Dataset'''
    def __init__(self, captions_path=None):
        self.captions_path = captions_path
        self.captions_dataset = {}
        self.word2vec_dataset = []
        self.caption_embeddings_dataset = {}
        self.clean = False
    def is_clean(self):
        '''return True if captions were cleaned, else False'''
        return self.clean
    def read_captions(self, clean=False, min_count=0):
        '''read all caption files'''
        if self.captions_path is None:
            raise ValueError("arg: captions_path is None")
        filepaths = glob.glob(self.captions_path+"*.txt")
        if len(filepaths) == 0:
            raise EmptyDataset("No .txt files found")
        self.clean = clean
        print(f"Reading captions from directory: {self.captions_path}")
        all_captions = {}
        for path in tqdm(filepaths):
            caption = Caption(path)
            caption.read(clean=clean)
            caption_id = caption.get_id()
            words = caption.get_data()
            if len(words) > 0:
                all_captions[caption_id] = words
        if min_count > 0:
            # keep only words in captions with count >= min_count
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
            for caption_id, caption in tqdm(all_captions.items()):
                high_freq_words = []
                for word in caption:
                    if word_counts[word] >= min_count:
                        high_freq_words.append(word)
                if len(high_freq_words) > 0:
                    high_freq_captions[caption_id] = high_freq_words
            self.__set_captions(high_freq_captions)
        else:
            self.__set_captions(all_captions)
    def read_captions_checkpoint(self, checkpoint):
        '''read previously stored .pkl captions (list of words) files from checkpoint path'''
        print(f"Reading captions from checkpoint: {checkpoint}")
        try:
            with open(checkpoint, 'rb') as file:
                all_captions = pickle.load(file)
        except:
            raise IncorrectFileFormat("Please specify the correct path to pickle file")
        if not isinstance(list(all_captions.values())[0][0], str):
            raise IncorrectFileFormat("dict value is not List[str]: Possibly incorrect pickle file")
        self.__set_captions(all_captions)
    def read_caption_embeddings_checkpoint(self, checkpoint):
        '''read previously stored caption embeddings .pkl files from checkpoint path'''
        print(f"Reading caption embeddings from checkpoint: {checkpoint}")
        try:
            with open(checkpoint, 'rb') as file:
                all_caption_embeddings = pickle.load(file)
        except:
            raise IncorrectFileFormat("Please specify the correct path to pickle file")
        if not isinstance(list(all_caption_embeddings.values())[0][0], float):
            raise IncorrectFileFormat("dict value not List[float]: Possibly incorrect file")
        self.caption_embeddings_dataset = all_caption_embeddings
    def make_caption_embeddings(self):
        '''make a single embedding for each caption'''
        all_captions = self.get_captions()
        if len(all_captions) == 0:
            raise EmptyDataset("Captions dataset is empty.")
        print("Making embeddings ...")
        embeddings_dataset = {}
        for caption_id, words in tqdm(all_captions.items()):
            embeddings_dataset[caption_id] = compute_embedding(words)
        self.caption_embeddings_dataset = embeddings_dataset
    def make_word2vec_dataset(self):
        '''make captions dataset for training word2vec'''
        # check if high frequency dataset has already been created
        # and if high frequency dataset was created with same min_count
        all_captions = self.get_captions()
        if len(all_captions) == 0:
            raise EmptyDataset('Captions dataset is empty')
        print("Making Word2Vec dataset ...")
        word2vec_dataset = []
        for words in tqdm(all_captions.values()):
            word2vec_dataset.append(list(words))
        self.word2vec_dataset = word2vec_dataset
    def get_captions(self):
        '''return words list for each caption along with their id'''
        return self.captions_dataset
    def get_caption_embeddings(self):
        '''return embeddings for each caption along with their id'''
        return self.caption_embeddings_dataset
    def get_word2vec_dataset(self):
        '''returns a word2vec dataset'''
        if len(self.word2vec_dataset) == 0:
            raise EmptyDataset("Empty dataset. Try calling .make_word2vec_dataset() first")
        return self.word2vec_dataset
    def get_split(self, ds_type, train, val, test, shuffle=False):
        '''split dataset into train, val and test sets'''
        # get dataset
        if ds_type == "captions":
            dataset = self.get_captions()
        elif ds_type == "embeddings":
            dataset = self.get_caption_embeddings()
        else:
            raise ValueError("ds_type should either be 'captions' or 'embeddings'")
        if train+val+test != 1.0:
            raise ValueError("Specify train, val and test to add up to 1.0")
        if len(dataset) == 0:
            raise EmptyDataset(f"{ds_type} dataset is empty.")
        dataset_keys = list(dataset.keys())
        if shuffle:
            random.shuffle(dataset_keys)
        # calculate training and validation set ending index
        train_end_index = ceil(len(dataset_keys)*train)
        val_end_index = train_end_index + ceil(len(dataset_keys)*val)
        test_end_index = max(val_end_index+ceil(len(dataset_keys)*test), len(dataset_keys))
        # generate train, val and test sets
        train_set = {id: dataset[id] for id in tqdm(dataset_keys[:train_end_index])}
        val_set = {id: dataset[id] for id in tqdm(dataset_keys[train_end_index:val_end_index])}
        test_set = {id: dataset[id] for id in tqdm(dataset_keys[val_end_index:test_end_index])}
        # return train, val and test sets
        return train_set, val_set, test_set
    def write_split(self, ds_type, train, val, test, checkpoint_dir, shuffle=False):
        '''write the train, val and test sets to checkpoint_dir directory'''
        train_set, val_set, test_set = self.get_split(ds_type, train, val, test, shuffle)
        splits = {"train": train_set, "val": val_set, "test": test_set}
        print(f"Writing splits to directory: {checkpoint_dir}")
        file_path = checkpoint_dir+"{}_{}.pkl"
        for split_type, split_ds in splits.items():
            with open(file_path.format(split_type, ds_type), 'wb') as file:
                pickle.dump(split_ds, file)
    def write_captions(self, checkpoint):
        '''write captions (list of words) to checkpoint path'''
        if checkpoint.split(".")[-1] not in ["pkl", "pickle"]:
            raise IncorrectFileFormat("checkpoint should end in .pkl or .pickle")
        all_captions = self.get_captions()
        if len(all_captions) == 0:
            raise EmptyDataset("Captions dataset is empty.")
        print(f"Writing captions dataset to checkpoint: {checkpoint}")
        with open(checkpoint, "wb") as file:
            pickle.dump(all_captions, file)
    def write_caption_embeddings(self, checkpoint):
        '''write caption embeddings to checkpoint path'''
        if checkpoint.split(".")[-1] not in ["pkl", "pickle"]:
            raise IncorrectFileFormat("checkpoint should end in .pkl or .pickle")
        all_caption_embeddings = self.get_caption_embeddings()
        if len(all_caption_embeddings) == 0:
            raise EmptyDataset("Caption Embeddings Dataset is empty.")
        print(f"Writing Caption Embeddings Dataset to checkpoint: {checkpoint}")
        with open(checkpoint, "wb") as file:
            pickle.dump(all_caption_embeddings, file)
    def write_word2vec_dataset(self, checkpoint):
        '''write Word2Vec dataset to checkpoint path'''
        if checkpoint.split(".")[-1] not in ["pkl", "pickle"]:
            raise IncorrectFileFormat("checkpoint should end in .pkl or .pickle")
        word2vec_dataset = self.get_word2vec_dataset()
        if len(word2vec_dataset) == 0:
            raise EmptyDataset("Word2Vec dataset is empty.")
        print(f"Writing Word2Vec Dataset to checkpoint: {checkpoint}")
        with open(checkpoint, "wb") as file:
            pickle.dump(word2vec_dataset, file)
    def __set_captions(self, captions):
        '''set self.captions_dataset to captions'''
        self.captions_dataset = captions
