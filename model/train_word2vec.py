'''Data Preparation and Training script for Word2Vec Model'''

from gensim.models import Word2Vec

from src.dataset import Dataset
import config.word2vec as wv_cfg
import config.dataset as ds_cfg


def make_dataset(captions_path, captions_checkpoint, word2vec_checkpoint):
    '''Make a word2vec dataset from txt caption files.
    Save clean high frequency captions and the word2vec dataset'''
    # make captions dataset
    dataset = Dataset(
        captions_path=captions_path,
        images_path=""
        )
    # read captions, clean it and drop words with count < 5
    dataset.read_captions(clean=True, min_count=5)
    # save captions dataset
    dataset.write_captions(captions_checkpoint)
    # make word2vec dataset
    dataset.make_word2vec_dataset()
    # save word2vec dataset
    dataset.write_word2vec_dataset(word2vec_checkpoint)
    return dataset

def make_word2vec_model(word2vec_dataset: Dataset, checkpoint, format='pickle'):
    '''train word2vec and save to disk'''
    # get word2vec dataset
    word2vec_ds = word2vec_dataset.get_word2vec_dataset()

    # train word2vec model
    model = Word2Vec(
        word2vec_ds,
        size=wv_cfg.SIZE,
        min_count=wv_cfg.MIN_COUNT,
        workers=wv_cfg.N_CORES,
        iter=wv_cfg.EPOCHS,
        window=wv_cfg.WINDOW)
    # save the trained model
    if format == "pickle":
        model.save(checkpoint)
    elif format == "word2vec":
        model.wv.save_word2vec_format(checkpoint)


if __name__ == "__main__":
    clean_captions_checkpoint = ds_cfg.PROCESSED_CAPTIONS_PATH + \
        "cleaned" + "/" + "clean_captions_min_count_5.pkl"
    word2vec_dataset_checkpoint = ds_cfg.PROCESSED_CAPTIONS_PATH + \
        "word2vec" + "/" + "word2vec_dataset.pkl"
    word2vec_model_checkpoint = "word2vec.model"
    w2v_dataset = make_dataset(
        ds_cfg.CAPTIONS_PATH,
        clean_captions_checkpoint,
        word2vec_dataset_checkpoint)
    make_word2vec_model(w2v_dataset, word2vec_model_checkpoint)
