'''Generate embeddings, train-val-test split and save to disk.'''

from src.dataset import CaptionDataset
import config.dataset as ds_cfg


def split_and_write(processed_captions_filepath, checkpoint_dir):
    '''generate embeddings, split dataset and save to "checkpoint_dir"'''
    dataset = CaptionDataset()
    dataset.read_captions_checkpoint(processed_captions_filepath)
    dataset.make_caption_embeddings()
    dataset.write_split(
        ds_type="embeddings",
        train=0.8,
        val=0.05,
        test=0.15,
        checkpoint_dir=checkpoint_dir
        )


if __name__ == "__main__":
    FILE_NAME = "clean_captions_min_count_5.pkl"
    CAPTIONS_FILEPATH = ds_cfg.PROCESSED_CAPTIONS_PATH + \
        "cleaned/" + FILE_NAME
    SAVE_DIR = ds_cfg.PROCESSED_CAPTIONS_PATH + "embeddings/"
    split_and_write(CAPTIONS_FILEPATH, SAVE_DIR)
