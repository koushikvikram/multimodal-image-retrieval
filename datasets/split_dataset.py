'''Generate embeddings for captions dataset, split it into training, validation and test sets and save them to disk.'''

from src.dataset import Dataset
import config.dataset as ds_cfg


def split_and_write(processed_captions_filepath, checkpoint_dir):
    dataset = Dataset()
    dataset.read_captions_checkpoint(processed_captions_filepath)
    dataset.make_caption_embeddings()
    dataset.write_split(ds_type="embeddings", train=0.8, val=0.05, test=0.15, checkpoint_dir=checkpoint_dir)

if __name__ == "__main__":
    captions_filepath = ds_cfg.PROCESSED_CAPTIONS_PATH + "cleaned/" + "clean_captions_min_count_5.pkl"
    save_dir = ds_cfg.PROCESSED_CAPTIONS_PATH + "embeddings/"
    split_and_write(captions_filepath, save_dir)
