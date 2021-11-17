'''Make word2vec dataset and save it to disk. 
In the process, also save cleaned captions and high-frequency cleaned captions to disk.'''

# from src.dataset import Dataset
from ..config import dataset 

# dataset = Dataset(captions_path="")
print(dataset.CAPTIONS_PATH)