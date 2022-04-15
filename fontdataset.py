import torch
from torch.utils.data import Dataset
import os
from pdb import set_trace as breakpoint
import pickle

class FontDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # load in data_dir/dataset.pkl
        self.data = pickle.load(open(os.path.join(data_dir, 'dataset.pkl'), 'rb'))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        font_index, char_index, path = self.data[idx]
        # read in path
        img = torch.load(path)
        return img, font_index, char_index

if __name__ == '__main__':
    # read in font_images_28
    dataset = FontDataset('font_images_28')
    print(len(dataset))
    print(dataset[0][0])