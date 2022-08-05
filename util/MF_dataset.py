# Zhen FENG, Aug. 5, 2022
# Email: zfeng@outlook.com

import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL

class MF_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=512, input_w=512,transform=[]):
        super(MF_dataset, self).__init__()

        assert split in ['train', 'validation', 'test'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.n_data    = len(self.names)

    def read_image(self, name, folder,split):
        file_path = os.path.join(self.data_dir, '%s/%s/%s.png' % (split,folder, name))
        image     = np.asarray(PIL.Image.open(file_path))
        return image

    def __getitem__(self, index):
        name  = self.names[index]

        rgb = self.read_image(name, 'rgb',self.split)
        tdisp = self.read_image(name, 'tdisp',self.split)
        label = self.read_image(name, 'label',self.split)
        for func in self.transform:
            rgb,tdisp,label = func(rgb,tdisp, label)
        rgb = np.asarray(PIL.Image.fromarray(rgb).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255
        tdisp = np.asarray(PIL.Image.fromarray(tdisp).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST), dtype=np.int64)
        return torch.tensor(rgb), torch.tensor(tdisp),torch.tensor(label), name

    def __len__(self):
        return self.n_data
