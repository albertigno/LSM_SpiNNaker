# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 18:14:11 2018

@author: alberto
"""

import torch.utils.data as data
import torch
import numpy as np
import h5py

class DatasetLoader(data.Dataset):
    def __init__(self, path=None, win=30, device='cpu', num_samples = None):

        self.device=device

        data = h5py.File(path, 'r')
        image, label = data[list(data.keys())[0]], data[list(data.keys())[1]]

        self.images = torch.from_numpy(np.array(image)).to(device)
        self.labels = torch.from_numpy(np.array(label)).float()

        self.images=self.images[:,:win,:]
        
        if num_samples is not None:
            self.images=self.images[:num_samples,:,:]
        
        self.num_sample = len(self.images)
        print('num sample: {}'.format(self.num_sample))
        
        print(self.images.size(),self.labels.size())

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return self.num_sample