import os
import random
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import FASDataset
from utils.utils import sample_frames

def get_dataset(tgt_data, tgt_test_num_frames, batch_size):

    print('Load Target Data')
    print('Target Data: ', tgt_data)
    tgt_train_data_valid = sample_frames(flag=3, num_frames=tgt_test_num_frames, dataset_name=tgt_data)
    tgt_test_data = sample_frames(flag=4, num_frames=tgt_test_num_frames, dataset_name=tgt_data)

    tgt_train_dataloader_valid = DataLoader(FASDataset(tgt_train_data_valid, train=False), batch_size=batch_size, shuffle=False)
    tgt_test_dataloader = DataLoader(FASDataset(tgt_test_data, train=False), batch_size=batch_size, shuffle=False)

    return tgt_train_dataloader_valid, tgt_test_dataloader








