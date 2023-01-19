import logging

import numpy as np
import torch
from torch.utils.data import Dataset


class LGDataset(Dataset):
  
  def __init__(self, data_path, feature_col_index=None):
    self.feature_col_index = feature_col_index
    self.logger = logging.getLogger()
    self.raw_data = np.load(data_path)
    self.data = self._parse_data(self.raw_data)
    self.seq_len = self.data.shape[1]
    self.data_shape = self.data.size()
    
  def _parse_data(self, raw_data):
    data = raw_data
    if self.feature_col_index is not None:
      data = data[:, :, self.feature_col_index]
    data = torch.from_numpy(data)
    return data.float()
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    return self.data[idx]
  
  class Column:
    voltage = 0
    current = 1
    temperature = 2
    soc = 3