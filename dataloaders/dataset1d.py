import json

import wfdb
from scipy.signal import find_peaks
from sklearn.preprocessing import scale
from torch.utils.data import Dataset, DataLoader
import numpy as np


class EcgDataset1D(Dataset):
    def __init__(self, ann_path, mapping_path):
        super().__init__()
        self.data = json.load(open(ann_path))
        self.mapper = json.load(open(mapping_path))

    def __getitem__(self, index):
        img = np.load(self.data[index]['path']).astype('float32')
        img = img.reshape(1, img.shape[0])

        return {
            "image": img,
            "class": self.mapper[self.data[index]['label']]
        }

    def get_dataloader(self, num_workers=4, batch_size=16, shuffle=True):
        data_loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return data_loader

    def __len__(self):
        return len(self.data)


def callback_get_label(dataset, idx):
    return dataset[idx]["class"]


class EcgPipelineDataset1D(Dataset):
    def __init__(self, path, mode=128):
        super().__init__()
        record = wfdb.rdrecord(path)
        self.signal = None
        self.mode = mode
        for sig_name, signal in zip(record.sig_name, record.p_signal.T):
            if sig_name in ['MLII', 'II'] and np.all(np.isfinite(signal)):
                self.signal = scale(signal).astype('float32')
        if self.signal is None:
            raise Exception("No MLII LEAD")

        self.peaks = find_peaks(self.signal, distance=180)[0]
        mask_left = (self.peaks - self.mode // 2) > 0
        mask_right = (self.peaks + self.mode // 2) < len(self.signal)
        mask = mask_left & mask_right
        self.peaks = self.peaks[mask]

    def __getitem__(self, index):
        peak = self.peaks[index]
        left, right = peak - self.mode // 2, peak + self.mode // 2

        img = self.signal[left:right]
        img = img.reshape(1, img.shape[0])

        return {
            "image": img,
            "peak": peak
        }

    def get_dataloader(self, num_workers=4, batch_size=16, shuffle=True):
        data_loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return data_loader

    def __len__(self):
        return len(self.peaks)
