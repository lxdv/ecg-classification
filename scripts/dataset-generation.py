import argparse
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from sklearn.preprocessing import scale
from wfdb import rdrecord

# Choose from peak to peak or centered
mode = [20, 20]
# mode = 128

image_size = 128
output_dir = '../data'

# dpi fix
fig = plt.figure(frameon=False)
dpi = fig.dpi

# fig size / image size
figsize = (image_size / dpi, image_size / dpi)
image_size = (image_size, image_size)


def plot(signal, filename):
    plt.figure(figsize=figsize, frameon=False)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0) # use for generation images with no margin
    plt.plot(signal)
    plt.savefig(filename)

    plt.close()

    im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    im_gray = cv2.resize(im_gray, image_size, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(filename, im_gray)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    args = parser.parse_args()


    ecg = args.file
    name = osp.basename(ecg)
    record = rdrecord(ecg)
    ann = wfdb.rdann(ecg, extension='atr')
    for sig_name, signal in zip(record.sig_name, record.p_signal.T):
        if not np.all(np.isfinite(signal)):
            continue
        signal = scale(signal)
        for i, (label, peak) in enumerate(zip(ann.symbol, ann.sample)):
            if label == '/': label = "\\"
            print('\r{} [{}/{}]'.format(sig_name, i + 1, len(ann.symbol)), end="")
            if isinstance(mode, list):
                if np.all([i > 0, i + 1 < len(ann.sample)]):
                    left = ann.sample[i - 1] + mode[0]
                    right = ann.sample[i + 1] - mode[1]
                else:
                    continue
            elif isinstance(mode, int):
                left, right = peak - mode // 2, peak + mode // 2
            else:
                raise Exception("Wrong mode in script beginning")

            if np.all([left > 0, right < len(signal)]):
                one_dim_data_dir = osp.join(output_dir, '1D', name, sig_name, label)
                two_dim_data_dir = osp.join(output_dir, '2D', name, sig_name, label)
                os.makedirs(one_dim_data_dir, exist_ok=True)
                os.makedirs(two_dim_data_dir, exist_ok=True)

                filename = osp.join(one_dim_data_dir, '{}.npy'.format(peak))
                np.save(filename, signal[left:right])
                filename = osp.join(two_dim_data_dir, '{}.png'.format(peak))

                plot(signal[left:right], filename)


