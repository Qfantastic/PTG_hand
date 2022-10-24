import numpy as np
import cv2
import matplotlib.image as mpimg
import sys, os
import imageio
index = np.load('index.npy')
index_l = index.shape[0]

raw_image_path = './data/20220520_161851/hololens/'
save_path = './data/20220520_161851/test_hololens/'

raw = imageio.imread(raw_image_path + 'sensor_depth_' + '000201' + '.png')
print(np.max(raw))
print(raw[156][182])
print(raw.shape)
for n in range(index_l):
    if (index[n] < 10):
        num_index_str = '00000' + str(index[n])

    elif (index[n] >= 100):
        num_index_str = '000' + str(index[n])
    else:
        num_index_str = '0000' + str(index[n])
    raw_image = cv2.imread(raw_image_path + 'sensor_depth_' + num_index_str + '.png')

