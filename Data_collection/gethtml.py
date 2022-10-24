import numpy as np
import os
import yaml
import pandas as pd
import argparse

import open3d as o3d
import numpy as np
import imageio
import yaml
import cv2
import math
import sys, os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")

    args = parser.parse_args()
    return args

args = parse_args()
sequence_folder = './data/20220909/' + args.folder + '/'


hololens2_color2world = pd.read_html(sequence_folder + 'hololens2_color2world.html')[0]
hololens2_tag2color = pd.read_html(sequence_folder + 'hololens2_tag2color.html')[0]
hololens2_depth2world = pd.read_html(sequence_folder + 'hololens2_depth2world.html')[0]
hololens2_color2world.drop('Unnamed: 0', axis =1, inplace = True)
hololens2_tag2color.drop('Unnamed: 0', axis =1, inplace = True)
hololens2_depth2world.drop('Unnamed: 0', axis =1, inplace = True)

sync_path = sequence_folder + 'synced_timestamps.html'

df_sync = pd.read_html(sync_path)[0]
converters = {c:lambda x: str(x) for c in df_sync.columns}
df_sync = pd.read_html(sync_path,converters=converters)[0]


time_s_holo_depth = df_sync['000001'][17]

depth_test = cv2.imread('./data/20220909/20220909_114327/hololens2/depth_000000.png', cv2.IMREAD_ANYDEPTH).astype('float32')
depth_h,depth_w = depth_test.shape

# projtest = np.fromfile('./2022-04-17/depth_lut.bin', dtype = np.float32)
# print(projtest[:10])

#proj = np.fromfile('./2022-04-17/depth_lut.bin', dtype = np.float32).reshape((depth_w,depth_h,2)).transpose((1,0,2))

# if time_s_holo_depth in hololens2_depth2world.columns:
# 	print('true')


with open('./2022-04-17/depth_lut.bin', mode="rb") as lut_file:
	lut = np.frombuffer(lut_file.read(), dtype="f")
	print('lut1:',lut.shape)
	lut = np.reshape(lut, (-1, 3))
	print('lut2:',lut.shape)
	print('lut2_0:',lut[90000])



depth_scale=1000.0
img = cv2.imread('./data/20220909/20220909_114327/hololens2/depth_000000.png', cv2.IMREAD_ANYDEPTH).astype(np.float64)
print(img.shape)
img /= depth_scale
img = np.tile(img.flatten().reshape((-1, 1)), (1, 3))

print('img.shape0',img.shape[0])
print(lut.shape)

points = img * lut

print(points.shape)

print('points_300:',points[90000])
print('lut_300:',lut[90000])
print('img_300:',img[90000])


print('lut_:',lut[80000:80030])



# count =0
# for column_name in hololens2_tag2color:
# 	# print(column_name)
# 	# #print(hololens2_tag2color[column_name][0])

# 	if column_name in hololens2_color2world.columns:
# 		print(column_name)
# 		tag = eval(hololens2_color2world[str(column_name)][0])
# 		print(tag)


# 	count = count+1
# 	if(count == 10):
# 		break


