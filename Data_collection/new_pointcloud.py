import numpy as np
import cv2

import sys, os
import imageio
import yaml
from utils import Camera
from PIL import Image
from matplotlib import pyplot as plt


# im_depth = imageio.imread('./data/20220520_161851/kinect_000684312712/depth_to_rgb_000042.png')
# print(np.max(im_depth))
# print(im_depth[19][566])
index = np.array([261, 263, 268, 270, 272, 273, 275, 277, 278, 279, 280])
index_center = np.array([[708-1, 495-1], [686-1, 486-1], [686, 484], [679, 483], [662, 477],
                         [658, 475], [652, 472], [646, 471], [642, 470], [640, 468], [635, 468]])

sequence_folder = './data/20220529/20220529_144535/'

# index = np.array([71, 73, 74, 76, 78, 79, 81, 83, 85, 87, 89, 91, 93, 95, 99, 101, 103, 105,
#                  107, 109, 112, 116, 120, 124, 128, 130])
# index_center = np.array([[620, 467],[622, 468],[619, 465],[621, 465],
#                          [621, 465], [621, 465], [619, 468], [619, 468],
#                          [622, 466],[618, 468], [620, 466], [620, 466], [620, 466],
#                          [621, 466],[622, 463], [622, 463], [622, 463], [619, 466], [619, 466],
#                          [619, 466], [619, 466], [619, 466], [619, 466], [619, 466] ,[619, 466],
#                          [619, 466]])


print(index.shape[0])
print(index_center.shape[0])

def pointcloud(SN, index, index_center, num):
    n_image = index[num]
    if (n_image < 10):
        num_index_str = '00000' + str(n_image)

    elif (n_image >= 100):
        num_index_str = '000' + str(n_image)
    else:
        num_index_str = '0000' + str(n_image)


    im_depth = imageio.imread(sequence_folder + 'kinect_000684312712/depth_to_rgb_' + num_index_str + '.png')

    center = index_center[num]
    pixel_x = center[0]
    pixel_y = center[1]
    depth_value = im_depth[pixel_y][pixel_x]
    print('depth_value',depth_value)



    file = open('2022-04-17/intrinsics/' + SN + '_640x480.yml').read()
    all_tag = yaml.safe_load(file)
    fx = float(all_tag['color']['fx'])
    fy = float(all_tag['color']['fy'])
    ux = float(all_tag['color']['ppx'])
    uy = float(all_tag['color']['ppy'])

    camera_points_x = (pixel_x - ux) * depth_value/fx
    camera_points_y = (pixel_y - uy) * depth_value/fy
    camera_points_z = depth_value
    xyz = np.zeros(3)
    xyz[0] = camera_points_x / 1000
    xyz[1] = camera_points_y / 1000
    xyz[2] = camera_points_z / 1000
    return xyz

kinect_coordinate = np.zeros((index.shape[0],3))

for i in range(kinect_coordinate.shape[0]):
    kinect_coordinate[i] = pointcloud('000684312712', index, index_center, i)
    print(index[i], i, kinect_coordinate[i])







import numpy as np
import cv2

import sys, os


from utils import Camera
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg






center_p = np.load(sequence_folder + 'hand_center.npy')

print('handcenter', center_p.shape[0])





translation = np.array([0.03209632350165928,0.002455738099463839,-0.003794187865719409])
rotation_Q = np.array([0.05120293357335243,0.00032522608234919075,0.0022803383345949882,0.9986856137362984])

x = 0.05120293357335243
y = 0.00032522608234919075
z = 0.0022803383345949882
w = 0.9986856137362984

rotation_matrix = np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
	[2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
	[2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])
rootation_matrix_inv = np.linalg.inv(rotation_matrix)




num_center = center_p.shape[0]
center_p_rgb = np.zeros((num_center,3))

for i in range(num_center):
	center_p_rgb[i] = center_p[i] - translation
	center_p_rgb[i] = (np.dot(rootation_matrix_inv,center_p_rgb[i])).reshape((1,3))

index_raw = np.load(sequence_folder + 'index.npy')
print('index_raw', index_raw.shape[0])
for i in range(index_raw.shape[0]):
    print(index_raw[i], center_p_rgb[i])

for i in range(index_raw.shape[0]):
    print(index_raw[i], center_p[i])



for i in range(4):
    print(center_p_rgb[i] - kinect_coordinate[i])

print(center_p_rgb[-1] - kinect_coordinate[-1])