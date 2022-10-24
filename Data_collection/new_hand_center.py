import numpy as np
import cv2

import sys, os


from utils import Camera
from PIL import Image
from matplotlib import pyplot as plt

# holo_depth_image = cv2.imread('./data/20220520_161851/hololens/test/' + 'depth_' + '000049' + '.png')
# abc = cv2.cvtColor(holo_depth_image, cv2.COLOR_BGR2RGB)
# plt.figure()
# plt.imshow(abc)
# plt.plot(100, 200, 'r+')
# plt.show()

# abc = '[[-0.02558695 -0.61194974 0.79048266 0.43700984]\\n [-0.57557159 0.65554323 0.48885625 0.08218343]\\n [-0.817351 -0.44247102 -0.36899423 1.54842496]\\n [ 0. 0. 0. 1. ]]'
# print(abc)
# d = abc.replace('\\n ', ',').replace(' ', ',').replace('[,', '[').replace(',]', ']')
# print(d)


# abc1 = '[[-3.18600937e-02 -2.40170064e-01 9.70207851e-01 1.46018547e-01]\n [-9.99487650e-01 4.68263268e-03 -3.16624340e-02 9.95169476e-04]\n [ 3.06124180e-03 -9.70719534e-01 -2.40196202e-01 7.09787909e-02]\n [ 0.00000000e+00 0.00000000e+00 0.00000000e+00 1.00000000e+00]]'
# d1 = abc1.replace('\\n ', ',').replace(' ', ',').replace('[,', '[').replace(',]', ']')
# print(eval(d1))
#
# a = np.array([[1,2,3],[4,5,6]])
# b = a*10
# print(b)





import numpy as np
import imageio
import yaml


from utils import Camera





sequence_folder = './data/20220529/20220529_144535/'
center_p = np.load(sequence_folder + 'hand_center.npy')
print('center_p',center_p)







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

bounding_box_w = np.zeros((8,3))

index = np.load(sequence_folder + 'index.npy')




print(center_p_rgb.shape)
print(index.shape)

print(index)





# center_p_rgb[25] = np.array([0.08325238, 0.15737609, 0.836])
# center_p_rgb[26] = np.array([0.05551678, 0.14841926, 0.852])
# center_p_rgb[27] = np.array([0.05718354, 0.14795259, 0.857])
# center_p_rgb[28] = np.array([0.04807569, 0.14747031, 0.862])
# center_p_rgb[29] = np.array([0.02552565, 0.14149973, 0.875])
# center_p_rgb[30] = np.array([0.02014082, 0.13940256, 0.879])
# center_p_rgb[31] = np.array([0.01188853, 0.13528652, 0.879])
# center_p_rgb[32] = np.array([0.00366107, 0.1348286,  0.885])
# center_p_rgb[33] = np.array([-0.00189074,  0.13435195,  0.891])
# center_p_rgb[34] = np.array([-0.00468431,  0.13171813,  0.892])
# center_p_rgb[35] = np.array([-0.01179367,  0.13319479,  0.902])

# print(index[25])
# center_p_rgb[25] = np.array([0.08325237572, 0.15737609498, 0.836])
# center_p_rgb[26] = np.array([0.05551677564, 0.14841925868, 0.852])
#
# for i in range(index.shape[0]):
# 	print(index[i],center_p_rgb[i])

# print('center_p_rgb',center_p_rgb)

# center_p_rgb[0] = np.array([-0.03592442,  0.1436214,   0.983])
# print(index[0],center_p_rgb[0])
# center_p_rgb[1] = np.array([-0.03298186,  0.14574641,  0.987])
# print(index[1],center_p_rgb[1])
# center_p_rgb[2] = np.array([-0.03750064,  0.1406957,   0.984])
# print(index[2],center_p_rgb[2])
# center_p_rgb[3] = np.array([-0.03435133,  0.14040973,  0.982])
# print(index[3],center_p_rgb[3])


# center_p_rgb[4] = np.array([-0.03445627,  0.14083868,  0.985])
# center_p_rgb[5] = np.array([-0.03452623,  0.14112465,  0.987])
# center_p_rgb[6] = np.array([-0.03799607,  0.14722307,  0.997])
# center_p_rgb[7] = np.array([-0.03795796,  0.1470754,  0.996])
# center_p_rgb[8] = np.array([-0.03291503,  0.14237615,  0.985])
# center_p_rgb[9] = np.array([-0.03947674,  0.14692774,  0.995])
# center_p_rgb[10] = np.array([-0.03610715,  0.14280978,  0.988])
# center_p_rgb[11] = np.array([-0.03603405,  0.14252069,  0.986])
# center_p_rgb[12] = np.array([-0.03603405,  0.14252069,  0.986])
# center_p_rgb[13] = np.array([-0.03452623,  0.14266523,  0.987])
# center_p_rgb[14] = np.array([-0.03278136,  0.1372043,   0.981])
# center_p_rgb[15] = np.array([-0.03278136,  0.1372043,   0.981])
# center_p_rgb[16] = np.array([-0.03281478,  0.13734416,  0.982])
# center_p_rgb[17] = np.array([-0.03784363,  0.1435325,   0.993])
# center_p_rgb[18] = np.array([-0.03795796,  0.14396613,  0.996])
# center_p_rgb[19] = np.array([-0.03791985,  0.14382159,  0.995])
# center_p_rgb[20] = np.array([-0.03704331,  0.14049707,  0.972])
# center_p_rgb[21] = np.array([-0.03670032,  0.13919617,  0.963])
# center_p_rgb[22] = np.array([-0.03753875,  0.14237615,  0.985])
# center_p_rgb[23] = np.array([-0.03761497,  0.14266523,  0.987])
# center_p_rgb[24] = np.array([-0.03753875,  0.14237615,  0.985])
#
# for i in range((9, 25)):
# 	center_p_rgb[i] = np.array([])


# center_p_rgb[-1] = np.array([-0.03750064,  0.1422316,   0.984])
# print(index[-1],center_p_rgb[-1])

my_camera_kinect = Camera(center_p_rgb,0.12, '000684312712')
pixel_coordinate = my_camera_kinect.world2pixel_target(center_p_rgb)
print('pixel_coordinate',pixel_coordinate)
print('pixel_coordinate_shape',pixel_coordinate.shape)
print('pixel_round',np.round(pixel_coordinate))

int_pixel_coordinate = np.round(pixel_coordinate)
bounding_box_p = my_camera_kinect.bounding_box_to_pixel(center_p_rgb,1)

def pointcloud(SN, index, index_center, num):
    n_image = index[num]
    if (n_image < 10):
        num_index_str = '00000' + str(n_image)

    elif (n_image >= 100):
        num_index_str = '000' + str(n_image)
    else:
        num_index_str = '0000' + str(n_image)


    im_depth = imageio.imread(sequence_folder + 'kinect_000684312712/depth_' + num_index_str + '.png')

    center = index_center[num]
    pixel_x = int(center[0])
    pixel_y = int(center[1])
    depth_value = im_depth[pixel_y][pixel_x]
    if depth_value == 0:
        for n in [[1,1],[1,-1],[-1,-1],[-1,1]]:
            depth_value = im_depth[pixel_y+n[0]][pixel_x+n[1]]
            if depth_value != 0:
                break



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

new_center_p_rgb = np.zeros((index.shape[0],3))
for i in range(new_center_p_rgb.shape[0]):
    new_center_p_rgb[i] = pointcloud('000684312712', index, int_pixel_coordinate, i)
    print(index[i], i, new_center_p_rgb[i])

np.save(sequence_folder + 'new_hand_center.npy', new_center_p_rgb)