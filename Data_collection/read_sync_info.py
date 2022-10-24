
import pandas as pd

import numpy as np
import imageio
import yaml
import argparse

from utils import Camera

def creat_hand_joints_npy(time_s, df_table):


    timestamp_matrix = df_table[str(time_s)]


    hand_0 = np.zeros(3)
    hand_1 = np.zeros(3)
    hand_2 = np.zeros(3)
    hand_3 = np.zeros(3)


    # hand_0_matrix = eval(timestamp_matrix[14].replace('\\n ', ',').replace(' ', ',').replace('[,', '[').replace(',]', ']'))
    # hand_1_matrix = eval(timestamp_matrix[15].replace('\\n ', ',').replace(' ', ',').replace('[,', '[').replace(',]', ']'))
    # hand_2_matrix = eval(timestamp_matrix[16].replace('\\n ', ',').replace(' ', ',').replace('[,', '[').replace(',]', ']'))
    # hand_3_matrix = eval(timestamp_matrix[17].replace('\\n ', ',').replace(' ', ',').replace('[,', '[').replace(',]', ']'))

    hand_0_matrix = eval(timestamp_matrix[14])
    hand_1_matrix = eval(timestamp_matrix[15])
    hand_2_matrix = eval(timestamp_matrix[16])
    hand_3_matrix = eval(timestamp_matrix[17])

    for i in range(3):
        hand_0[i] = hand_0_matrix[i][3]
        hand_1[i] = hand_1_matrix[i][3]
        hand_2[i] = hand_2_matrix[i][3]
        hand_3[i] = hand_3_matrix[i][3]

    hand_center = (hand_0+hand_1+hand_2+hand_3)/4

    return hand_1, hand_center



def read_info_by_sync(df_sync, num_index):
    if(num_index<10):
        num_index_str = '00000'+str(num_index)

    elif (num_index >= 1000):
        num_index_str = '00' + str(num_index)

    elif(num_index>=100 and num_index<1000):
        num_index_str = '000'+str(num_index)
    else:
        num_index_str = '0000'+str(num_index)

    hand_0 = np.zeros(3)
    hand_1 = np.zeros(3)
    hand_2 = np.zeros(3)
    hand_3 = np.zeros(3)


    hand_0_matrix = eval(df_sync[num_index_str][21])[14]
    hand_1_matrix = eval(df_sync[num_index_str][21])[15]
    hand_2_matrix = eval(df_sync[num_index_str][21])[16]
    hand_3_matrix = eval(df_sync[num_index_str][21])[17]

    for i in range(3):
        hand_0[i] = hand_0_matrix[i][3]
        hand_1[i] = hand_1_matrix[i][3]
        hand_2[i] = hand_2_matrix[i][3]
        hand_3[i] = hand_3_matrix[i][3]

    hand_center = (hand_0+hand_1+hand_2+hand_3)/4


    world2depth = np.linalg.inv(eval(df_sync[num_index_str][16]))
    world2color = np.linalg.inv(eval(df_sync[num_index_str][17]))

    camera_info = eval(df_sync[num_index_str][18])

    return hand_1, hand_center, world2depth, world2color, camera_info



def find_all_index(df_sync):


    size_c = df_sync.shape[1] - 1
    index_all = []

    for num_index in range(size_c):
        if (num_index < 10):
            num_index_str = '00000' + str(num_index)

        elif (num_index >= 1000):
            num_index_str = '00' + str(num_index)

        elif (num_index >= 100 and num_index < 1000):
            num_index_str = '000' + str(num_index)
        else:
            num_index_str = '0000' + str(num_index)
        flag = 1
        for i in range(16,24):
            if (df_sync[num_index_str][i] != df_sync[num_index_str][i]):
                flag = 0

        if(flag == 1):
            index_all.append(num_index)

    return np.array(index_all)


def pointcloud(SN, index, index_center, num):
    num_index = index[num]
    if (num_index < 10):
        num_index_str = '00000' + str(num_index)

    elif (num_index >= 1000):
        num_index_str = '00' + str(num_index)

    elif (num_index >= 100 and num_index < 1000):
        num_index_str = '000' + str(num_index)
    else:
        num_index_str = '0000' + str(num_index)


    im_depth = imageio.imread(sequence_folder + 'kinect_000684312712/depth_' + num_index_str + '.png')   #*********************

    center = index_center[num]
    pixel_x = int(center[0])
    pixel_y = int(center[1])
    depth_value = im_depth[pixel_y][pixel_x]
    if depth_value == 0:
        for n in [[1,1],[1,-1],[-1,-1],[-1,1]]:
            depth_value = im_depth[pixel_y+n[0]][pixel_x+n[1]]
            if depth_value != 0:
                break







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









def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")

    args = parser.parse_args()
    return args

args = parse_args()
sequence_folder = './data/20220922/' + args.folder + '/'




if __name__ == '__main__':

    sync_path = sequence_folder + 'synced_info.html'

    df_sync = pd.read_html(sync_path)[0]
    converters = {c:lambda x: str(x) for c in df_sync.columns}
    df_sync = pd.read_html(sync_path,converters=converters)[0]


    index = find_all_index(df_sync)

    print('index',index)



    num_index = index.shape[0]
    all_hand_1 = np.zeros((num_index,3))
    all_hand_center = np.zeros((num_index,3))
    all_world2depth = np.zeros((num_index,4,4))
    all_world2color = np.zeros((num_index,4,4))
    all_camera_info = np.zeros((num_index, 3, 3))

    for i in range(num_index):
    	all_hand_1[i], all_hand_center[i], all_world2depth[i], all_world2color[i], all_camera_info[i] = \
            read_info_by_sync(df_sync,index[i])


    np.save(sequence_folder + 'hand_1.npy', all_hand_1)
    np.save(sequence_folder + 'hand_center.npy', all_hand_center)
    np.save(sequence_folder + 'world2depth.npy', all_world2depth)
    np.save(sequence_folder + 'world2color.npy', all_world2color)
    np.save(sequence_folder + 'index.npy', index)
    np.save(sequence_folder + 'camera_info.npy', all_camera_info)




    center_p = all_hand_1             #*************************

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

    np.save(sequence_folder + 'old_hand_center.npy', center_p_rgb)

    bounding_box_w = np.zeros((8,3))


    my_camera_kinect = Camera(center_p_rgb,0.12, '000684312712')
    pixel_coordinate = my_camera_kinect.world2pixel_target(center_p_rgb)

    int_pixel_coordinate = np.round(pixel_coordinate)
    bounding_box_p = my_camera_kinect.bounding_box_to_pixel(center_p_rgb,1)

    new_center_p_rgb = np.zeros((index.shape[0],3))
    for i in range(new_center_p_rgb.shape[0]):
        new_center_p_rgb[i] = pointcloud('000684312712', index, int_pixel_coordinate, i)
        #print(index[i], i, new_center_p_rgb[i])

    np.save(sequence_folder + 'new_hand_center.npy', new_center_p_rgb)









