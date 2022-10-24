


import cv2
import yaml

# # file = open('parameters/apriltag_640x480.yml','r')


# def points_cloud(image_path, intrinsic_path,size):
# 	raw_image = cv2.imread(image_path)
# 	file = open(intrinsic_path,'r')
# 	all_tag = yaml.safe_load(file)
# 	fx = float(all_tag['color']['fx'])
# 	fy = float(all_tag['color']['fy'])
# 	ux = float(all_tag['color']['ppx'])
# 	uy = float(all_tag['color']['ppy'])
# 	p_cloud = np.zeros((size*size,3))




import pandas as pd
import numpy as np
from PIL import Image
import imageio
# import OpenEXR
import struct
import os


from matplotlib import pyplot as plt



def get_pointcloud(color_image,depth_image,camera_intrinsics):
    """ creates 3D point cloud of rgb images by taking depth information

        input : color image: numpy array[h,w,c], dtype= uint8
                depth image: numpy array[h,w] values of all channels will be same

        output : camera_points, color_points - both of shape(no. of pixels, 3)
    """

    image_height = depth_image.shape[0]
    image_width = depth_image.shape[1]
    pixel_x,pixel_y = np.meshgrid(np.linspace(0,image_width-1,image_width),
                                  np.linspace(0,image_height-1,image_height))
    camera_points_x = np.multiply(pixel_x-camera_intrinsics[0,2],depth_image/camera_intrinsics[0,0])
    camera_points_y = np.multiply(pixel_y-camera_intrinsics[1,2],depth_image/camera_intrinsics[1,1])
    camera_points_z = depth_image
    camera_points = np.array([camera_points_x,camera_points_y,camera_points_z]).transpose(1,2,0).reshape(-1,3)

    color_points = color_image.reshape(-1,3)

    # Remove invalid 3D points (where depth == 0)
    valid_depth_ind = np.where(depth_image.flatten() > 0)[0]
    camera_points = camera_points[valid_depth_ind,:]
    color_points = color_points[valid_depth_ind,:]

    return camera_points,color_points

def write_pointcloud(filename,xyz_points,rgb_points=None):

    """ creates a .pkl file of the point clouds generated

    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))
    fid.close()

np.set_printoptions(suppress=True)
def hand_joints(joint):
	rang = 0.003
	camera_points = np.zeros((3*3*3,3))
	center_z = joint[2]
	n = 0
	for x in range(-1,2):
		for y in range(-1,2):
			for z in range(-1,2):
				t = joint.copy() + np.array([rang*x,rang*y,rang*z])
				camera_points[n] = t
				n = n+1
				# color_points = np.row_stack((color_points,[0,255,0]))

	return camera_points, center_z


def world2pixel(x,intrinsic):
	t = np.zeros(x.shape)
	fx = intrinsic[0]
	fy = intrinsic[1]
	ux = intrinsic[2]
	uy = intrinsic[3]
	t[:, 0] = x[:, 0] * fx / x[:, 2] + ux
	t[:, 1] = x[:, 1] * fy / x[:, 2] + uy
	return t[:,0:2]

def hand_joints_pixel(joints,intrinsic,center_z,color_image,depth_image):
	pixel = np.round(world2pixel(joints,intrinsic))
	pixel = np.transpose(pixel)
	x_min = int(np.min(pixel[0]))
	x_max = int(np.max(pixel[0]))
	y_min = int(np.min(pixel[1]))
	y_max = int(np.max(pixel[1]))

	print('x_min',x_min)
	print('x_max',x_max)

	for i in range(x_min,x_max):
		for j in range(y_min,y_max):
			color_image[j][i] = [0,255,0]
			depth_image[j][i] = center_z*1000

	return color_image, depth_image

def hand_joints_pixel_red(joints,intrinsic,center_z,color_image,depth_image):
	pixel = np.round(world2pixel(joints,intrinsic))
	pixel = np.transpose(pixel)
	x_min = int(np.min(pixel[0]))
	x_max = int(np.max(pixel[0]))
	y_min = int(np.min(pixel[1]))
	y_max = int(np.max(pixel[1]))

	print('x_min',x_min)
	print('x_max',x_max)

	for i in range(x_min,x_max):
		for j in range(y_min,y_max):
			color_image[j][i] = [255,0,0]
			depth_image[j][i] = center_z*1000

	return color_image, depth_image


    # camera_points = np.row_stack((camera_points,[1,2,3]))
    # color_points = np.row_stack((color_points,[255,255,0]))
    # camera_points = np.row_stack((camera_points,[1,2,3]))
    # color_points = np.row_stack((color_points,[255,255,0]))
    # camera_points = np.row_stack((camera_points,[1,2,3]))
    # color_points = np.row_stack((color_points,[255,255,0]))
    # print(camera_points.shape)
    # print(camera_points[-5:])
    # print(color_points.shape)
    # print(color_points[-5:])


    # for x in range(-1,2):
    # 	for y in range(-1,2):
    # 		for z in range(-1,2):
    			
def creat_hand_joints_npy(time_s):



    body_tracking_path = 'data/20220424_151205/kinect_000684312712_body_tracking_full.html'
    df = pd.read_html(body_tracking_path)
    df_table = df[0]

    print('df',df_table[str(time_s)])

    timestamp_matrix = df_table[str(time_s)]
    df = np.array(df).reshape((32,128))

    hand_0 = np.zeros(3)
    hand_1 = np.zeros(3)
    hand_2 = np.zeros(3)
    hand_3 = np.zeros(3)


    hand_0_matrix = eval(timestamp_matrix[14])
    hand_1_matrix = eval(timestamp_matrix[15])
    hand_2_matrix = eval(timestamp_matrix[16])
    hand_3_matrix = eval(timestamp_matrix[17])
    # print('hand_0',hand_0_matrix)
    for i in range(3):
        hand_0[i] = hand_0_matrix[i][3]
        hand_1[i] = hand_1_matrix[i][3]
        hand_2[i] = hand_2_matrix[i][3]
        hand_3[i] = hand_3_matrix[i][3]

    hand_center = (hand_0+hand_1+hand_2+hand_3)/4
    print('hand_1',hand_1)
    return hand_1, hand_center


def read_info_by_sync(df, num_index):
    if(num_index<10):
        num_index_str = '00000'+str(num_index)

    elif(num_index>=100):
        num_index_str = '000'+str(num_index)
    else:
        num_index_str = '0000'+str(num_index)

    time_s_body = df[num_index_str][18]  #--------------the /kinect_000684312712/body_tracking_data in time_stamps_synced.html
    hand_1, hand_center = creat_hand_joints_npy(time_s_body)

    time_s_holo_depth = df[num_index_str][22]
    time_s_holo_color = df[num_index_str][21]





############################################################
#  Main
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='create point cloud from depth and rgb image.')
    # parser.add_argument('--rgb_filename', required=True,
    #                     help='path to the rgb image')
    # parser.add_argument('--depth_filename', required=True,
    #                     help="path to the depth image ")
    parser.add_argument('--output_directory', required=True,
                        help="directory to save the point cloud file")
    # parser.add_argument('--fx', required=True, type=float,
    #                     help="focal length along x-axis (longer side) in pixels")
    # parser.add_argument('--fy', required=True, type=float,
    #                     help="focal length along y-axis (shorter side) in pixels")
    # parser.add_argument('--cx', required=True, type=float,
    #                     help="centre of image along x-axis")
    # parser.add_argument('--cy', required=True, type=float,
    #                     help="centre of image along y-axis")

    args = parser.parse_args()


    # camera_intrinsics  = [[fx 0 cx],
    #                       [0 fy cy],
    #                       [0 0 1]]
    intrinsic_path = 'parameters/intrinsics/kinect_000684312712_720P.yml'

    file = open(intrinsic_path,'r')
    all_tag = yaml.safe_load(file)
    fx = float(all_tag['color']['fx'])
    fy = float(all_tag['color']['fy'])
    ux = float(all_tag['color']['ppx'])
    uy = float(all_tag['color']['ppy'])
    camera_intrinsics  = np.asarray([[fx, 0, ux], [0, fy, uy], [0, 0, 1]])


    num_index = 117


    filename = 'pointCloud'+str(num_index)+'.ply'
    output_filename = os.path.join(args.output_directory, filename)

    if(num_index<10):
        rgb_filename = 'data/20220424_151205/kinect_000684312712/rgb_00000'+str(num_index)+'.jpg'
        depth_filename = 'data/20220424_151205/kinect_000684312712/depth_to_rgb_00000'+str(num_index)+'.png'
        num_index_str = '00000'+str(num_index)

    elif(num_index>=100):
        rgb_filename = 'data/20220424_151205/kinect_000684312712/rgb_000'+str(num_index)+'.jpg'
        depth_filename = 'data/20220424_151205/kinect_000684312712/depth_to_rgb_000'+str(num_index)+'.png'
        num_index_str = '000'+str(num_index)
    else:
        rgb_filename = 'data/20220424_151205/kinect_000684312712/rgb_0000'+str(num_index)+'.jpg'
        depth_filename = 'data/20220424_151205/kinect_000684312712/depth_to_rgb_0000'+str(num_index)+'.png'
        num_index_str = '0000'+str(num_index)

    print("Creating the point Cloud file at : ", output_filename )

    # Getting the data from html files

    sync_path = 'data/20220424_151205/time_stamps_synced.html'

    df = pd.read_html(sync_path)[0]

    converters = {c:lambda x: str(x) for c in df.columns}
    df = pd.read_html(sync_path,converters=converters)[0]



    df[num_index_str] = df[num_index_str].astype(str)
    print(df[num_index_str])
    # df = np.array(df).reshape((24,233))
    # df_body_tracking = df[-6][1:233].copy()

    # df_body_tracking = df[num_index_str][18].astype(np.int64)




    time_s = df[num_index_str][18]

    # time_s = 1650831130106530674 #85

    # time_s = 1650831130240132406 #87

    # time_s = 1650831130306748355 #88

    # time_s = 1650831130373436355 #90

    # time_s = 1650831130439965364 #92

    # time_s = 1650831130506525391 #93

    # time_s = 1650831130573169391 #95

    # time_s = 1650831130706504192 #97

    # time_s = 1650831130773209459 #99

    # time_s = 1650831130839912851 #101

    # time_s = 1650831130906583112 #103

    # time_s = 1650831130973238112 #105

    # time_s = 1650831131040207629 #107

    # time_s = 1650831131173640690 #109

    # time_s = 1650831131240085812 #110

    # time_s = 1650831131306814660 #112

    # time_s = 1650831131573370446 #116

    # time_s = 1650831131640147268 #117

    # time_s = 1650831131840217120 #119

    # time_s = 1650831131906859206  #121


    # time_s = 1650831132240317970 #127

    # time_s = 1650831132306947886 #129

    # time_s = 1650831132373740170 #130




    np.set_printoptions(suppress=True)


    print('time_s',time_s)



    time_set = np.array([1650831131640147268,1650831130706504192,1650831135775587535])
    
    num_hand_1 = time_set.shape[0]

    all_hand_1 = np.zeros((num_hand_1,3))
    all_hand_center = np.zeros((num_hand_1,3))
    
    for i in range(num_hand_1):
    	all_hand_1[i], all_hand_center[i] =creat_hand_joints_npy(time_set[i])

	    

    np.save('hand_1.npy',all_hand_1)
    np.save('hand_center.npy',all_hand_center)


    time_set_holo_depth = [1650831131878390312,1650831130964634895,1650831136032037973]
    holo_full = 'data/20220424_151205/holo_tfs_full.html'
    df = pd.read_html(holo_full)
    df_table = df[0]

    world2depth = np.zeros((num_hand_1,4,4))
    world2color = np.zeros((num_hand_1,4,4))
    
    # for i in range()
    world2depth[0] = np.linalg.inv(eval(df_table[str(time_set_holo_depth[0])][0]))
    world2depth[1] = np.linalg.inv(eval(df_table[str(time_set_holo_depth[1])][0]))

    time_set_holo_color = [1650831131901198148, 1650831130970584392,1650831136031211614]
    world2color[0] = np.linalg.inv(eval(df_table[str(time_set_holo_color[0])][2]))
    world2color[1] = np.linalg.inv(eval(df_table[str(time_set_holo_color[1])][2]))


    np.save('world2depth.npy',world2depth)
    np.save('world2color.npy',world2color)





    body_tracking_path = 'data/20220424_151205/kinect_000684312712_body_tracking_full.html'
    df = pd.read_html(body_tracking_path)
    df_table = df[0]

    print('df',df_table[str(time_s)])

    timestamp_matrix = df_table[str(time_s)]
    df = np.array(df).reshape((32,128))

    hand_0 = np.zeros(3)
    hand_1 = np.zeros(3)
    hand_2 = np.zeros(3)
    hand_3 = np.zeros(3)


    hand_0_matrix = eval(timestamp_matrix[14])
    hand_1_matrix = eval(timestamp_matrix[15])
    hand_2_matrix = eval(timestamp_matrix[16])
    hand_3_matrix = eval(timestamp_matrix[17])
    # print('hand_0',hand_0_matrix)
    for i in range(3):
        hand_0[i] = hand_0_matrix[i][3]
        hand_1[i] = hand_1_matrix[i][3]
        hand_2[i] = hand_2_matrix[i][3]
        hand_3[i] = hand_3_matrix[i][3]
    



    # hand_0 = np.array([0.055407654494047165,0.00398477166891098,1.0787612199783325])
    # hand_1 = np.array([0.03450549393892288,0.09149225056171417,1.0338621139526367])
    # hand_2 = np.array([0.009893583133816719,0.18675851821899414,1.0033929347991943])
    # hand_3 = np.array([0.07642655819654465,0.09864285588264465,1.0070242881774902])
    hand_center = (hand_0+hand_1+hand_2+hand_3)/4
    print('hand_1',hand_1)


    translation = np.array([0.03209632350165928,0.002455738099463839,-0.003794187865719409])
    rotation_Q = np.array([0.05120293357335243,0.00032522608234919075,0.0022803383345949882,0.9986856137362984])

    x = 0.05120293357335243
    y = 0.00032522608234919075
    z = 0.0022803383345949882
    w = 0.9986856137362984

    rotation_matrix = np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
		[2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
		[2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])

    print('rotation',rotation_matrix)
    print('translation',translation)

    abc0 = np.append(rotation_matrix[0],[translation[0]],axis=0)
    abc1 = np.append(rotation_matrix[1],[translation[1]],axis=0)
    abc2 = np.append(rotation_matrix[2],[translation[2]],axis=0)
    abc3 = np.array([0,0,0,1])

    abc = np.zeros((4,4))
    abc[0] = abc0
    abc[1] = abc1
    abc[2] = abc2
    abc[3] = abc3
    print(abc)

    print('inv',np.linalg.inv(abc))


    rootation_matrix_inv = np.linalg.inv(rotation_matrix)

    hand_0_rgb = np.zeros(3)
    hand_1_rgb = np.zeros(3)
    hand_2_rgb = np.zeros(3)
    hand_3_rgb = np.zeros(3)
    hand_center_rgb = np.zeros(3)

    hand_0_rgb = hand_0 - translation
    hand_0_rgb = np.dot(rootation_matrix_inv,hand_0_rgb)
    hand_1_rgb = hand_1 - translation
    hand_1_rgb = np.dot(rootation_matrix_inv,hand_1_rgb)
    hand_2_rgb = hand_2 - translation
    hand_2_rgb = np.dot(rootation_matrix_inv,hand_2_rgb)
    hand_3_rgb = hand_3 - translation
    hand_3_rgb = np.dot(rootation_matrix_inv,hand_3_rgb)
    hand_center_rgb = hand_center - translation
    hand_center_rgb = np.dot(rootation_matrix_inv,hand_center_rgb)

    if(num_index<10):
        rgb_filename = 'data/20220424_151205/kinect_000684312712/rgb_00000'+str(num_index)+'.jpg'
        depth_filename = 'data/20220424_151205/kinect_000684312712/depth_to_rgb_00000'+str(num_index)+'.png'

    elif(num_index>=100):
        rgb_filename = 'data/20220424_151205/kinect_000684312712/rgb_000'+str(num_index)+'.jpg'
        depth_filename = 'data/20220424_151205/kinect_000684312712/depth_to_rgb_000'+str(num_index)+'.png'
    else:
        rgb_filename = 'data/20220424_151205/kinect_000684312712/rgb_0000'+str(num_index)+'.jpg'
        depth_filename = 'data/20220424_151205/kinect_000684312712/depth_to_rgb_0000'+str(num_index)+'.png'






    im_color = imageio.imread(rgb_filename)
    im_depth = imageio.imread(depth_filename)

    joint_0, joint_0_z = hand_joints(hand_0_rgb)
    joint_1, joint_1_z = hand_joints(hand_1_rgb)
    joint_2, joint_2_z = hand_joints(hand_2_rgb)
    joint_3, joint_3_z = hand_joints(hand_3_rgb)
    joint_center, joint_center_z = hand_joints(hand_center_rgb)



    color_image, depth_image = hand_joints_pixel(joint_0,np.array([fx,fy,ux,uy]),joint_0_z,im_color,im_depth)
    color_image, depth_image = hand_joints_pixel(joint_1,np.array([fx,fy,ux,uy]),joint_1_z,color_image,depth_image)
    color_image, depth_image = hand_joints_pixel(joint_2,np.array([fx,fy,ux,uy]),joint_2_z,color_image,depth_image)
    color_image, depth_image = hand_joints_pixel(joint_3,np.array([fx,fy,ux,uy]),joint_3_z,color_image,depth_image)
    color_image, depth_image = hand_joints_pixel_red(joint_center,np.array([fx,fy,ux,uy]),joint_center_z,color_image,depth_image)


    plt.imshow(color_image)
    plt.show()
    print('joints',joint_center)

    color_data = color_image[450-60:450+60,570-60:570+60].copy()

    depth_data = depth_image[450-60:450+60,570-60:570+60].copy()

    camera_points, color_points = get_pointcloud(color_data, depth_data, camera_intrinsics)



    # camera_points_h, color_points_h = hand_joints(hand_0_rgb,camera_points, color_points)
    # camera_points_h, color_points_h = hand_joints(hand_1_rgb,camera_points_h, color_points_h)
    # camera_points_h, color_points_h = hand_joints(hand_2_rgb,camera_points_h, color_points_h)
    # camera_points_h, color_points_h = hand_joints(hand_3_rgb,camera_points_h, color_points_h)
    # camera_points_h, color_points_h = hand_joints(hand_center_rgb,camera_points_h, color_points_h)

    # print(camera_points)

    # print(camera_points[0].dtype)

    # print(camera_points_h)

    # print(camera_points_h[0].dtype)


    # print(color_points)

    # print(color_points_h)





    write_pointcloud(output_filename, camera_points, color_points)
