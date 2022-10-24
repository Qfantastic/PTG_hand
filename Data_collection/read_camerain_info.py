
import open3d as o3d
import numpy as np
import imageio
import yaml
import cv2
import math
import sys, os

from sklearn.cluster import KMeans

from yellowbrick.cluster.elbow import kelbow_visualizer

from utils import Camera
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import argparse

import time


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--folder')
	parser.add_argument('--save_folder')
	args = parser.parse_args()
	return args


args = parse_args()
sequence_folder = './data/20220922/' + args.folder + '/'            #*********************
center_p_rgb = np.load(sequence_folder + 'old_hand_center.npy')
center_p_rgb_left = np.load(sequence_folder + 'old_hand_center_left.npy')
index = np.load(sequence_folder + 'index.npy')
camera_info = np.load(sequence_folder + 'camera_info.npy')





bounding_box_w = np.zeros((8,3))

def cameras_transfer(center_p_rgb):
	my_camera_kinect = Camera(center_p_rgb, 0.10, '000684312712')
	pixel_coordinate = my_camera_kinect.world2pixel_target(center_p_rgb)
	# print('pixel_coordinate',pixel_coordinate)
	bounding_box_p = my_camera_kinect.bounding_box_to_pixel(center_p_rgb, 1)

	my_camera_0 = Camera(center_p_rgb, 0.17, '000684312712')
	target_center_0 = my_camera_0.transfer_camera_inv()
	pixel_coordinate_0 = my_camera_0.world2pixel(target_center_0)
	bounding_box_p_0 = my_camera_0.bounding_box_to_pixel(target_center_0, 0)

	# # print('target_center_0',target_center_0)
	# my_camera_1 = Camera(target_center_0, 0.17, '105322251225')
	# target_center_1 = my_camera_1.transfer_camera_inv()
	# pixel_coordinate_1 = my_camera_1.world2pixel_target(target_center_1)
	# bounding_box_p_1 = my_camera_1.bounding_box_to_pixel(target_center_1, 1)
	#
	# # print('target_center_0',target_center_0)
	# my_camera_2 = Camera(target_center_0, 0.17, '046122250168')
	# target_center_2 = my_camera_2.transfer_camera_inv()
	# pixel_coordinate_2 = my_camera_2.world2pixel_target(target_center_2)
	# bounding_box_p_2 = my_camera_2.bounding_box_to_pixel(target_center_2, 1)

	my_camera_holo = Camera(target_center_0, 0.15, 'holoworld_' + args.folder)
	target_center_holo = my_camera_holo.transfer_camera_inv()

	return pixel_coordinate, bounding_box_p, target_center_holo


def crop_kinect_depth_kmeans(abc,image_index,bounding_box_p, bounding_box_p_left):

	top = int(bounding_box_p[0][1])
	down = int(bounding_box_p[2][1])

	left = int(bounding_box_p[0][0])
	right = int(bounding_box_p[1][0])

	crop_image = abc[top:down, left:right]
	#crop_image_left = abc[bounding_box_p_left[0][1]:bounding_box_p_left[2][1], bounding_box_p_left[0][0]:bounding_box_p_left[1][0]]

	print('crop_image*********', crop_image.shape)

	ori_crop = crop_image /10
	#ori_crop_left = crop_image_left * 50

	ori_save_folder = os.path.join(f"./result_{args.save_folder}_{args.folder}",'kinect_crop')

	if not os.path.isdir(ori_save_folder):
		os.makedirs(ori_save_folder)

	cv2.imwrite(os.path.join(ori_save_folder,f"ori_right_{image_index}.png"), ori_crop)
	#cv2.imwrite(os.path.join(ori_save_folder, f"ori_left_{image_index}.png"), ori_crop_left)

	crop_image_1D = crop_image.reshape(crop_image.shape[0]*crop_image.shape[1],1)

	crop_image_1D_index = np.argwhere(crop_image_1D > 0)


	crop_real_1D = np.zeros((crop_image_1D_index.shape[0],1))

	for i in range(crop_image_1D_index.shape[0]):

		num_index = int(crop_image_1D_index[i][0])
		crop_real_1D[i] = crop_image_1D[num_index]

	#kelbow_visualizer(KMeans(), crop_real_1D, k=(1, 10))
	crop_real_1D = crop_real_1D.reshape(-1,1)
	print('crop_real_1D.shape',crop_real_1D)
	kmeans = KMeans(n_clusters=4,random_state=0).fit(crop_real_1D)
	#kelbow_visualizer(KMeans(), crop_image_1D, k=(1, 10))

	kmeans_result = kmeans.labels_
	# print('kmeans_result.shape',kmeans_result.shape)
	#
	k = 4
	avg = np.zeros(k)
	min_avg = 100000
	min = 0
	for n_k in range(k):
		class_hand = np.argwhere(kmeans_result == n_k)
		sum = 0
		for i in class_hand:
			sum = int(crop_real_1D[int(i)]) + sum
		avg = sum / class_hand.shape[0]
		if(avg<min_avg):
			min_avg = avg
			min = n_k
	class_hand = np.argwhere(kmeans_result == min)



	print(class_hand)
	crop_hand_1D = np.zeros(crop_image_1D.shape)

	for i in class_hand:
		num_index = int(crop_image_1D_index[int(i)][0])
		crop_hand_1D[num_index] = 1000
	crop_hand = crop_hand_1D.reshape(crop_image.shape[0], crop_image.shape[1])

	crop_kmeans_save_folder = os.path.join(f"./result_{args.save_folder}_{args.folder}", 'kinect_crop_kmeans')


	if not os.path.isdir(crop_kmeans_save_folder):
		os.makedirs(crop_kmeans_save_folder)

	cv2.imwrite(os.path.join(crop_kmeans_save_folder,f"kmeans_right_{image_index}.png"),crop_hand)









def draw_bounding(n,n_image, path, save_path, center_p, bounding_box_p, center_p_left, bounding_box_p_left):
	if (n_image < 10):
		num_index_str = '00000' + str(n_image)

	elif (n_image >= 1000):
		num_index_str = '00' + str(n_image)

	elif (n_image >= 100 and n_image < 1000):
		num_index_str = '000' + str(n_image)
	else:
		num_index_str = '0000' + str(n_image)

	abc_kinect = cv2.imread(path + 'color_' + num_index_str + '.jpg')
	abc_kinect = cv2.cvtColor(abc_kinect, cv2.COLOR_BGR2RGB)

	plt.figure()
	plt.imshow(abc_kinect)
	plt.plot(center_p[n][0], center_p[n][1], 'o', color = 'red')

	plt.plot(center_p_left[n][0], center_p_left[n][1], 'o', color = 'red')

	for i in range(3):
		plt.plot([bounding_box_p[n][i][0], bounding_box_p[n][i + 1][0]],
				 [bounding_box_p[n][i][1], bounding_box_p[n][i + 1][1]], color='green')
		plt.plot([bounding_box_p_left[n][i][0], bounding_box_p_left[n][i + 1][0]],
				 [bounding_box_p_left[n][i][1], bounding_box_p_left[n][i + 1][1]], color='green')


	plt.plot([bounding_box_p[n][0][0], bounding_box_p[n][3][0]],
			 [bounding_box_p[n][0][1], bounding_box_p[n][3][1]], color='green')
	plt.plot([bounding_box_p_left[n][0][0], bounding_box_p_left[n][3][0]],
			 [bounding_box_p_left[n][0][1], bounding_box_p_left[n][3][1]], color='green')
	plt.savefig(os.path.join(save_path, 'result') + str(n_image) + '.png', dpi=200)
	print('saving image %i' % n_image)
	plt.close()

save_folder = './result_' + args.save_folder + '_' + args.folder              #*********************

if not os.path.isdir(save_folder):
	os.makedirs(save_folder)

if not os.path.isdir(save_folder + '/test_kinect/'):
	os.makedirs(save_folder + '/test_kinect/')
if not os.path.isdir(save_folder + '/test_105322251564/'):
	os.makedirs(save_folder + '/test_105322251564/')
if not os.path.isdir(save_folder + '/test_105322251225/'):
	os.makedirs(save_folder + '/test_105322251225/')
if not os.path.isdir(save_folder + '/test_046122250168/'):
	os.makedirs(save_folder + '/test_046122250168/')

index = np.load(sequence_folder + 'index.npy')








pixel_coordinate, bounding_box_p, target_center_holo = cameras_transfer(center_p_rgb)
pixel_coordinate_left, bounding_box_p_left, target_center_holo_left = cameras_transfer(center_p_rgb_left)


np.save(save_folder + '/center.npy',pixel_coordinate)


# for n in range(index.shape[0]):
# 	path_image = sequence_folder + 'kinect_000684312712/'
# 	save_path = save_folder + '/test_kinect/'
# 	draw_bounding(n,index[n], path_image, save_path, pixel_coordinate,
# 				  bounding_box_p, pixel_coordinate_left, bounding_box_p_left)
	#
	# kinect_depth = cv2.imread(os.path.join('./data/20220922/',args.folder,'kinect_000684312712/',
	# 									   f"depth_{index[n]:06d}.png"),-1)
	#
	# kinect_color = cv2.imread(os.path.join('./data/20220922/',args.folder,'kinect_000684312712/',
	# 									   f"color_{index[n]:06d}.jpg"),-1)
	#
	# segment_train_color_path = os.path.join('./segment_train_data', args.folder, 'original_color_images/')
	#
	# segment_train_crop_path = os.path.join('./segment_train_data',args.folder,'crop_images/')
	#
	# segment_train_image_path = os.path.join('./segment_train_data',args.folder,'original_depth_images/')
	#
	# segment_train_txt_path = os.path.join('./segment_train_data',args.folder,'box_txt/')
	#
	# if not os.path.isdir(segment_train_color_path):
	# 	os.makedirs(segment_train_color_path)
	#
	# if not os.path.isdir(segment_train_crop_path):
	# 	os.makedirs(segment_train_crop_path)
	#
	# if not os.path.isdir(segment_train_image_path):
	# 	os.makedirs(segment_train_image_path)
	#
	# if not os.path.isdir(segment_train_txt_path):
	# 	os.makedirs(segment_train_txt_path)
	#
	# cv2.imwrite(os.path.join(segment_train_color_path, f"color_{index[n]:06d}.jpg"), kinect_color)
	# cv2.imwrite(os.path.join(segment_train_image_path,f"depth_{index[n]:06d}.png"),kinect_depth)
	#
	#
	# x1 = int(bounding_box_p[n][0][0])
	# y1 = int(bounding_box_p[n][0][1])
	#
	# x2 = int(bounding_box_p[n][2][0])
	# y2 = int(bounding_box_p[n][2][1])
	#
	# x1_left = int(bounding_box_p_left[n][0][0])
	# y1_left = int(bounding_box_p_left[n][0][1])
	#
	# x2_left = int(bounding_box_p_left[n][2][0])
	# y2_left = int(bounding_box_p_left[n][2][1])
	#
	# crop_kinect_depth = kinect_depth[y1:y2,x1:x2]
	# crop_kinect_depth_left = kinect_depth[y1_left:y2_left, x1_left:x2_left]
	#
	# cv2.imwrite(os.path.join(segment_train_crop_path, f"depth_right_{index[n]:06d}.png"), crop_kinect_depth)
	# cv2.imwrite(os.path.join(segment_train_crop_path, f"depth_left_{index[n]:06d}.png"), crop_kinect_depth_left)
	# with open(os.path.join(segment_train_txt_path,f"depth_{index[n]:06d}.txt") , 'w') as f:
	#
	# 	save_str_left = str(x1_left) + ' ' + str(y1_left) + ' ' + str(x2_left) + ' ' + str(y2_left)
	# 	save_str =  str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2)
	# 	f.write(save_str)
	# 	f.write('\n')
	# 	f.write(save_str_left)








	#crop_kinect_depth_kmeans(kinect_depth,index[n],bounding_box_p[n],bounding_box_p_left[n])


	#*********************
	# path_image = sequence_folder + '105322251564/'
	# save_path = save_folder + '/test_105322251564/'
	# save_path = save_folder + '/test_105322251564/'
	# draw_bounding(n,index[n], path_image, save_path, pixel_coordinate_0, bounding_box_p_0)
	# path_image = sequence_folder + '105322251225/'
	# save_path = save_folder + '/test_105322251225/'
	# draw_bounding(n,index[n], path_image, save_path, pixel_coordinate_1, bounding_box_p_1)
	# path_image = sequence_folder + '046122250168/'
	# save_path = save_folder + '/test_046122250168/'
	# draw_bounding(n,index[n], path_image, save_path, pixel_coordinate_2, bounding_box_p_2)




world2depth = np.load(sequence_folder + 'world2depth.npy')


world2color = np.load(sequence_folder + 'world2color.npy')       #*********************
# print('world2color.npy',world2color.shape)




def world2pixel(x,intrinsic):
	t = x.copy()
	fx = intrinsic[0]
	fy = intrinsic[1]
	ux = intrinsic[2]
	uy = intrinsic[3]

	t[0] = x[0] * fx / x[2] + ux
	t[1] = x[1] * fy / x[2] + uy
	return t[0:2]


def world2pixel_box(x, intrinsic):

	fx = intrinsic[0]
	fy = intrinsic[1]
	ux = intrinsic[2]
	uy = intrinsic[3]
	


	x[:, 0] = x[:, 0] * fx / x[:, 2] + ux
	x[:, 1] = x[:, 1] * fy / x[:, 2] + uy
	return x[:,0:2]


def world2pixel_holo(x, intrinsic):
	pixel = x.copy()
	fx = intrinsic[0]
	fy = intrinsic[1]
	ux = intrinsic[2]
	uy = intrinsic[3]

	pixel[:, 0] = x[:, 0] * fx / x[:, 2] + ux
	pixel[:, 1] = x[:, 1] * fy / x[:, 2] + uy
	return pixel[:,0:2]


def bounding_box_to_pixel(x,intrinsic,range_box):
	sign = [1,-1]
	bounding_box_w = np.zeros((8,3))
	bounding_box_p = np.zeros((4,2))
	n = 0
	for i in range(2):
		for j in range(2):
			for m in range(2):
				bounding_box_w[n,0] = x[0] + sign[i]*range_box
				bounding_box_w[n,1] = x[1] + sign[j]*range_box
				bounding_box_w[n,2] = x[2] + sign[m]*0.6*range_box
				n = n+1

	result = world2pixel_box(bounding_box_w,intrinsic)


	l_x = np.zeros(8)
	l_y = np.zeros(8)
	for j in range(8):
		l_x[j] = result[j,0]
		l_y[j] = result[j,1]
	bound_x_min = np.min(l_x)
	bound_x_max = np.max(l_x)
	bound_y_min = np.min(l_y)
	bound_y_max = np.max(l_y)

	bounding_box_p[0,:] = np.array([bound_x_min,bound_y_min])
	bounding_box_p[1,:] = np.array([bound_x_max,bound_y_min])
	bounding_box_p[2,:] = np.array([bound_x_max,bound_y_max])
	bounding_box_p[3,:] = np.array([bound_x_min,bound_y_max])
	return bounding_box_p

def depth_bounding_box_to_pixel(x,range_box,shape1,depth_file,lut):
	sign = [1,-1]
	bounding_box_w = np.zeros((8,3))
	bounding_box_p = np.zeros((4,2))
	n = 0
	for i in range(2):
		for j in range(2):
			for m in range(2):
				bounding_box_w[n,0] = x[0] + sign[i]*range_box
				bounding_box_w[n,1] = x[1] + sign[j]*range_box
				bounding_box_w[n,2] = x[2] + sign[m]*range_box*0.4
				n = n+1

	l_x = np.zeros(8)
	l_y = np.zeros(8)
	for p in range(8):
		min_distant_index = find_lut_points(bounding_box_w[p],depth_file,lut)

		l_x[p] = min_distant_index[0]
		l_y[p] = min_distant_index[1]




	bound_x_min = np.min(l_x)
	bound_x_max = np.max(l_x)
	bound_y_min = np.min(l_y)
	bound_y_max = np.max(l_y)

	bounding_box_p[0,:] = np.array([bound_x_min,bound_y_min])
	bounding_box_p[1,:] = np.array([bound_x_max,bound_y_min])
	bounding_box_p[2,:] = np.array([bound_x_max,bound_y_max])
	bounding_box_p[3,:] = np.array([bound_x_min,bound_y_max])
	return bounding_box_p




def draw_holo(abc, holo_color_box, holo_color_box_left, n_image):


	plt.figure()
	plt.imshow(abc)

	for i in range(3):
		plt.plot([holo_color_box[i][0], holo_color_box[i + 1][0]],
				 [holo_color_box[i][1], holo_color_box[i + 1][1]], color='green')
		plt.plot([holo_color_box_left[i][0], holo_color_box_left[i + 1][0]],
				 [holo_color_box_left[i][1], holo_color_box_left[i + 1][1]], color='green')


	plt.plot([holo_color_box[0][0], holo_color_box[3][0]],
			 [holo_color_box[0][1], holo_color_box[3][1]], color='green')
	plt.plot([holo_color_box_left[0][0], holo_color_box_left[3][0]],
			 [holo_color_box_left[0][1], holo_color_box_left[3][1]], color='green')



	if not os.path.isdir(save_folder + '/test_holo_color/'):
		os.makedirs(save_folder + '/test_holo_color/')
	save_path = save_folder + '/test_holo_color/'
	plt.savefig(os.path.join(save_path, 'holo_color_') + str(n_image) + '.png', dpi=200)
	print('saving holo color image %i' % n_image)
	plt.close()





def draw_holo_depth(abc, center,holo_depth_center,range_box,n_image,shape1,
					center_left,holo_depth_center_left,depth_file,lut):


	plt.figure()
	plt.imshow(abc)
	plt.plot(center[0], center[1], 'r+')
	plt.plot(center_left[0], center_left[1], 'r+')
	holo_depth_box = depth_bounding_box_to_pixel(holo_depth_center,range_box,shape1,depth_file,lut)
	holo_depth_box_left = depth_bounding_box_to_pixel(holo_depth_center_left, range_box, shape1,depth_file,lut)
	for i in range(3):
		plt.plot([holo_depth_box[i][0], holo_depth_box[i + 1][0]],
				 [holo_depth_box[i][1], holo_depth_box[i + 1][1]], color='green')
		plt.plot([holo_depth_box_left[i][0], holo_depth_box_left[i + 1][0]],
				 [holo_depth_box_left[i][1], holo_depth_box_left[i + 1][1]], color='green')


	plt.plot([holo_depth_box[0][0], holo_depth_box[3][0]],
			 [holo_depth_box[0][1], holo_depth_box[3][1]], color='green')
	plt.plot([holo_depth_box_left[0][0], holo_depth_box_left[3][0]],
			 [holo_depth_box_left[0][1], holo_depth_box_left[3][1]], color='green')



	if not os.path.isdir(save_folder + '/test_holo_depth/'):
		os.makedirs(save_folder + '/test_holo_depth/')
	save_path = save_folder + '/test_holo_depth/'
	plt.savefig(os.path.join(save_path, 'holo_depth_') + str(n_image) + '.png', dpi=200)
	print('saving holo depth image %i' % n_image)
	plt.close()

def crop_holo_depth(abc, center,holo_depth_center,range_box,proj,n_image,shape1,depth_image,color_abc):



	holo_depth_box = depth_bounding_box_to_pixel(holo_depth_center,range_box,proj,shape1)

	print('************holo_depth_box:')
	print(holo_depth_box)

	array_x_top = int(holo_depth_box[0][1])
	print(array_x_top)
	array_x_down = int(holo_depth_box[3][1])
	print(array_x_down)
	arary_y_left = int(holo_depth_box[0][0])
	arary_y_right = int(holo_depth_box[1][0])

	print('crop_image*********', abc.shape)

	crop_image = abc[array_x_top:array_x_down,arary_y_left:arary_y_right]




	print('crop_image*********', crop_image.shape)



	crop_image_1D = crop_image.reshape(crop_image.shape[0]*crop_image.shape[1],1)

	crop_image_1D_index = np.argwhere(crop_image_1D > 0)


	crop_real_1D = np.zeros((crop_image_1D_index.shape[0],1))

	for i in range(crop_image_1D_index.shape[0]):

		num_index = int(crop_image_1D_index[i][0])
		crop_real_1D[i] = crop_image_1D[num_index]

	#kelbow_visualizer(KMeans(), crop_real_1D, k=(1, 10))
	crop_real_1D = crop_real_1D.reshape(-1,1)
	print('crop_real_1D.shape',crop_real_1D)
	kmeans = KMeans(n_clusters=4,random_state=0).fit(crop_real_1D)
	#kelbow_visualizer(KMeans(), crop_image_1D, k=(1, 10))

	kmeans_result = kmeans.labels_
	# print('kmeans_result.shape',kmeans_result.shape)
	#
	k = 4
	avg = np.zeros(k)
	min_avg = 100000
	min = 0
	for n_k in range(k):
		class_hand = np.argwhere(kmeans_result == n_k)
		sum = 0
		for i in class_hand:
			sum = int(crop_real_1D[int(i)]) + sum
		avg = sum / class_hand.shape[0]
		if(avg<min_avg):
			min_avg = avg
			min = n_k
	class_hand = np.argwhere(kmeans_result == min)

	# class_hand0 = np.argwhere(kmeans_result == 0)
	# sum = 0
	# for i in class_hand0:
	# 	sum = int(crop_real_1D[int(i)])+ sum
	# avg[0] = sum/class_hand0.shape[0]
	#
	# class_hand1 = np.argwhere(kmeans_result == 1)
	# sum = 0
	# for i in class_hand1:
	# 	sum = int(crop_real_1D[int(i)]) + sum
	# avg[1] = sum / class_hand0.shape[0]
	#
	# class_hand2 = np.argwhere(kmeans_result == 2)
	# sum = 0
	# for i in class_hand2:
	# 	sum = int(crop_real_1D[int(i)]) + sum
	# avg[2] = sum / class_hand0.shape[0]
	#
	# class_hand3 = np.argwhere(kmeans_result == 3)
	# sum = 0
	# for i in class_hand3:
	# 	sum = int(crop_real_1D[int(i)]) + sum
	# avg[3] = sum / class_hand0.shape[0]
	#
	# min_hand = np.argmin(avg)

	print(class_hand)
	crop_hand_1D = np.zeros(crop_image_1D.shape)

	for i in class_hand:
		num_index = int(crop_image_1D_index[int(i)][0])
		crop_hand_1D[num_index] = 1000
	crop_hand = crop_hand_1D.reshape(crop_image.shape[0], crop_image.shape[1])



	save_path = './result_' + args.save_folder + '_' + args.folder + '/crop_depth/'

	if not os.path.isdir(save_path):
		os.makedirs(save_path)


	plt.figure()
	plt.imshow(crop_image)
	plt.savefig(save_path + str(n_image) + '.png', dpi=200)

	save_path = './result_' + args.save_folder + '_' + args.folder + '/crop_hand/'

	if not os.path.isdir(save_path):
		os.makedirs(save_path)


	plt.figure()
	plt.imshow(crop_hand)
	plt.savefig(save_path + str(n_image) + '.png', dpi=200)

	#
	proj = np.fromfile('./2022-04-17/depth_lut.bin', dtype=np.float32).reshape(
		(depth_w, depth_h, 2)).transpose((1, 0, 2))

	Hand_point_cloud = np.zeros((class_hand.shape[0],3))
	holo_color_intrinsic = [535.777527, 536.745988, 208.623853, 118.486698]

	pixel = world2pixel_box(Hand_point_cloud, holo_color_intrinsic)
	print('pixel***********',pixel)


	if not os.path.isdir(save_path):
		os.makedirs(save_path)
	plt.savefig(os.path.join(save_path, 'result') + str(n_image) + '.png', dpi=200)





	num = 0
	for i in class_hand:
		num_index = int(crop_image_1D_index[int(i)][0])
		h0 = int(num_index/crop_image.shape[1]) + array_x_top
		w0 = int(num_index%crop_image.shape[1]) + arary_y_left



		Hand_point_cloud[num] = holo_2D_to_3D_point(h0, w0, proj, depth_image)
		#print(holo_2D_to_3D_point(h0, w0, proj, depth_image))
		num = num+1


	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(Hand_point_cloud)







	if not os.path.isdir(save_folder + '/crop_holo_depth/'):
		os.makedirs(save_folder + '/crop_holo_depth/')
	save_path = save_folder + '/crop_holo_depth/'

	#mpimg.imsave(os.path.join(save_path, 'holo_depth_') + str(n_image) + '.png', crop_image)
	o3d.io.write_point_cloud(os.path.join(save_folder, 'holo_depth_') + str(n_image) + '.ply', pcd)
	#cv2.imwrite(os.path.join(save_path, 'holo_depth_') + str(n_image) + '.png', crop_image)
	print('saving holo depth image %i' % n_image)


def save_yolo_format(abc, center,holo_depth_center,range_box,n_image,shape1,
				 center_left,holo_depth_center_left,folder,depth_file,lut):

	holo_depth_box = depth_bounding_box_to_pixel(holo_depth_center, range_box, shape1,depth_file,lut)
	width = holo_depth_box[1][0] - holo_depth_box[0][0]
	height = holo_depth_box[3][1] - holo_depth_box[0][1]
	holo_depth_box_left = depth_bounding_box_to_pixel(holo_depth_center_left, range_box, shape1,depth_file,lut)
	width_left = holo_depth_box_left[1][0] - holo_depth_box_left[0][0]
	height_left = holo_depth_box_left[3][1] - holo_depth_box_left[0][1]

	x_center = (holo_depth_box[1][0] + holo_depth_box[0][0])/2
	y_center = (holo_depth_box[3][1] + holo_depth_box[0][1])/2
	x_center_left = (holo_depth_box_left[1][0] + holo_depth_box_left[0][0])/2
	y_center_left = (holo_depth_box_left[3][1] + holo_depth_box_left[0][1])/2

	h = abc.shape[0]
	w = abc.shape[1]

	# divide x_center and width by image width, and y_center and height by image height
	width_one = float(width/w)
	height_one = float(height/h)
	x_center_one = float(x_center/w)
	y_center_one = float(y_center/h)
	width_one_left = float(width_left/w)
	height_one_left = float(height_left/h)
	x_center_one_left = float(x_center_left/w)
	y_center_one_left = float(y_center_left/h)

	if not os.path.isdir(save_folder + '/labels/'+ folder+'/' ):
		os.makedirs(save_folder + '/labels/'+ folder+'/')

	if (n_image < 10):
		num_index_str = '00000' + str(n_image)

	elif (n_image >= 1000):
		num_index_str = '00' + str(n_image)

	elif (n_image >= 100 and n_image < 1000):
		num_index_str = '000' + str(n_image)
	else:
		num_index_str = '0000' + str(n_image)

	with open(save_folder + '/labels/'+ folder +'/' + args.folder + num_index_str + '.txt', 'w') as f:
		save_str = '0 ' + str(x_center_one) + ' ' + str(y_center_one) + ' ' + str(width_one) + ' ' + str(height_one)
		save_str_left = '0 ' + str(x_center_one_left) + ' ' + str(y_center_one_left) + ' ' + str(width_one_left) + ' ' + str(height_one_left)
		f.write(save_str)
		f.write('\n')
		f.write(save_str_left)













num_image = index.shape[0]








def holo_2D_to_3D_point(h0,w0,proj,depth_image):

	u = int(h0)
	v = int(proj.shape[1]-w0)
	x_unit = proj[u,v,0]
	y_unit = proj[u,v,1]

	z = depth_image[h0,w0] / 1000


	c = z/math.sqrt(1+math.pow(x_unit,2)+math.pow(y_unit,2))

	x = x_unit * c
	y = y_unit * c
	real_z = math.sqrt(math.pow(z,2)-math.pow(x,2)-math.pow(y,2))

	return np.array([x,y,real_z])


def holo_2D_to_3D(abc, proj, center, depth_image, intrinsic, range_box, n_image,n):
	h0 = int(center[1])
	w0 = int(center[0])
	u = int(center[1])
	v = int(proj.shape[1] - center[0])
	x_unit = proj[u,v,0]
	y_unit = proj[u,v,1]

	z = depth_image[h0,w0] / 1000





	c = z/math.sqrt(1+math.pow(x_unit,2)+math.pow(y_unit,2))

	x = x_unit * c
	y = y_unit * c
	real_z = math.sqrt(math.pow(z,2)-math.pow(x,2)-math.pow(y,2))


	tf = np.dot(world2color[n], np.linalg.inv(world2depth[n]))


	hand_center_3D_depth = np.array([x,y,real_z,1])
	hand_center_3D_color = np.dot(tf,hand_center_3D_depth)[0:3]
	bounding_box_p = bounding_box_to_pixel(hand_center_3D_color, intrinsic, range_box)
	center_p = world2pixel(hand_center_3D_color,intrinsic)

	plt.figure()
	plt.imshow(abc)
	plt.plot(center_p[0], center_p[1], 'r+')

	for i in range(3):
		plt.plot([bounding_box_p[i][0], bounding_box_p[i + 1][0]],
				 [bounding_box_p[i][1], bounding_box_p[i + 1][1]], color='green')

	plt.plot([bounding_box_p[0][0], bounding_box_p[3][0]],
			 [bounding_box_p[0][1], bounding_box_p[3][1]], color='green')

	save_path = './result_' + args.save_folder + '_' + args.folder + '/test_holo_color/'


	if not os.path.isdir(save_path):
		os.makedirs(save_path)
	plt.savefig(os.path.join(save_path, 'result') + str(n_image) + '.png', dpi=200)
	print('saving image %i' % n_image)
	plt.close()




save_folder = './result_' + args.save_folder + '_' + args.folder  # *********************


if not os.path.isdir(save_folder):
	os.makedirs(save_folder)

if not os.path.isdir(save_folder + '/test_kinect/'):
	os.makedirs(save_folder + '/test_kinect/')





def find_lut_points(depth_xyz,depth_file,lut, depth_w=512,depth_h=512):
	# start_time = time.time()
	# depth_scale = 1000.0
	d = np.linalg.norm(depth_xyz)
	# img = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH).astype(np.float64)
	# depth_h, depth_w = img.shape
	# img /= depth_scale

	# for i in range(depth_h):
	# 	for j in range(depth_w):
	# 		img[i][j] = d
	img = np.full(lut.shape, d)
	# img = np.tile(img.flatten().reshape((-1, 1)), (1, 3))


	points = img * lut
	distant = np.zeros(img.shape[0])
	# min_distant_index = np.zeros(2)
	# for i in range(img.shape[0]):
	# 	distant[i] = np.linalg.norm(depth_xyz - points[i])
	#
	# min_distant_n = np.argwhere(distant == distant.min())

	dd = np.linalg.norm(points - depth_xyz, axis=1)
	min_distant_n = np.argmin(dd)
	# print("min_distant_n： ", min_distant_n)
	# min_distant_index[0] = int(min_distant_n%depth_w)
	# min_distant_index[1] = int(min_distant_n/depth_w)



	x = int(min_distant_n%depth_w)
	y = int(min_distant_n/depth_w)



	# total_time = time.time() - start_time
	# print("total_time:", str(total_time))

	# return min_distant_index
	return (x,y)








def holo_3D_to_2D_color(sequence_folder, world2color, target_center_holo, target_center_holo_left, index,camera_intrinsic):
	center_all = np.zeros((num_image, 2))
	tf_sum = np.zeros((4,4))

	if not os.path.isdir(save_folder + '/test_crop_color/'):
		os.makedirs(save_folder + '/test_crop_color/')
	save_path = save_folder + '/test_crop_color/'

	if not os.path.isdir(save_folder + '/test_crop_color_left/'):
		os.makedirs(save_folder + '/test_crop_color_left/')
	save_path_left = save_folder + '/test_crop_color_left/'

	for n in range(num_image):



		holo_color_coordinate = np.dot(world2color[n], np.append(target_center_holo[n], [1]))[0:3]
		holo_color_coordinate_left = np.dot(world2color[n], np.append(target_center_holo_left[n], [1]))[0:3]





		intrinsic_matrix = camera_intrinsic[n]
		holo_color_intrinsic = np.zeros(4)
		holo_color_intrinsic[0] = intrinsic_matrix[0][0]
		holo_color_intrinsic[1] = intrinsic_matrix[1][1]
		holo_color_intrinsic[2] = intrinsic_matrix[0][2]
		holo_color_intrinsic[3] = intrinsic_matrix[1][2]

		holo_color_box = bounding_box_to_pixel(holo_color_coordinate, holo_color_intrinsic, 0.08)
		holo_color_box_left = bounding_box_to_pixel(holo_color_coordinate_left, holo_color_intrinsic, 0.08)



		abc_color = cv2.imread(os.path.join(sequence_folder, f"hololens2/color_{index[n]:06d}.jpg"),-1)
		#abc_color = cv2.cvtColor(abc_color, cv2.COLOR_BGR2RGB)


		height_color = int(abc_color.shape[0])
		width_color = int(abc_color.shape[1])

		w = int(holo_color_box[2][0]-holo_color_box[0][0])
		h = int(holo_color_box[2][1]-holo_color_box[0][1])

		w_left = int(holo_color_box_left[2][0] - holo_color_box_left[0][0])
		h_left = int(holo_color_box_left[2][1] - holo_color_box_left[0][1])

		x1 = int(holo_color_box[0][0])
		y1 = int(holo_color_box[0][1])
		x2 = int(holo_color_box[2][0])
		y2 = int(holo_color_box[2][1])

		x1_left = int(holo_color_box_left[0][0])
		y1_left = int(holo_color_box_left[0][1])
		x2_left = int(holo_color_box_left[2][0])
		y2_left = int(holo_color_box_left[2][1])

		if ((width_color-x1)>0.75*w and (height_color-y1)>0.75*h and x2>0.75*w and y2>0.75*h):
			crop_color = abc_color[y1:y2,x1:x2]
			try:
				cv2.imwrite(os.path.join(save_path,f"crop_color_{index[n]}.jpg"),crop_color)
			except cv2.error as error:
				print("[Error]: {}".format(error))

		if ((width_color - x1_left) > 0.75 * w_left and (height_color - y1_left) > 0.75 * h_left and x2_left > 0.75 * w_left
				and y2_left > 0.75 * h_left):

			crop_color_left = abc_color[y1_left:y2_left, x1_left:x2_left]
			try:
				cv2.imwrite(os.path.join(save_path_left, f"crop_color_{index[n]}.jpg"), crop_color_left)
			except cv2.error as error:
				print("[Error]: {}".format(error))






		#draw_holo(abc_color,holo_color_box,holo_color_box_left, index[n])








def holo_3D_to_2D(sequence_folder, world2depth, target_center_holo, target_center_holo_left, index,lut):
	center_all = np.zeros((num_image, 2))
	tf_sum = np.zeros((4,4))




	for n in range(num_image):
		# if (index[n] < 10):
		# 	num_index_str = '00000' + str(index[n])
		#
		# elif (index[n] >= 1000):
		# 	num_index_str = '00' + str(index[n])
		#
		# elif (index[n] >= 100 and index[n] < 1000):
		# 	num_index_str = '000' + str(index[n])
		# else:
		# 	num_index_str = '0000' + str(index[n])

		# depth_file = sequence_folder + 'hololens2/' + 'depth_' + num_index_str + '.png'



		depth_file = os.path.join(sequence_folder, f"hololens2/depth_{index[n]:06d}.png")



		tf = np.dot(world2color[n], np.linalg.inv(world2depth[n]))
		# print('************tf:', tf)
		# print('************n:',n)
		tf_sum = tf_sum + tf
		holo_depth_coordinate = np.dot(world2depth[n], np.append(target_center_holo[n], [1]))[0:3]
		holo_depth_coordinate_left = np.dot(world2depth[n], np.append(target_center_holo_left[n], [1]))[0:3]


		min_distant_index = find_lut_points(holo_depth_coordinate,depth_file,lut)
		min_distant_index_left = find_lut_points(holo_depth_coordinate_left,depth_file,lut)



		# *********************88

		# holo_depth_image = cv2.imread(
		# 	sequence_folder + 'hololens/' + 'depth_' + num_index_str + '.png',-1)
		holo_depth_image = cv2.imread(depth_file, -1)

		abc = np.array(holo_depth_image) * 50

		# abc = np.array(holo_depth_image)

		# abc = cv2.cvtColor(abc, cv2.COLOR_BGR2RGB)
		# print('abc.shape[1]', abc.shape[1])
		#
		# convert_min_distant_index = np.zeros(2)
		# convert_min_distant_index[0] = abc.shape[1] - min_distant_index[1]
		# convert_min_distant_index[1] = min_distant_index[0]
		# convert_min_distant_index_left = np.zeros(2)
		# convert_min_distant_index_left[0] = abc.shape[1] - min_distant_index_left[1]
		# convert_min_distant_index_left[1] = min_distant_index_left[0]
		#
		# center_all[n][0] = convert_min_distant_index[0]
		# center_all[n][1] = convert_min_distant_index[1]



		draw_holo_depth(abc, min_distant_index, holo_depth_coordinate, 0.10, index[n],
						abc.shape[1],min_distant_index_left, holo_depth_coordinate_left,depth_file,lut)  # ********************

		save_yolo_format(abc, min_distant_index, holo_depth_coordinate, 0.10, index[n], abc.shape[1],
						 min_distant_index_left, holo_depth_coordinate_left,args.save_folder,depth_file,lut)

		# holo_color_image0 = cv2.imread(sequence_folder + 'hololens/' + 'color_' + num_index_str + '.jpg')
		# color_abc = cv2.cvtColor(holo_color_image0, cv2.COLOR_BGR2RGB)
		# print('max_abc:',np.max(color_abc))
		# print('min_abc:', np.min(color_abc))
		# print('abc.shape:',color_abc.shape)

		# holo_depth_image0 = cv2.imread(sequence_folder + 'hololens/' + 'depth_' + num_index_str + '.png',-1)

		# holo_color_intrinsic = [535.777527, 536.745988, 208.623853, 118.486698]
#
# ***********************
# 		print(holo_depth_image0.shape)
# 		holo_depth_image0_1D= holo_depth_image0.reshape(holo_depth_image0.shape[0]*holo_depth_image0.shape[1],1)
# 		print(holo_depth_image0_1D.shape)
# 		plt.figure()
# 		plt.imshow(holo_depth_image0)
#
# 		save_path = './result_' + args.save_folder + '_' + args.folder + '/original_holo_depth/'
#
# 		if not os.path.isdir(save_path):
# 			os.makedirs(save_path)
# 		plt.savefig(save_path + str(index[n]) + '.png', dpi=200)
# 		print('saving image %i' % index[n])
# **************************
		#kelbow_visualizer(KMeans(), holo_depth_image0_1D, k=(1, 10))

		#crop_holo_depth(abc, convert_min_distant_index, holo_depth_coordinate, 0.12, proj, index[n], abc.shape[1],
		#				holo_depth_image0,color_abc)
		#holo_2D_to_3D(color_abc, proj, convert_min_distant_index, holo_depth_image0, holo_color_intrinsic, 0.08, index[n],n)

	tf_avg = tf_sum / 20



if __name__=="__main__":
	curr_dir = os.path.dirname(os.path.abspath(__file__))
	print("saving images of ", args.folder)

	lut_file = os.path.join(curr_dir,"./2022-04-17/depth_lut.bin")
	with open(lut_file, mode="rb") as lut_file:
		lut = np.frombuffer(lut_file.read(), dtype="f")
		lut = np.reshape(lut, (-1, 3))
	start_t = time.time()
	holo_3D_to_2D_color(sequence_folder, world2color, target_center_holo,target_center_holo_left, index,camera_info)

	holo_3D_to_2D(sequence_folder, world2depth, target_center_holo,target_center_holo_left, index,lut)
	end_t = time.time()
	print("frame_time:",end_t-start_t)




