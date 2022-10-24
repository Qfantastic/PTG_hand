import os
from PIL import Image
import numpy as np

# image_path = './result/'

# files = os.listdir(image_path)



def getgif(image_path,sn,index,filename):
	file_first_path = image_path + filename + str(index[0]) + '.png'
	img = Image.open(file_first_path)
	images = []
	print('saving ' + sn + '.gif')
	for i in index[1:]:
		img_path = image_path + filename +str(i) + '.png'
		images.append(Image.open(img_path))
		img.save('gif/'+sn+'.gif', save_all = True, append_images = images, loop=0, duration=150)


image_path = './gif/'
index = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
				  22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
getgif(image_path,'holo_color',index,'holo_color_')


#
# image_path = './result_0610_0/test1/'
# index = np.array([268, 270, 272, 273, 275, 277, 278, 279, 280, 282, 284, 286, 287])
# getgif(image_path,'holo_depth',index,'holo_depth_')
#
# image_path = './result_0610_0/test2/'
# index = np.array([533, 537, 539, 540, 542, 544, 549, 563, 567, 569, 574, 576])
# getgif(image_path,'holo_depth0',index,'holo_depth_')
#
#
# image_path = './result_0610_1/test_holo_depth/'
# index = np.array([71, 73, 74, 76, 78, 79, 81, 83, 85, 87, 89, 91, 93, 95, 99, 101, 103,
# 				  105, 107, 109, 112, 116, 120, 124, 128, 130])
# getgif(image_path,'holo_depth1',index,'holo_depth_')
#
# # image_path = './result_0530_1/holo_1_raw/'
# # index = np.array([71, 73, 74, 76, 78, 79, 81, 83, 85, 87, 89, 91, 93, 95, 99, 101, 103,
# # 				  105, 107, 109, 112, 116, 120, 124, 128, 130])
# # getgif(image_path,'holo_depth1_raw',index,'holo_depth_')
# #
# #
# # image_path = './result_0530_1/kinect_1/'
# # index = np.array([71, 73, 74, 76, 78, 79, 81, 83, 85, 87, 89, 91, 93, 95, 99, 101, 103,
# # 				  105, 107, 109, 112, 116, 120, 124, 128, 130])
# # getgif(image_path,'kinect_1',index,'result')
#
# image_path = './result_0610_2/test_holo_depth/'
# index = np.array([98, 114, 120, 124, 126, 130, 132, 134, 138, 140, 142, 147, 151])
# getgif(image_path,'holo_depth2',index,'holo_depth_')


# image_path = './result_0530_2/kinect2/'
# index = np.array([114, 120, 124, 126, 130, 132, 134, 138, 140, 142, 147, 151])
# getgif(image_path,'kinect2',index,'result')