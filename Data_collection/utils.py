import numpy as np
import yaml

# file = open('parameters/extrinsics/037522251142.yml').read()
# all_tag = yaml.safe_load(file)
# fff = float(all_tag['rotation'][0])

# print(fff)




class Camera():

	def __init__(self, center, box_range = 0.10, SN = None) :
		self.center = center
		self.range = box_range
		self.num = np.array(center).shape[0]

		#file = open('parameters/intrinsics/105322251564_640x480.yml').read()
		file = open('2022-04-17/intrinsics/105322251564_640x480.yml').read()
		all_tag = yaml.safe_load(file)
		self.fx_0 = float(all_tag['color']['fx'])
		self.fy_0 = float(all_tag['color']['fy'])
		self.ux_0 = float(all_tag['color']['ppx'])
		self.uy_0 = float(all_tag['color']['ppy'])

		#file = open('parameters/intrinsics/' + SN + '_640x480.yml').read()
		file = open('2022-04-17/intrinsics/' + SN + '_640x480.yml').read()
		all_tag = yaml.safe_load(file)
		self.fx_1 = float(all_tag['color']['fx'])
		self.fy_1 = float(all_tag['color']['fy'])
		self.ux_1 = float(all_tag['color']['ppx'])
		self.uy_1 = float(all_tag['color']['ppy'])

		# file = open('parameters/extrinsics/' + SN + '.yml').read()
		file = open('2022-04-17/extrinsics/' + SN + '.yml').read()
		all_tag = yaml.safe_load(file)
		self.T_r = np.zeros((3,3))
		self.T_t = np.zeros(3)
		for i in range(3):
			for j in range(3):
				self.T_r[i][j] = float(all_tag['rotation'][i*3 + j])

		for m in range(3):
			self.T_t[m] = float(all_tag['translation'][m])



	def pixel2world(self):
		x = np.zeros(self.center.shape)
		x[:, 0] = (self.center[:, 0] - self.ux) * self.center[:, 2] / self.fx
		x[:, 1] = (self.center[:, 1] - self.uy) * self.center[:, 2] / self.fy
		return x

	def world2pixel_target(self,x):
		t = np.zeros(x.shape)
		t[:, 0] = x[:, 0] * self.fx_1 / x[:, 2] + self.ux_1
		t[:, 1] = x[:, 1] * self.fy_1 / x[:, 2] + self.uy_1
		return t[:,0:2]

	def world2pixel(self,x):
		t = np.zeros(x.shape)
		t[:, 0] = x[:, 0] * self.fx_0 / x[:, 2] + self.ux_0
		t[:, 1] = x[:, 1] * self.fy_0 / x[:, 2] + self.uy_0
		return t[:,0:2]

	# def world2pixel(self):
	# 	x = np.zeros(self.center.shape)
	# 	x[:, 0] = self.center[:, 0] * self.fx_0 / self.center[:, 2] + self.ux_0
	# 	x[:, 1] = self.center[:, 1] * self.fy_0 / self.center[:, 2] + self.uy_0
	# 	return x[:,0:2]

	def world2pixel_box(self, x, fl):

		if fl ==0 :
			x[:,:, 0] = x[:,:, 0] * self.fx_0 / x[:,:, 2] + self.ux_0
			x[:,:, 1] = x[:,:, 1] * self.fy_0 / x[:,:, 2] + self.uy_0
			return x[:,:,0:2]

		if fl ==1 :
			x[:,:, 0] = x[:,:, 0] * self.fx_1 / x[:,:, 2] + self.ux_1
			x[:,:, 1] = x[:,:, 1] * self.fy_1 / x[:,:, 2] + self.uy_1
			return x[:,:,0:2]


	def transfer_camera_inv(self):

		target_coordinate = np.zeros((self.num,3))
		for i in range(self.num):
			target_coordinate[i] = self.center[i] - self.T_t
			target_coordinate[i] = np.dot(np.linalg.inv(self.T_r), target_coordinate[i])

		return target_coordinate


	def transfer_camera(self):
		target_coordinate = np.zeros((self.num,3))
		for i in range(self.num):
			target_coordinate[i] = np.dot(self.T_r,self.center[i])
			target_coordinate[i] = target_coordinate[i] - self.T_t

		return target_coordinate





	def bounding_box_to_pixel(self,x,fl):
		sign = [1,-1]
		bounding_box_w = np.zeros((self.num,8,3))
		bounding_box_p = np.zeros((self.num,4,2))
		n = 0
		for i in range(2):
			for j in range(2):
				for m in range(2):
					bounding_box_w[:,n,0] = x[:,0] + sign[i]*self.range
					bounding_box_w[:,n,1] = x[:,1] + sign[j]*self.range
					bounding_box_w[:,n,2] = x[:,2] + sign[m]*self.range*0.6
					n = n+1

		result = self.world2pixel_box(bounding_box_w,fl)

		for i in range(self.num):
			l_x = np.zeros(8)
			l_y = np.zeros(8)
			for j in range(8):
				l_x[j] = result[i,j,0]
				l_y[j] = result[i,j,1]
			bound_x_min = np.min(l_x)
			bound_x_max = np.max(l_x)
			bound_y_min = np.min(l_y)
			bound_y_max = np.max(l_y)

			bounding_box_p[i,0,:] = np.array([bound_x_min,bound_y_min])
			bounding_box_p[i,1,:] = np.array([bound_x_max,bound_y_min])
			bounding_box_p[i,2,:] = np.array([bound_x_max,bound_y_max])
			bounding_box_p[i,3,:] = np.array([bound_x_min,bound_y_max])
		return bounding_box_p

# center_p = np.load('hand_center_p.npy')
# mycamera = Camera(center_p, 0.10, '000684312712')
# print(mycamera.T_r)