import numpy as np
from math import sqrt
from util import rgb2gray
import matplotlib.pyplot as plt
from PIL import Image
from skimage.restoration import estimate_sigma

HAAR_MATRIX = np.array([[1,1,1,1, 1,1,1,1],
					[1,1,1,1, -1,-1,-1,-1],
					[sqrt(2),sqrt(2),-sqrt(2),-sqrt(2), 0,0,0,0],
					[0,0,0,0, sqrt(2),sqrt(2),-sqrt(2),-sqrt(2)],
					[2,-2,0,0, 0,0,0,0],
					[0,0,2,-2, 0,0,0,0],
					[0,0,0,0, 2,-2,0,0],
					[0,0,0,0, 0,0,2,-2]]) / (sqrt(8))
INV_HAAR_MATRIX = np.transpose(HAAR_MATRIX, axes=(1,0))


def haar_2d(neighboring_cubes, haar_matrix, inv_haar_matrix, gamma=3, noise_sigma=1, enhance=True):
	assert len(neighboring_cubes) % 2 == 0
	haar_transform = []
	inv_haar_transform = []
	for i in range(len(neighboring_cubes)):
		haar_cube = np.zeros_like(neighboring_cubes[0])
		for j in range(len(neighboring_cubes)):
			haar_cube += neighboring_cubes[j] * haar_matrix[i,j]
		haar_transform.append(haar_cube)
	if enhance:
		haar_transform = enhance_haar(haar_transform, noise_sigma, gamma=gamma)
	for i in range(len(neighboring_cubes)):
		invhaar_cube = np.zeros_like(neighboring_cubes[0])
		for j in range(len(neighboring_cubes)):
			invhaar_cube += haar_transform[j] * inv_haar_matrix[i,j]
		inv_haar_transform.append(invhaar_cube)
	return inv_haar_transform


def enhance_haar(cubes, noise_sigma=1, threshold_1=30.0, threshold_2=8.0, gamma=3):
	enhanced_cubes = []
	for cube in cubes:
		cube_tmp = np.array(cube)
		cube_tmp[np.absolute(cube)<=(threshold_1*noise_sigma)] = cube[np.absolute(cube)<=(threshold_1*noise_sigma)] * gamma
		cube_tmp[np.absolute(cube)<(threshold_2*noise_sigma)] = 0
		enhanced_cubes.append(cube_tmp)
	return enhanced_cubes


def neighbor_list(kernel, num_neighbor, shift=0):
	availible_neighbors = [] 
	for i in range(kernel-1):
		availible_neighbors.append((kernel//2, kernel//2-i))
	for i in range(kernel-1):
		availible_neighbors.append((kernel//2-i, -(kernel//2)))
	for i in range(kernel-1):
		availible_neighbors.append((-(kernel//2), kernel//2-i))
	for i in range(kernel-1):
		availible_neighbors.append((kernel//2-i, kernel//2))	
	assert len(availible_neighbors) >= num_neighbor
	
	return availible_neighbors[shift::(len(availible_neighbors)//num_neighbor)]


def nonlocal_haar(img, noise_sigma, gamma=3, haar_matrix=HAAR_MATRIX, inv_haar_matrix=INV_HAAR_MATRIX,kernel=3, step=8, num_neighbor=8, cube_size=7,  start=None, end=None, enhance=True):

	image_shape = img.shape
	neighbors = neighbor_list(kernel, num_neighbor)
	tmp_img = np.zeros_like(img)
	weight = np.zeros_like(img)
	start = (kernel//2, kernel//2) if start==None else start
	end = (image_shape[0]-cube_size-(kernel//2), image_shape[1]-cube_size-(kernel//2))
	row_list = list(range(start[0], end[0], step))
	column_list = list(range(start[1], end[1], step))
	row_list.append(end[0])
	column_list.append(end[1])
	for i in row_list:
		for j in column_list:
			k_neighbors = []
			for neighbor in neighbors:
				cube = img[i+neighbor[0]:i+neighbor[0]+cube_size, j+neighbor[1]:j+neighbor[1]+cube_size]
				k_neighbors.append(cube)
			haar_neighbors = haar_2d(k_neighbors, haar_matrix, inv_haar_matrix, noise_sigma=noise_sigma, gamma=gamma, enhance=enhance)
			
			for index, neighbor in enumerate(neighbors):
				weight[i+neighbor[0]:i+neighbor[0]+cube_size,j+neighbor[1]:j+neighbor[1]+cube_size] += 1
				tmp_img[i+neighbor[0]:i+neighbor[0]+cube_size,j+neighbor[1]:j+neighbor[1]+cube_size] += haar_neighbors[index]
	weight[weight==0] = 1
	haar_img = tmp_img / weight
	return haar_img

if __name__=="__main__":
	#img_file = './bm3d_demos/image_Lena512rgb.png'
	#img_file = './data/filter_test/slice_4.png'
	img_file = './data/filter_test/Lena-original-gray.png'
	img_rgb = np.array(Image.open(img_file))
	img = rgb2gray(img_rgb) if img_rgb.ndim > 2 else img_rgb
	#plt.imshow(img, cmap='gray')
	#plt.show()
	img = np.array(img, dtype=np.float32)
	img = img / np.amax(img)
	sigmas = estimate_sigma(img, multichannel=False)
	noise_sigma = np.mean(sigmas)
	print(noise_sigma)
	img_haar = nonlocal_haar(img, noise_sigma=noise_sigma*0.25, gamma=3)
	#img_haar_2 = nonlocal_haar(img, noise_sigma=noise_sigma*0.15, gamma=5)

	fig, axes = plt.subplots(1)
	diff = img_haar - img
	axes.imshow(np.concatenate((img/np.amax(img), img_haar/np.amax(img_haar)), axis=1), cmap='gray')
	#axes[0].imshow(img, cmap='gray')
	#axes[1].imshow(img_haar, cmap='gray')
	#axes[2].imshow(img-img_haar, cmap='gray')
	#plt.tight_layout()
	plt.show()
