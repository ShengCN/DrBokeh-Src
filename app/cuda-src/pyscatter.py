""" Pytorch Scattering Layer Implementation 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

def soft_step(a,b):
    """ Approxiate Differentiable a > b 
    """
    return 1/(1+torch.exp(-(a-b-0.05)*100))

def distance_kernel(size):
	xgrid = (torch.arange(size)-size//2).repeat(size).view(size, size)
	disfield = torch.stack((xgrid, xgrid.T),dim=-1)
	return torch.sqrt(torch.pow(disfield[...,0],2) + torch.pow(disfield[...,1], 2))

def lens_shape_mask(size, shape='circle'):
	def circle(size):
		kernel = torch.zeros((size, size))
		diskernel = distance_kernel(size)
		kernel[diskernel<size//2] = 1.0
		return kernel

	lens_shape = {
		'circle':circle
	}
	if shape not in lens_shape.keys():
		raise ValueError("Shape input {} has not been implemented yet".format(shape))

	return lens_shape[shape](size)

def scatter(x, lens=21, lens_effects=20):
	""" Scatteirng Rendering Layer 
			x ~ 4 x H x W: 	relative disparity w.r.t. focal plane
			lens:			lens size
			lens_effects:   lens scattering strength
	"""
	b, c, h, w = x.shape
	if c != 4 or lens % 2 == 0:
		raise ValueError("Scattering Input is wrong. {} {}".format(c, lens))
	 
	ret = x.clone()
	# replicate paddign
	padding = torch.nn.ReplicationPad2d(lens//2) 
	paddedx, offset = padding(x).detach(), lens//2

	# scattering kernels
	diskernel, lens_mask = distance_kernel(lens).to(x), lens_shape_mask(lens,'circle').to(x)
	dscatter = torch.abs(paddedx[:,3]) * lens_effects	

	for bi in tqdm(range(b)):
		dips = dscatter[bi, 3]
		for hi in range(offset, h+offset):
			for wi in range(offset, w+offset):
				rgb_neighbours = paddedx[bi, :3, hi-lens//2:hi+lens//2+1, wi-lens//2:wi+lens//2+1] 
				dneighbours = dscatter[bi, hi-lens//2:hi+lens//2+1, wi-lens//2:wi+lens//2+1]

				# reweight by area
				area_reweights = 1./(dneighbours+1)

				# check scattering effects
				weights = soft_step(dneighbours+0.1, diskernel) * area_reweights * lens_mask

				ret[bi,:3, hi-offset, wi-offset] = (rgb_neighbours * weights).sum()/max(weights.sum(),1e-6)
	
	return ret

if __name__ == '__main__':
	import time
	import logging
	log_file = 'test.log'
	logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s - %(filename)s %(funcName)s %(asctime)s ", handlers=[
			logging.FileHandler(log_file),
			logging.StreamHandler()
		])

	test = torch.zeros(1,4,128,128)
	test[:,:,50:100,50:100] = 1.0
	device = torch.device('cpu')
	test = test.to(device)

	start = time.time()
	test_blur = scatter(test, lens=11)
	end = time.time()
	logging.info("CPU: {}s".format(end-start))
	plt.imsave('original.png', np.clip(test[0].detach().cpu().numpy().transpose(1,2,0)[:,:,:3],0.0,1.0).copy(order='C'))
	plt.imsave('cpu.png', np.clip(test_blur[0].detach().cpu().numpy().transpose(1,2,0)[:,:,:3],0.0,1.0).copy(order='C'))

	device = torch.device('cuda:0')
	test = test.to(device)
	start = time.time()
	test_blur = scatter(test, lens=11)
	end = time.time()
	logging.info("GPU: {}s".format(end-start))
	plt.imsave('gpu.png', np.clip(test_blur[0].detach().cpu().numpy().transpose(1,2,0),0.0,1.0)[:,:,:3].copy(order='C'))

	test = torch.zeros(5,4,256,256)
	test[:,:,50:200,50:200] = 1.0

	device = torch.device('cpu')
	test = test.to(device)
	start = time.time()
	test_blur = scatter(test)
	end = time.time()
	logging.info("CPU: {}s".format(end-start))

	device = torch.device('cuda:0')
	test = test.to(device)
	start = time.time()
	test_blur = scatter(test)
	end = time.time()
	logging.info("GPU: {}s".format(end-start))
