import torch
import scatter_cuda
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

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
	
ksize = 21
# kernel = (distance_kernel(ksize) * lens_shape_mask(ksize))
kernel, lens_mask = distance_kernel(ksize), lens_shape_mask(ksize)

device=torch.device('cuda:0')
to_tensor, padding = transforms.ToTensor(), torch.nn.ReplicationPad2d(ksize//2)
img = to_tensor(Image.open('rgb_test.png').convert('RGB').resize((256,256))).unsqueeze(dim=0).repeat(5,1,1,1)
img2 = torch.zeros(1,3,256,256)
img2[:,:,50:200,50:200] = 1.0
img = torch.cat((img, img2), dim=0)


b,c,h,w = img.shape
lens_effect = 10.0
disp = torch.abs(1.0-torch.linspace(0.0,1.0,h)).expand(b,1,w,h).permute(0,1,3,2)
img, kernel, lens_mask = padding(torch.cat((img, disp*lens_effect), dim=1)).to(device), kernel.to(device), lens_mask.to(device)
blur = torch.zeros(b, 3, h, w).to(device)

import pdb; pdb.set_trace()
import time
start = time.time()
y = scatter_cuda.forward(img, kernel, lens_mask, blur)
end = time.time()
print("Time: {}s".format(end-start))

bi = y.shape[0]
for i in range(bi):
	t = y[i].detach().cpu().numpy().transpose(1,2,0).copy(order='c')
	plt.imsave('{}.png'.format(i), t)

plt.imsave('ori.png', img[0, :3].detach().cpu().numpy().transpose(1,2,0).copy(order='c'))
print(img.shape, kernel.shape, y.shape)
