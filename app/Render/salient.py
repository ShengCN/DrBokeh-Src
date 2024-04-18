import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '.')

from app.Render.Salient.LDF.train_fine.net import LDF
from app.Render.Salient.LDF.train_fine import dataset


class Salient_Inference:
    def __init__(self):
        model_snapshot = 'app/Render/Salient/LDF/train_fine/out/model-40'
        resnet_path    = 'app/Render/Salient/LDF/res/resnet50-19c8e357.pth'

        self.cfg = dataset.Config(datapath='',
                                  snapshot=model_snapshot,
                                  mode='test')
        self.net = LDF(self.cfg, resnet_path)

        self.net.train(False)
        self.net.cuda()


        self.normalize = Normalize(mean=self.cfg.mean, std=self.cfg.std)
        self.resize    = Resize(352, 352)
        self.totensor  = ToTensor()



    def inference(self, rgb: np.array):
        """ Given the RGB np.array image, compute the salient map and return

        :param rgb: H x W x 3, np.array
        :returns: H x W x 1, np.array

        """
        assert rgb.shape[2] == 3, 'Salient inference input should have 3 channels({})'.format(rgb.shape[2])

        with torch.no_grad():
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255.0).astype(np.uint8)
            # image = torch.tensor(rgb.transpose((2,0,1)))[None, ...].cuda().float()

            H, W = rgb.shape[:2]
            image = self.transform(rgb).cuda().float()
            image, shape  = image.cuda().float(), (H, W)
            outb1, outd1, out1, outb2, outd2, out2 = self.net(image, shape)
            out  = out2
            pred = torch.sigmoid(out[0,0]).cpu().numpy()

            return pred


    def transform(self, img: np.array):
        """ Transform the np array to the specific format

        :param rgb: H x W x 3
        :returns: 352 x 352 torch.tensor

        """
        img = self.normalize(img)
        img = self.resize(img)
        img = self.totensor(img)[None, ...]
        return img



"""
---------------------------------------------------
"""
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, image, mask=None, body=None, detail=None):
        image = (image - self.mean)/self.std
        if mask is None:
            return image
        return image, mask/255, body/255, detail/255


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, body=None, detail=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        body  = cv2.resize( body, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        detail= cv2.resize( detail, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, body, detail


class ToTensor(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask  = torch.from_numpy(mask)
        body  = torch.from_numpy(body)
        detail= torch.from_numpy(detail)
        return image, mask, body, detail

