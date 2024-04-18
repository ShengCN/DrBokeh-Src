import numpy as np
import logging
import os
import sys
import traceback
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './app/Render/Inpainting/lama')

from app.Render.Inpainting.lama.saicinpainting.evaluation.utils import move_to_device
from app.Render.Inpainting.lama.saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F

import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from app.Render.Inpainting.lama.saicinpainting.training.data.datasets import make_default_val_dataset
from app.Render.Inpainting.lama.saicinpainting.training.trainers import load_checkpoint
from app.Render.Inpainting.lama.saicinpainting.utils import register_debug_signal_handlers

from lama_inpaint.ffc import FFCResNetGenerator


class RGB_Inpainting_Inference:
    def __init__(self):
        config = 'app/Render/Inpainting/lama/configs/prediction/default.yaml'
        with open(config, 'r') as f:
            predict_config = OmegaConf.create(yaml.safe_load(f))

        predict_config.model.path = 'app/Render/Inpainting/lama/big-lama'
        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(predict_config.model.path,
                                       'models',
                                       predict_config.model.checkpoint)

        self.device = torch.device('cuda:0')

        # self.model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        # self.model.freeze()
        # self.model.to(self.device)

        # self.predict_config = predict_config


        config = 'app/Render/lama_inpaint/config.yaml'
        with open(config, 'r') as f:
            predict_config = OmegaConf.create(yaml.safe_load(f))

        self.model = FFCResNetGenerator(**predict_config).to(self.device)
        checkpoint = 'app/Render/lama_inpaint/inpnet.pth'
        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()


    def inference(self, rgb: np.array, mask: np.array): #
        """ Given np array input, do inpainting for the masked regions.

        :param rgb: H x W x 3 np array
        :param mask: H x W x 1 np array
        :returns: Inpainted results

        """
        assert rgb.shape[-1] == 3 and mask.shape[-1] == 1, \
            'RGB and mask channels should be 3({}) and 1({}).'.format(rgb.shape[-1], mask.shape[-1])

        #TODO, feature refinement
        device         = self.device
        # predict_config = self.predict_config
        batch          = {
            'image': torch.tensor(rgb.transpose((2,0,1)))[None, ...].float(),
            'mask': torch.tensor(mask.transpose((2,0,1)))[None, ...].float(),
        }

        ori_h, ori_w = rgb.shape[:2]

        with torch.no_grad():
            batch         = move_to_device(batch, device)
            batch['mask'] = (batch['mask'] > 0) * 1

            try:
                # batch                       = self.model(batch)
                batch = self.model(batch['image'], batch['mask'])

            except BaseException as err:
                print(traceback.format_exc())
                return None

            cur_res = batch[0].detach().cpu().numpy().transpose((1,2,0))[:ori_h, :ori_w]
            # cur_res = cv2.resize(batch[0].detach().cpu().numpy().transpose((1,2,0)),(ori_w, ori_h))

        return cur_res

