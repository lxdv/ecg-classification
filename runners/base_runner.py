import os
import os.path as osp
import numpy as np
from datetime import datetime

import torch
from tqdm import tqdm

from utils.network_utils import load_checkpoint


class BaseRunner:
    def __init__(self, config):
        self.config = config
        self.exp_name = self.config.get('exp_name', None)
        if self.exp_name is None:
            self.exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.res_dir = osp.join(self.config['exp_dir'], self.exp_name, 'results')
        os.makedirs(self.res_dir, exist_ok=True)

        self.model = self._init_net()

        self.inference_loader = self._init_dataloader()

        pretrained_path = self.config.get('model_path', False)
        if pretrained_path:
            load_checkpoint(pretrained_path, self.model)
        else:
            raise Exception("model_path doesnt't exist in config. Please specify checkpoint path")

    def _init_net(self):
        raise NotImplemented

    def _init_dataloader(self):
        raise NotImplemented

    def inference(self):
        self.model.eval()

        gt_class = np.empty(0)
        pd_class = np.empty(0)

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.inference_loader)):
                inputs = batch['image'].to(self.config['device'])

                predictions = self.model(inputs)

                classes = predictions.topk(k=1)[1].view(-1).cpu().numpy()

                gt_class = np.concatenate((gt_class, batch['class'].numpy()))
                pd_class = np.concatenate((pd_class, classes))

        np.savetxt(osp.join(self.res_dir, "predictions.txt"), pd_class)

        class_accuracy = sum(pd_class == gt_class) / pd_class.shape[0]

        print('Validation CLASS accuracy - {:4f}'.format(class_accuracy))
