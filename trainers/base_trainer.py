import os
import os.path as osp
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from utils.network_utils import load_checkpoint


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.exp_name = self.config.get('exp_name', None)
        if self.exp_name is None:
            self.exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.log_dir = osp.join(self.config['exp_dir'], self.exp_name, 'logs')
        self.pth_dir = osp.join(self.config['exp_dir'], self.exp_name, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.pth_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.model = self._init_net()
        self.optimizer = self._init_optimizer()

        self.train_loader, self.val_loader = self._init_dataloaders()

        pretrained_path = self.config.get('model_path', False)
        if pretrained_path:
            self.training_epoch, self.total_iter = load_checkpoint(pretrained_path, self.model,
                                                                   optimizer=self.optimizer)

        else:
            self.training_epoch = 0
            self.total_iter = 0

        self.epochs = self.config.get('epochs', int(1e5))

    def _init_net(self):
        raise NotImplemented

    def _init_optimizer(self):
        raise NotImplemented

    def _init_dataloaders(self):
        raise NotImplemented
