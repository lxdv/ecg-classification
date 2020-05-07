import os
import os.path as osp
from datetime import datetime

import numpy as np
import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.network_utils import load_checkpoint, save_checkpoint


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
        self.criterion = nn.CrossEntropyLoss().to(self.config['device'])

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

    def _init_dataloaders(self):
        raise NotImplemented

    def _init_optimizer(self):
        optimizer = getattr(optim, self.config['optim'])(self.model.parameters(), **self.config['optim_params'])
        return optimizer

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        gt_class = np.empty(0)
        pd_class = np.empty(0)

        for i, batch in enumerate(self.train_loader):
            inputs = batch['image'].to(self.config['device'])
            targets = batch['class'].to(self.config['device'])

            predictions = self.model(inputs)
            loss = self.criterion(predictions, targets)

            classes = predictions.topk(k=1)[1].view(-1).cpu().numpy()

            gt_class = np.concatenate((gt_class, batch['class'].numpy()))
            pd_class = np.concatenate((pd_class, classes))

            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % 10 == 0:
                print("\tIter [%d/%d] Loss: %.4f" % (i + 1, len(self.train_loader), loss.item()))

            self.writer.add_scalar("Train loss (iterations)", loss.item(), self.total_iter)
            self.total_iter += 1

        total_loss /= len(self.train_loader)
        class_accuracy = sum(pd_class == gt_class) / pd_class.shape[0]

        print('Train loss - {:4f}'.format(total_loss))
        print('Train CLASS accuracy - {:4f}'.format(class_accuracy))

        self.writer.add_scalar('Train loss (epochs)', total_loss, self.training_epoch)
        self.writer.add_scalar('Train CLASS accuracy', class_accuracy, self.training_epoch)

    def val(self):
        self.model.eval()
        total_loss = 0

        gt_class = np.empty(0)
        pd_class = np.empty(0)

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.val_loader)):
                inputs = batch['image'].to(self.config['device'])
                targets = batch['class'].to(self.config['device'])

                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)

                classes = predictions.topk(k=1)[1].view(-1).cpu().numpy()

                gt_class = np.concatenate((gt_class, batch['class'].numpy()))
                pd_class = np.concatenate((pd_class, classes))

                total_loss += loss.item()

        total_loss /= len(self.val_loader)
        class_accuracy = sum(pd_class == gt_class) / pd_class.shape[0]

        print('Validation loss - {:4f}'.format(total_loss))
        print('Validation CLASS accuracy - {:4f}'.format(class_accuracy))

        self.writer.add_scalar('Validation loss', total_loss, self.training_epoch)
        self.writer.add_scalar('Validation CLASS accuracy', class_accuracy, self.training_epoch)

    def loop(self):
        for epoch in range(self.training_epoch + 1, self.epochs + 1):
            print("Epoch - {}".format(self.training_epoch))
            self.train_epoch()
            save_checkpoint({
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch,
                'total_iter': self.total_iter
            }, osp.join(self.pth_dir, '{:0>8}.pth'.format(epoch)))
            self.val()

            self.training_epoch += 1
