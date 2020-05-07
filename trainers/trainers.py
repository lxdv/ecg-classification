import os.path as osp

import numpy as np
import torch
from torch import optim, nn
from tqdm import tqdm

from dataloaders.dataset2d import EcgDataset2D
from models.models_2d import HeartNet, MobileNetV2, AlexNet, VGG16bn, ResNet, ShuffleNet
from trainers.base_trainer import BaseTrainer
from utils.network_utils import save_checkpoint


class Trainer2D(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def _init_net(self):
        model = ShuffleNet(num_classes=self.config['num_classes'])
        model = model.to(self.device)
        return model

    def _init_dataloaders(self):
        train_loader = EcgDataset2D(self.train_json, self.mapping_json).get_dataloader(
            batch_size=self.config['batch_size'], num_workers=self.config['num_workers']
        )
        val_loader = EcgDataset2D(self.val_json, self.mapping_json).get_dataloader(
            batch_size=self.config['batch_size'], num_workers=self.config['num_workers']
        )

        return train_loader, val_loader

    def _init_optimizer(self):
        optimizer = getattr(optim, self.config['optim'])(self.model.parameters(), **self.config['optim_params'])
        return optimizer

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        gt_class = np.empty(0)
        pd_class = np.empty(0)

        for i, batch in enumerate(self.train_loader):
            inputs = batch['image'].to(self.device)
            targets = batch['class'].to(self.device)

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

            # if (self.total_iter + 1) % 1000 == 0:
            #     for g in self.optimizer.param_groups:
            #         lr = g['lr']
            #         g['lr'] *= 0.95
            #         print('LR changed from {} to {}'.format(lr, g['lr']))

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
                inputs = batch['image'].to(self.device)
                targets = batch['class'].to(self.device)

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
