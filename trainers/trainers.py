from dataloaders.dataset1d import EcgDataset1D
from dataloaders.dataset2d import EcgDataset2D
from models import models2d, models1d
from trainers.base_trainer import BaseTrainer


class Trainer2D(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def _init_net(self):
        model = getattr(models2d, self.config['model'])(num_classes=self.config['num_classes'])
        model = model.to(self.config['device'])
        return model

    def _init_dataloaders(self):
        train_loader = EcgDataset2D(self.config['train_json'], self.config['mapping_json']).get_dataloader(
            batch_size=self.config['batch_size'], num_workers=self.config['num_workers']
        )
        val_loader = EcgDataset2D(self.config['val_json'], self.config['mapping_json']).get_dataloader(
            batch_size=self.config['batch_size'], num_workers=self.config['num_workers']
        )

        return train_loader, val_loader


class Trainer1D(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def _init_net(self):
        model = getattr(models1d, self.config['model'])(num_classes=self.config['num_classes'])
        model = model.to(self.config['device'])
        return model

    def _init_dataloaders(self):
        train_loader = EcgDataset1D(self.config['train_json'], self.config['mapping_json']).get_dataloader(
            batch_size=self.config['batch_size'], num_workers=self.config['num_workers']
        )
        val_loader = EcgDataset1D(self.config['val_json'], self.config['mapping_json']).get_dataloader(
            batch_size=self.config['batch_size'], num_workers=self.config['num_workers']
        )

        return train_loader, val_loader
