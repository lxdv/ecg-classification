from dataloaders.dataset1d import EcgPipelineDataset1D
from models import models1d
from pipelines.base_pipeline import BasePipeline


class Pipeline1D(BasePipeline):
    def __init__(self, config):
        super().__init__(config)

    def _init_net(self):
        model = getattr(models1d, self.config["model"])(
            num_classes=self.config["num_classes"],
        )
        model = model.to(self.config["device"])
        return model

    def _init_dataloader(self):
        inference_loader = EcgPipelineDataset1D(self.config["ecg_data"]).get_dataloader(
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
        )

        return inference_loader
