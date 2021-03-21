import json
import os
import os.path as osp
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import torch
import wfdb
from tqdm import tqdm

from utils.network_utils import load_checkpoint


class BasePipeline:
    def __init__(self, config):
        self.config = config
        self.exp_name = self.config.get("exp_name", None)
        if self.exp_name is None:
            self.exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.res_dir = osp.join(self.config["exp_dir"], self.exp_name, "results")
        os.makedirs(self.res_dir, exist_ok=True)

        self.model = self._init_net()

        self.pipeline_loader = self._init_dataloader()

        self.mapper = json.load(open(config["mapping_json"]))
        self.mapper = {j: i for i, j in self.mapper.items()}

        pretrained_path = self.config.get("model_path", False)
        if pretrained_path:
            load_checkpoint(pretrained_path, self.model)
        else:
            raise Exception(
                "model_path doesnt't exist in config. Please specify checkpoint path",
            )

    def _init_net(self):
        raise NotImplemented

    def _init_dataloader(self):
        raise NotImplemented

    def run_pipeline(self):
        self.model.eval()
        pd_class = np.empty(0)
        pd_peaks = np.empty(0)

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.pipeline_loader)):
                inputs = batch["image"].to(self.config["device"])

                predictions = self.model(inputs)

                classes = predictions.topk(k=1)[1].view(-1).cpu().numpy()

                pd_class = np.concatenate((pd_class, classes))
                pd_peaks = np.concatenate((pd_peaks, batch["peak"]))

        pd_class = pd_class.astype(int)
        pd_peaks = pd_peaks.astype(int)

        annotations = []
        for label, peak in zip(pd_class, pd_peaks):
            if (
                peak < len(self.pipeline_loader.dataset.signal)
                and self.mapper[label] != "N"
            ):
                annotations.append(
                    {
                        "x": peak,
                        "y": self.pipeline_loader.dataset.signal[peak],
                        "text": self.mapper[label],
                        "xref": "x",
                        "yref": "y",
                        "showarrow": True,
                        "arrowcolor": "black",
                        "arrowhead": 1,
                        "arrowsize": 2,
                    },
                )

        if osp.exists(self.config["ecg_data"] + ".atr"):
            ann = wfdb.rdann(self.config["ecg_data"], extension="atr")
            for label, peak in zip(ann.symbol, ann.sample):
                if peak < len(self.pipeline_loader.dataset.signal) and label != "N":
                    annotations.append(
                        {
                            "x": peak,
                            "y": self.pipeline_loader.dataset.signal[peak] - 0.1,
                            "text": label,
                            "xref": "x",
                            "yref": "y",
                            "showarrow": False,
                            "bordercolor": "#c7c7c7",
                            "borderwidth": 1,
                            "borderpad": 4,
                            "bgcolor": "#ffffff",
                            "opacity": 1,
                        },
                    )

        fig = go.Figure(
            data=go.Scatter(
                x=list(range(len(self.pipeline_loader.dataset.signal))),
                y=self.pipeline_loader.dataset.signal,
            ),
        )
        fig.update_layout(
            title="ECG",
            xaxis_title="Time",
            yaxis_title="ECG Output Value",
            title_x=0.5,
            annotations=annotations,
            autosize=True,
        )

        fig.write_html(
            osp.join(self.res_dir, osp.basename(self.config["ecg_data"] + ".html")),
        )
