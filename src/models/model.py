import torch.nn as nn
from src.models import *
import torch


class Model(nn.Module):
    available_models = ["lenet"]
    keytypes = {"type": str, "classname": str, "pretrained": bool, "feature_extract": bool}

    def __init__(self, model_info, num_classes: int) -> None:
        super().__init__()
        Model.verify_info(model_info)
        self._model_info = model_info
        self._num_classes = num_classes
        self._model = Model.get_model(model_info, num_classes)

    @property
    def model_info(self):
        return self.model_info

    @property
    def num_classes(self):
        return self._num_classes

    def same_setup(self, other):
        return self.model_info == other.model_info and self.num_classes == other.num_classes

    def __str__(self):
        return str(self.model_info)

    @staticmethod
    def verify_info(model_info):
        for k, t in Model.keytypes.items():
            assert k in model_info, f"Missing key {k} in model_info"
            assert type(model_info[k]) == t, f"Invalid type for key {k}: expected {t}, given {type(model_info[k])}"

    @staticmethod
    def get_model(model_info, num_classes: int):
        if model_info.type in Model.available_models:
            if model_info.pretrained:
                raise ValueError(f"No pretrained model available for {model_info.type}")
            else:
                m = eval(model_info.classname)(num_classes)
        else:
            m = Model.download(model_info)
        Model.set_parameter_requires_grad(m, model_info.feature_extract)
        if model_info.type == "mobilenetv2":
            m.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(m.last_channel, num_classes),
            )
        return m

    @staticmethod
    def download(model_info):
        import torchvision.models as models
        model_methods = {"mobilenetv2": models.mobilenet_v2}
        assert model_info.type in model_methods, "Unknown model to download"
        return model_methods[model_info.type](pretrained=model_info.pretrained)

    @staticmethod
    def set_parameter_requires_grad(target_model, feature_extracting):
        if feature_extracting:
            for param in target_model.parameters():
                param.requires_grad = False

    def __clipping(self, c):
        total_sum = torch.zeros(1, 1)
        for param in self._model.parameters():
            total_sum += torch.sum(torch.pow(param, 2))
        weight_norm = torch.sqrt(total_sum)
        scale_factor = max(1, weight_norm / c)
        for param in self._model.parameters():
            param.__additional_data.div_(scale_factor)

    def add_gaussian_noise(self, sigma2, c):
        self.__clipping(c)
        for param in self._model.parameters():
            noise = torch.normal(mean=0, std=torch.ones(param.size()) * sigma2)
            noise = noise.to(param.__device)
            param.__additional_data.add_(noise)

    def forward(self, x):
        return self._model.forward(x)

    def has_batchnorm(self):
        for m in self._model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                return True
        return False
