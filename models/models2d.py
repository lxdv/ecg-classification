import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet as efficientnet


class HeartNet(nn.Module):
    def __init__(self, num_classes=7):
        super(HeartNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(64, eps=0.001),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(64, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(128, eps=0.001),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(128, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(256, eps=0.001),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(256, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 16 * 256, 2048),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2048, eps=0.001),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16 * 16 * 256)
        x = self.classifier(x)
        return x


class MobileNetV2(models.MobileNetV2):
    def __init__(self, num_classes=8):
        super().__init__(num_classes=num_classes)


class AlexNet(models.AlexNet):
    def __init__(self, num_classes=8):
        super().__init__(num_classes=num_classes)


def VGG16(num_classes=8):
    model = models.vgg16_bn()
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def ResNet18(num_classes=8):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def ResNet34(num_classes=8):
    model = models.resnet34()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def ShuffleNet(num_classes=8):
    model = models.shufflenet_v2_x1_0()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def EfficientNetB4(num_classes=8):
    model = efficientnet.from_name('efficientnet-b4')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model