import numpy as np

import torch
import torch.nn as nn

from torchvision import models
from torch.nn import functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ResNetClassifier(nn.Module):
    def __init__(self, n_class=2, pretrained=True, freeze_features=True):
        super(ResNetClassifier, self).__init__()
        # original_model = models.resnet34(pretrained=pretrained)
        original_model = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(original_model.children())[:-2],
                                      nn.AdaptiveAvgPool2d(1), Flatten())
        self.classifier = nn.Linear(original_model.fc.in_features, n_class)
        for p in self.features.parameters():
            p.requires_grad = freeze_features

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x):
        logit = self.forward(x)
        return F.softmax(logit, dim=1)


class VGGClassifier(nn.Module):
    def __init__(self, n_class=2, pretrained=True, freeze_features=True):
        super(VGGClassifier, self).__init__()
        original_model = models.vgg16(pretrained=pretrained)
        self.features = nn.Sequential(original_model.features, Flatten(),
                                      original_model.classifier[:-1])
        self.classifier = nn.Linear(original_model.classifier[-1].in_features,
                                    n_class)
        for p in self.features.parameters():
            p.requires_grad = freeze_features

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x):
        logit = self.forward(x)
        return F.softmax(logit, dim=1)


if __name__ == '__main__':
    # test
    res34 = ResNetClassifier()
    inputs = torch.rand(2, 3, 299, 299)
    outputs = res34(inputs)
    print(outputs.shape)

    vgg16 = VGGClassifier()
    inputs = torch.rand(2, 3, 224, 224)
    outputs = vgg16(inputs)
    print(outputs.shape)
