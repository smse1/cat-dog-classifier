import torch.nn as nn
import torchvision.models as models


def get_model(num_classes: int = 2, use_pretrained: bool = True):
    if use_pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model