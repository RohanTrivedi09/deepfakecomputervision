"""ResNet18 architecture builder — mirrors notebooks/deepfake_training.ipynb's
build_model() so a saved state_dict from that notebook loads back in cleanly.
"""
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 2) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
