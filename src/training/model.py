"""Model builder: ResNet18 pretrained on ImageNet, fine-tuned for binary real/fake classification."""
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(freeze_backbone: bool = False) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
