from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import (resnet18, resnet34, resnet50, resnet101, resnet152, mobilenet_v2)

def trim_network_at_index(network: nn.Module, index: int = -1) -> nn.Module:
    return nn.Sequential(*list(network.children())[:index])

def calculate_backbone_feature_dim(backbone: nn.Module, input_shape: Tuple[int, int, int]) -> int:
    tensor = torch.ones(1, *input_shape)
    with torch.no_grad():
        output_feat = backbone(tensor)
    return output_feat.shape[-1]

RESNET_VERSION_TO_MODEL = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152
}

class ResNetBackbone(nn.Module):
    def __init__(self, version: str):
        super().__init__()
        if version not in RESNET_VERSION_TO_MODEL:
            raise ValueError(f"Parameter version must be one of {list(RESNET_VERSION_TO_MODEL.keys())}. "
                             f"Received {version}.")
        self.backbone = RESNET_VERSION_TO_MODEL[version](pretrained=True)
        self.backbone = trim_network_at_index(self.backbone, -2)  # Trim the network to remove the classification head
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        backbone_features = self.backbone(input_tensor)
        return torch.flatten(backbone_features, start_dim=1)

class MobileNetBackbone(nn.Module):
    def __init__(self, version: str):
        super().__init__()
        if version != 'mobilenet_v2':
            raise NotImplementedError(f"Only mobilenet_v2 has been implemented, received {version}.")
        
        self.backbone = trim_network_at_index(mobilenet_v2(pretrained=True), -1)
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        backbone_features = self.backbone(input_tensor)
        return backbone_features.mean([2, 3])

# Example usage of calculate_backbone_feature_dim
input_shape = (3, 500, 500)
resnet_backbone = ResNetBackbone('resnet50')
feature_dim = calculate_backbone_feature_dim(resnet_backbone, input_shape)
print(f'Feature dimension for ResNet backbone: {feature_dim}')

mobilenet_backbone = MobileNetBackbone('mobilenet_v2')
feature_dim = calculate_backbone_feature_dim(mobilenet_backbone, input_shape)
print(f'Feature dimension for MobileNet backbone: {feature_dim}')
