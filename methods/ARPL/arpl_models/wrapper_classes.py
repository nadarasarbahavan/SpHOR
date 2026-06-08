import torch
from torch import nn
from models.model_utils import transform_moco_state_dict

from methods.ARPL.arpl_models.resnetABN import resnet50ABN
from methods.ARPL.arpl_models.ABN import MultiBatchNorm
import timm
from config import places_supervised_path
import torchvision.models as models
import torch.nn as nn
class TimmResNetWrapper(nn.Module):

    def __init__(self, num_classes=100):

        super().__init__()
        self.location= places_supervised_path
        checkpoint = torch.load(self.location, map_location="cpu")
        print("Checkpoint keys:", checkpoint.keys())
        self.resnet = models.resnet50(num_classes=365) 
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") 
            new_state_dict[new_key] = v
        
        missing, unexpected = self.resnet.load_state_dict(new_state_dict, strict=True)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        self.in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        self.fc = nn.Linear(self.in_features, num_classes)

        self.feat_dim = self.in_features

    def forward(self, x, return_features=True, dummy_label=None):
        embedding = self.resnet(x)
        embedding = torch.flatten(embedding, 1)  
        preds = self.fc(embedding.detach())

        if return_features:
            return embedding, preds
        else:
            return preds


class TimmResNet50Detached(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet
    def forward(self, x, return_features=True, dummy_label=None):

        x = self.resnet.forward_features(x)
        embedding = self.resnet.global_pool(x)
        if self.resnet.drop_rate:
            embedding = torch.nn.functional.dropout(embedding, p=float(self.drop_rate), training=self.training)
        preds = self.resnet.fc(embedding.detach())
        if return_features:
            return embedding, preds
        else:
            return preds

class TimmResNet50(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet
    def forward(self, x, return_features=True, dummy_label=None):

        x = self.resnet.forward_features(x)
        embedding = self.resnet.global_pool(x)
        if self.resnet.drop_rate:
            embedding = torch.nn.functional.dropout(embedding, p=float(self.drop_rate), training=self.training)
        preds = self.resnet.fc(embedding.detach())

        if return_features:
            return embedding, preds
        else:
            return preds