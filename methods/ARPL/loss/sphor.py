

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from methods.ARPL.loss.LabelSmoothing import smooth_cross_entropy_loss


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):

    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1

    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist



class sphor(nn.Module):
    def __init__(self, **options):
        super(sphor, self).__init__()
        self.temp = 1 
        self.label_smoothing = options['label_smoothing']
        self.feat_dim =  options['feat_dim']
        self.no_cls =  options['no_cls']
        self.proj_dim = int(options['proj_dim']) 
        self.proj_layer = nn.Linear(self.feat_dim, self.proj_dim)
        self.base_loss = Sphor(self.no_cls, self.proj_dim, temperature=0.07)


    def forward(self, x, y, labels=None):
        
        if isinstance(x, list):
            indexes =  labels[3]
            lambdamix = labels[2]
            if indexes is not None: 
                logits = y[0]
                logits_mixup = y[1]
                if labels is None:
                    return logits, 0

                projections = self.proj_layer(x[0]).unsqueeze(1) 
                projections_mixup = self.proj_layer(x[1]).unsqueeze(1) 

                smoothed_labels = smooth_one_hot(true_labels=labels[0], smoothing=self.label_smoothing, classes=self.no_cls)
                smoothed_labels_mixup = smooth_one_hot(true_labels=labels[1], smoothing=self.label_smoothing, classes=self.no_cls)
    
                smoothed_labels_mixup = lambdamix * smoothed_labels + (1 - lambdamix) * smoothed_labels_mixup
                projections = F.normalize(projections, dim=-1)
                projections_mixup = F.normalize(projections_mixup, dim=-1)
                repl_loss = self.base_loss(projections.unsqueeze(1),smoothed_labels) + self.base_loss(projections_mixup.unsqueeze(1),smoothed_labels_mixup) #+
            else:
                logits = y[0]
                if labels is None:
                    return logits, 0

                projections = self.proj_layer(x[0]).unsqueeze(1) 
                smoothed_labels = smooth_one_hot(true_labels=labels[0], smoothing=self.label_smoothing, classes=self.no_cls)
                projections = F.normalize(projections, dim=-1)
                repl_loss = self.base_loss(projections.unsqueeze(1),smoothed_labels) 
        else:

            logits = y

            if labels is None:
                return logits, 0
        ce_loss = F.cross_entropy(logits / self.temp, labels[0]) 
        loss = 0.05*ce_loss + repl_loss  
        return projections, loss
    
class Sphor(nn.Module):
    def __init__(self, nocls, proj, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, proj_dim=128, **options):
        super(Sphor, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.no_cls = nocls
        self.logit_feature_size = proj

        self.prototypes = nn.Parameter(torch.zeros(self.no_cls,self.logit_feature_size))
        self.__init_weight()
        self.proxy_labels = torch.arange(0, self.no_cls) 
        self.one_hot_proxy_labels = smooth_one_hot(true_labels=self.proxy_labels, smoothing=0, classes=self.no_cls)
        self.eps = 1e-12
        self.device = None


    def __init_weight(self):
        nn.init.kaiming_normal_(self.prototypes)
    
    def repulse(self):
        prototype_contrast_feature = F.normalize(self.prototypes, p=2, dim=1)
        label_mask = 1- torch.mm(self.one_hot_proxy_labels.to(self.device), torch.transpose(self.one_hot_proxy_labels, 0, 1).to(self.device))
        
        anchor_dot_contrast = torch.matmul(prototype_contrast_feature, torch.transpose(prototype_contrast_feature, 0, 1))
        anchor_dot_contrast = anchor_dot_contrast**2
        anchor_dot_contrast= torch.div(anchor_dot_contrast, self.temperature)
        
        pos = (label_mask * torch.exp(anchor_dot_contrast)).sum(1) / (label_mask.sum(1) + 1e-6) # avoid nan when mask is 0. #This is the coefficent part. 
        log_prob = torch.log(pos)
        loss = (self.temperature / self.base_temperature) * log_prob.sum()
        return loss 
    

    
    def forward(self, features, labels=None,uncertainities=None, mask=None ):
        device = features.device
        self.device = features.device

        features = F.normalize(features, dim=-1)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            labels = F.normalize(labels, p=2, dim=1)
            mask = torch.mm(labels.to(device), torch.transpose(self.one_hot_proxy_labels, 0, 1).to(device))
        else:
            mask = mask.float().to(device)


        mask.requires_grad=False
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        prototype_contrast_feature = F.normalize(self.prototypes, p=2, dim=1)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        mask = mask.repeat(anchor_count, 1)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, torch.transpose(prototype_contrast_feature, 0, 1)),
            self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
        neg = logits_max + torch.log(torch.sum(torch.exp(anchor_dot_contrast - logits_max), dim=1, keepdim=True) + 1e-12)
        log_prob = anchor_dot_contrast - neg

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6) # avoid nan when mask is 0. #This is the coefficent part. 
        

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss  + 0.05*self.repulse()
    



