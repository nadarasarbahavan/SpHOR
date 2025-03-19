import torch
import torch.nn as nn
import torch.nn.functional as F


def get_smooth_labels(labels, classid_list, smoothing_coeff=0.1):
    label_positive = 1 - smoothing_coeff
    label_negative = smoothing_coeff / (len(classid_list)-1)

    targets = label_negative * torch.ones([labels.shape[0], len(classid_list)], requires_grad=False).to(labels.device)
    for j, label in enumerate(labels):
        if label.item() not in classid_list:
            continue
        index = classid_list.index(label.item())
        targets[j, index] = label_positive
    return targets


class GvmFL(nn.Module):
    def __init__(self, classid_training, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, logit_feature_size=128):
        super(GvmFL, self).__init__()
        self.classid_training = classid_training
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.prototypes = nn.Parameter(torch.zeros(len(classid_training),logit_feature_size))
        self.__init_weight()
        self.proxy_labels = torch.tensor(classid_training).contiguous().view(-1, 1) 
        self.one_hot_proxy_labels = get_smooth_labels(self.proxy_labels, classid_training, 0) 
        self.device = None
        self.margin = None #0.1

    def __init_weight(self):
        nn.init.kaiming_normal_(self.prototypes)
    
    def init_class_prototypes(self, model, dataloader):
        model.eval()
        prototype_sums = torch.zeros_like(self.prototypes) 
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader):
                images = torch.cat([images[0], images[1]], dim=0)
                images, labels = images.cuda(), labels.cuda()
                features = model(images) 
                for j, feature in enumerate(features):
                    try:
                        index = self.classid_training.index(labels[j].item())
                    except:
                        continue
                    prototype_sums[index,:] += feature
            self.prototypes.data = F.normalize(prototype_sums, p=2, dim=1)


    def repulse(self):
        prototype_contrast_feature = F.normalize(self.prototypes, p=2, dim=1)
        label_mask = 1- torch.mm(self.one_hot_proxy_labels.to(self.device), torch.transpose(self.one_hot_proxy_labels, 0, 1).to(self.device))
        anchor_dot_contrast = torch.div(
            torch.matmul(prototype_contrast_feature, torch.transpose(prototype_contrast_feature, 0, 1)),
            self.temperature)
        pos = (label_mask * torch.exp(anchor_dot_contrast)).sum(1) / (label_mask.sum(1) + 1e-6) 
        log_prob = torch.log(pos)
        loss = (self.temperature / self.base_temperature) * log_prob.mean()
        return loss 

    
    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        self.device = device

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
            torch.matmul(anchor_feature, torch.transpose(prototype_contrast_feature, 0, 1))- self.margin * mask,
            self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
        neg = logits_max + torch.log(torch.sum(torch.exp(anchor_dot_contrast - logits_max), dim=1, keepdim=True) + 1e-12)
        log_prob = anchor_dot_contrast - neg

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6) 

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss 
    



class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

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
            mask = torch.mm(labels, torch.transpose(labels, 0, 1))
        else:
            mask = mask.float().to(device)

        mask.requires_grad=False
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))


        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask


        exp_logits = torch.exp(logits) * logits_mask  
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6) 

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

