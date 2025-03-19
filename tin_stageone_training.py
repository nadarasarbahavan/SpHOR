import time
import numpy as np
import torch
from torch.cuda.amp import autocast as autocast, GradScaler
from models.simCNN_contrastive import *
from models.ContrasiveLoss_SoftLabel import *

from configs.splits import splits_2020

from evaluation import openset_eval_contrastive_logits
from utils import load_ImageNet200, load_ImageNet200_contrastive,load_ImageNet200_contrastive_versiontwo, get_smooth_labels,load_ImageNet200_versiontwo, load_ImageNet200_versiontwo_testing
from mixup import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--id', default=2, type=int, help='random seed')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate for training')
parser.add_argument('--model_suffix', default='baseline.pt', type=str,
                    help='Model suffix for loading encoder and classifier')
parser.add_argument('--lr', default=0.5, type=float, help='Learning rate')
parser.add_argument('--label_smoothing', default=0.1, type=float, help='Label smoothing coefficient')
parser.add_argument('--dispweight', default=0.1, type=float, help='weight_class')
parser.add_argument('--dataset', type=str, choices=['mnist', 'tin'], required=True, help='Choose dataset: mnist or imagenet200')
parser.add_argument('--feature_dim', default=32, type=int, help='Feature dimension')
parser.add_argument('--disable_augmentations', action='store_true',
                    help='Disable augmentations if flag is set')

args = parser.parse_args()
lr = args.lr
label_smoothing_coeff = args.label_smoothing
feature_dim = args.feature_dim
dispweight = args.dispweight
model_suffix = args.model_suffix
splid_id = args.id 
learning_rate = lr 

for arg, value in vars(args).items():
    print(f"{arg}: {value}")
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_root = './data/tinyimagenet/train/'
    TOTAL_CLASS_NUM = 200
    classid_all = [i for i in range(0, TOTAL_CLASS_NUM)]



    batch_size = 128
    lr = learning_rate 
    num_contrastive_epochs = 1 #600 
    percentile = 5
    temperature = 0.1 
    label_smoothing_coeff = label_smoothing_coeff 
    

    model_folder_path = './saved_models/'

    

    if args.dataset == 'mnist':
        # Data = MNIST_OSR_Contrastive(known=classid_training, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=32, ifrandaug=False)
        # train_loader_contrastive = Data.train_loader
        # projector_head = 'nospeclinear'
        # classid_training = splits[splid_id] 
        # classid_training.sort()
        pass
    elif args.dataset == 'tin':
        classid_training = splits_2020['tiny_imagenet'][splid_id] 
        classid_training.sort()
        train_loader_contrastive, train_classes = load_ImageNet200_contrastive_versiontwo([train_root], category_indexs=classid_training, batchSize=batch_size, useRandAugment=True)
        projector_head = 'linear'
    # elif args.dataset == 'cif10':
    #     train_loader_contrastive, train_classes = load_ImageNet200_contrastive_versiontwo([train_root], category_indexs=classid_training, batchSize=batch_size, useRandAugment=True)
    #     projector_head = 'nospeclinear'      
    # elif args.dataset == 'cif10cifar100':
    #     train_loader_contrastive, train_classes = load_ImageNet200_contrastive_versiontwo([train_root], category_indexs=classid_training, batchSize=batch_size, useRandAugment=True)
    #     projector_head = 'nospeclinear'
    else:
        print ("Invalid")
    print("==> Traitning Class ", classid_training)



    best_epoch = -1
    best_auc = 0
    

    criterion = GvmFL(classid_training=classid_training ,temperature=temperature, base_temperature=temperature,logit_feature_size=feature_dim)
    criterion.margin = 0 
    criterion.to(device)


    feature_encoder = simCNN_contrastive(classid_list=classid_training, head=projector_head, logit_dim=feature_dim)
    feature_encoder.to(device)
    criterion.to(device)
    criterion.init_class_prototypes(feature_encoder,train_loader_contrastive)
    

    optimizer = torch.optim.SGD(feature_encoder.parameters(), lr = lr,
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=1e-4)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=5)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=lr * 1e-3)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])
    proto_optimizer = torch.optim.Adam([criterion.prototypes], lr=0.001)

    

    
    scaler = GradScaler()
    

    for epoch in range(1, num_contrastive_epochs+1):
        feature_encoder.train()
        time1 = time.time()
        total_loss = 0
        disp_loss = 0
        comp_loss = 0
        for i, (images, labels) in enumerate(train_loader_contrastive):
            targets = get_smooth_labels(labels, classid_training, label_smoothing_coeff)
            images_mixup, targets_mixup, targets_a, targets_b, lam = mixup_data_contrastive(images, targets, alpha=1,
                                                                                            use_cuda=False)

            images = torch.cat([images[0], images[1]], dim=0)
            images = images.to(device)
            targets = targets.to(device)

            images_mixup = torch.cat([images_mixup[0], images_mixup[1]], dim=0)
            images_mixup = images_mixup.to(device)
            targets_mixup = targets_mixup.to(device)

            bsz = targets.shape[0]

            optimizer.zero_grad()
            proto_optimizer.zero_grad()

            with autocast():
                logits = feature_encoder(images)
                logits1, logits2 = torch.split(logits, [bsz, bsz], dim=0)
                logits = torch.cat([logits1.unsqueeze(1), logits2.unsqueeze(1)], dim=1)

                logits_mixup = feature_encoder(images_mixup)
                logits3, logits4 = torch.split(logits_mixup, [bsz, bsz], dim=0)
                logits_mixup = torch.cat([logits3.unsqueeze(1), logits4.unsqueeze(1)], dim=1)

                logits_combine = torch.cat([logits, logits_mixup], dim=0)
                targets_combine = torch.cat([targets, targets_mixup], dim=0)
                
                temp_comp_loss =  criterion(logits_combine, targets_combine)
                temp_disp_loss = criterion.repulse()
                loss = dispweight*temp_disp_loss + temp_comp_loss
            


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(proto_optimizer)
            scaler.update()

            total_loss += loss
            disp_loss += temp_disp_loss
            comp_loss += temp_comp_loss


        print('epoch {}: contrastive_loss = {:.3f},  '.format(epoch, total_loss))
        print('comp {:.3f}: disp = {:.3f},  '.format(comp_loss, disp_loss))
        time2 = time.time()
        scheduler.step()
        print('time for this epoch: {:.3f} minutes'.format((time2 - time1) / 60.0))

        torch.save(feature_encoder, model_folder_path +f'tinyimagenet_encoder_{splid_id}{model_suffix}') 
