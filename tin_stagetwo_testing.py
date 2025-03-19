import time
import numpy as np
import torch
from torch.cuda.amp import autocast as autocast, GradScaler
from models.simCNN_contrastive import *
from evaluation import openset_eval_contrastive, openset_eval_contrastive_logits, openset_eval_contrastive_ALUN
from utils import load_ImageNet200,load_ImageNet200_versiontwo, get_smooth_labels, load_ImageNet200_versiontwo_testing
from mixup import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--id', default=2, type=int, help='random seed')
parser.add_argument('--model_suffix', default='main.pt', type=str,
                    help='Model suffix for loading encoder and classifier')
args = parser.parse_args()
splid_id = str(args.id)
model_suffix = args.model_suffix


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_root = './data/tinyimagenet/train/'
    model_folder_path = './saved_models/'

    TOTAL_CLASS_NUM = 200
    classid_all = [i for i in range(0, TOTAL_CLASS_NUM)]

    batch_size = 128
    num_classifier_epochs = 20
    percentile = 5
    temperature = 2 
    feature_dim = 128

    
    feature_encoder = torch.load(model_folder_path + f'tinyimagenet_encoder_{splid_id}{model_suffix}') 
    feature_encoder.to(device)
    classid_training = feature_encoder.classid_list
    classid_training.sort()
    feature_encoder.eval()
    print("==> Training Class: ", classid_training)

    try:
        classifier = torch.load(model_folder_path +f'tinyimagenet_classifierlinear_{splid_id}{model_suffix}')
    except:
        train_loader_classifier, train_classes = load_ImageNet200_versiontwo([train_root], category_indexs=classid_training, train=True, batchSize=batch_size, useRandAugment=False)
        validation_loader, train_classes = load_ImageNet200_versiontwo([train_root], category_indexs=classid_training, train=False, batchSize=batch_size, useRandAugment=False)

        best_epoch = -1
        best_auc = 0
        scaler = GradScaler()

        classifier = LinearClassifier(classid_list=classid_training, feature_dim=feature_dim)  
        classifier.to(device) 

        optimizer_classifier = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_classifier, T_max=num_classifier_epochs)



        for classifier_epoch in range(num_classifier_epochs):
            classifier.train()
            time3 = time.time()
            total_classification_loss = 0
            for i, (images_classifier, labels_classifier) in enumerate(train_loader_classifier):
                images_classifier = images_classifier.to(device)
                labels_np = labels_classifier.numpy()
                labels_classifier = labels_classifier.to(device)
                targets = get_smooth_labels(labels_classifier, classid_training, smoothing_coeff=0)
        
                optimizer_classifier.zero_grad()
                with autocast():
                    with torch.no_grad():
                        features = feature_encoder.get_feature(images_classifier)
                    logits = classifier(features)
                    classification_loss = classifier.get_loss(logits, targets,temperature=temperature)

                scaler.scale(classification_loss).backward()
                scaler.step(optimizer_classifier)
                scaler.update()
                total_classification_loss += classification_loss

            scheduler_classifier.step()
            print('classifier_epoch {}: classification_loss = {:.3f}'.format(classifier_epoch, total_classification_loss))
            time4 = time.time()
            print('time for this epoch: {:.3f} minutes'.format((time4 - time3) / 60.0))

        with autocast():
            thresholds = classifier.estimate_threshold_logits(feature_encoder, validation_loader,percentile=percentile)
            print(thresholds)

        torch.save(classifier, model_folder_path  +f'tinyimagenet_classifierlinear_{splid_id}{model_suffix}') 
    
    test_root = './data/tinyimagenet/val'
    classid_known = feature_encoder.classid_list
    classid_unknown = list(set(classid_all) - set(classid_known))

    test_loader_known, _ = load_ImageNet200_versiontwo_testing([test_root], category_indexs=classid_known, train=False, batchSize=batch_size, shuffle=False)
    test_loader_unknown, _ = load_ImageNet200_versiontwo_testing([test_root], category_indexs=classid_unknown, train=False, batchSize=batch_size, shuffle=False)

    _, _, _, AUROC = openset_eval_contrastive_logits(feature_encoder, classifier, test_loader_known, test_loader_unknown)
    alignment, uniformity = openset_eval_contrastive_ALUN(feature_encoder, classifier, test_loader_known, test_loader_unknown)
    

    print("==> Known Class: ", classid_known)
    print("==> Unknown Class: ", classid_unknown)

    print(' AUC = {:.3f}%'.format(AUROC * 100))
    print(f'Alignment: {alignment:.4f}, Uniformity: {uniformity:.4f}')
