import time
from torch.cuda.amp import autocast as autocast, GradScaler

from models.simCNN_contrastive import *
from models.ContrasiveLoss_SoftLabel import *
from evaluation import openset_eval_F1_contrastive, closedset_eval, openset_eval_contrastive_logitsnormed
from utils import load_cifar, load_ImageNet_crop, load_ImageNet_resize, get_smooth_labels, load_ImageNet200_versiontwo_testing, load_ImageNet200_versiontwo_testing_resize, load_ImageNet200_versiontwo_testing_crop ,load_LSUN_versiontwo_testing_resize, load_LSUN_versiontwo_testing_crop, load_noise_dataset
from mixup import *

from utils_other import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--id', default=5, type=int, help='random seed')
parser.add_argument('--model_suffix', default='main.pt', type=str,
                    help='Model suffix for loading encoder and classifier')
args = parser.parse_args()
model_suffix = args.model_suffix
splid_id = str(args.id) 


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classid_unknown = [0]

    percentile_min = 5
    percentile_max = 6 

    model_folder_path = './saved_models/'
    encoder = torch.load(model_folder_path + f'cifar10_encoder_{splid_id}{model_suffix}') 
    classifier = torch.load(model_folder_path + f'cifar10_classifierlinear_{splid_id}{model_suffix}') 
    encoder.to(device)
    classifier.to(device)
    classid_known = encoder.classid_list

    batch_size = 100 

    Data = CIFAR10_OSR_classifier(known=classid_known, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=32)
    validation_loader = Data.valloader

    Data = CIFAR10_OSR(known=classid_known, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=32)
    test_loader_known = Data.test_loader


    test_root = './data/tinyimagenet/val'
    test_loader_unknown, _ = load_ImageNet200_versiontwo_testing_resize([test_root], category_indexs=classid_unknown, train=False, batchSize=batch_size, shuffle=False)

    for percentile in range(percentile_min, percentile_max+1):
        with autocast():
            thresholds = classifier.estimate_threshold_logits(encoder, validation_loader,percentile=percentile)

        accuracy_overall, accuracy_known, accuracy_unknown, f1_score = openset_eval_F1_contrastive(encoder,
                                                                                                classifier,
                                                                                                test_loader_known,
                                                                                                test_loader_unknown)

        print('TIN Resize percentile = {:.0f} - known acc = {:.3f}%, unknown acc = {:.3f}%, all acc = {:.3f}%, F1 = {:.3f} '.format(
        percentile, accuracy_known * 100, accuracy_unknown * 100, accuracy_overall * 100, f1_score))

    test_root = './data/tinyimagenet/val'
    test_loader_unknown_crop, _ = load_ImageNet200_versiontwo_testing_crop([test_root], category_indexs=classid_unknown, train=False, batchSize=batch_size, shuffle=False)

    for percentile in range(percentile_min, percentile_max+1):
        with autocast():
            thresholds = classifier.estimate_threshold_logits(encoder, validation_loader,percentile=percentile)

        accuracy_overall, accuracy_known, accuracy_unknown, f1_score = openset_eval_F1_contrastive(encoder,
                                                                                                classifier,
                                                                                                test_loader_known,
                                                                                                test_loader_unknown_crop)
        print('TIN crop percentile = {:.0f} - known acc = {:.3f}%, unknown acc = {:.3f}%, all acc = {:.3f}%, F1 = {:.3f} '.format(
        percentile, accuracy_known * 100, accuracy_unknown * 100, accuracy_overall * 100, f1_score))

    test_root = './data/LSUN/test'
    test_loader_unknown, _ = load_LSUN_versiontwo_testing_resize([test_root], category_indexs=classid_unknown, train=False, batchSize=batch_size, shuffle=False)
    print("test classes:", classid_unknown)

    for percentile in range(percentile_min, percentile_max+1):
        with autocast():
            thresholds = classifier.estimate_threshold_logits(encoder, validation_loader,percentile=percentile)

        accuracy_overall, accuracy_known, accuracy_unknown, f1_score = openset_eval_F1_contrastive(encoder,
                                                                                                classifier,
                                                                                                test_loader_known,
                                                                                                test_loader_unknown)
        print('LSUNRESIZE percentile = {:.0f} - known acc = {:.3f}%, unknown acc = {:.3f}%, all acc = {:.3f}%, F1 = {:.3f} '.format(
        percentile, accuracy_known * 100, accuracy_unknown * 100, accuracy_overall * 100, f1_score))

    test_root = './data/LSUN/test'
    test_loader_unknown_crop, _ = load_LSUN_versiontwo_testing_crop([test_root], category_indexs=classid_unknown, train=False, batchSize=batch_size, shuffle=False)

    for percentile in range(percentile_min, percentile_max+1):
        with autocast():
            thresholds = classifier.estimate_threshold_logits(encoder, validation_loader,percentile=percentile)
        
        accuracy_overall, accuracy_known, accuracy_unknown, f1_score = openset_eval_F1_contrastive(encoder,
                                                                                                classifier,
                                                                                                test_loader_known,
                                                                                                test_loader_unknown_crop)
        print('LSUNCROP percentile = {:.0f} - known acc = {:.3f}%, unknown acc = {:.3f}%, all acc = {:.3f}%, F1 = {:.3f} '.format(
        percentile, accuracy_known * 100, accuracy_unknown * 100, accuracy_overall * 100, f1_score))

