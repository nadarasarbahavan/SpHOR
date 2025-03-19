import time
from torch.cuda.amp import autocast as autocast, GradScaler

from models.simCNN_contrastive import *
from models.ContrasiveLoss_SoftLabel import *
from evaluation import openset_eval_F1_contrastive,UMAP_plot
from utils import load_cifar, load_ImageNet_crop, load_ImageNet_resize, get_smooth_labels, load_ImageNet200_versiontwo_testing, load_ImageNet200_versiontwo_testing_resize, load_ImageNet200_versiontwo_testing_crop ,load_LSUN_versiontwo_testing_resize, load_LSUN_versiontwo_testing_crop, load_noise_dataset
from mixup import *

from utils_other import *
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, Lambda
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
from torchvision.datasets import Omniglot
from torch.utils.data import Subset

def load_noise_OmniDataset(batchSize, num_samples=10000, img_size=(3, 32, 32), shuffle=True):    
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        Lambda(lambda x: x.repeat(3, 1, 1))   
    ])
    test_dataset = Omniglot(root='./data', background=False, transform=transform, download=True)
    subset_indices = list(range(num_samples))
    subset_dataset = Subset(test_dataset, subset_indices)
    dataLoader = torch.utils.data.DataLoader(subset_dataset, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return dataLoader, None  

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
    percentile_min = 1
    percentile_max = 2


    model_folder_path = './saved_models/'
    encoder = torch.load(model_folder_path +f'mnist_encoder_{splid_id}{model_suffix}') 
    classifier = torch.load(model_folder_path +f'mnist_classifierlinear_{splid_id}{model_suffix}')  
        

    encoder.to(device)
    classifier.to(device)
    classid_known = encoder.classid_list


    batch_size = 12891

    Data = MNIST_OSR(known=classid_known, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=32)
    test_loader_known = Data.test_loader

    Data = MNIST_OSR_classifier(known=classid_known, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=32)
    train_loader_classifier = Data.train_loader
    validation_loader = Data.valloader

    test_loader_unknown_omni, _ = load_noise_OmniDataset(batchSize=batch_size, shuffle=False)

    MNISTNOISE_OSRtest_loader_unknown =  MNISTNOISE_OSR(known=classid_known, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=32)
    test_loader_unknown_mnistnoise = MNISTNOISE_OSRtest_loader_unknown.test_loader

    test_loader_unknown_noise, _ = load_noise_dataset(batchSize=batch_size, shuffle=False)


    for percentile in range(percentile_min, percentile_max+1):

        with autocast():
            thresholds = classifier.estimate_threshold_logits(encoder, validation_loader,percentile=percentile)
            print(thresholds)
    
        accuracy_overall, accuracy_known, accuracy_unknown, f1_score = openset_eval_F1_contrastive(encoder,
                                                                                                classifier,
                                                                                                test_loader_known,
                                                                                                test_loader_unknown_omni)
        print('OMNIGLOT percentile = {:.0f} - known acc = {:.3f}%, unknown acc = {:.3f}%, all acc = {:.3f}%, F1 = {:.3f} '.format(


        percentile, accuracy_known * 100, accuracy_unknown * 100, accuracy_overall * 100, f1_score))


        accuracy_overall, accuracy_known, accuracy_unknown, f1_score = openset_eval_F1_contrastive(encoder,
                                                                                                classifier,
                                                                                                test_loader_known,
                                                                                                test_loader_unknown_mnistnoise)
        print('Noisy MNIST percentile = {:.0f} - known acc = {:.3f}%, unknown acc = {:.3f}%, all acc = {:.3f}%, F1 = {:.3f} '.format(

            
        percentile, accuracy_known * 100, accuracy_unknown * 100, accuracy_overall * 100, f1_score))


        accuracy_overall, accuracy_known, accuracy_unknown, f1_score = openset_eval_F1_contrastive(encoder,
                                                                                                classifier,
                                                                                                test_loader_known,
                                                                                                test_loader_unknown_noise)
        print('NOISE percentile = {:.0f} - known acc = {:.3f}%, unknown acc = {:.3f}%, all acc = {:.3f}%, F1 = {:.3f} '.format(

            
        percentile, accuracy_known * 100, accuracy_unknown * 100, accuracy_overall * 100, f1_score))

    UMAP_plot(encoder, classifier, test_loader_known, test_loader_unknown_omni, model_name="./umaps/MNIST"+str(model_suffix)+str(id))
    
