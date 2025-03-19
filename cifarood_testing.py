from evaluation import openset_eval_contrastive_logits, UMAP_plot, openset_eval_contrastive_odd, openset_eval_contrastive_logitsnormed
from utils import load_ImageNet200,load_ImageNet200_versiontwo, get_smooth_labels
from mixup import *
from utils_other import *

import matplotlib.pyplot as plt


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
    test_root = './data/tinyimagenet/train/'

    TOTAL_CLASS_NUM = 200
    classid_all = [i for i in range(0, TOTAL_CLASS_NUM)]

    batch_size = 128

    model_folder_path = './saved_models/'
    encoder = torch.load(model_folder_path + f'cifar10_encoder_{splid_id}{model_suffix}') 
    classifier = torch.load(model_folder_path + f'cifar10_classifierlinear_{splid_id}{model_suffix}')

    classid_known = encoder.classid_list
    classid_unknown = list(set(classid_all) - set(classid_known))




    Data = CIFAR10_OSR(known=classid_known, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=32)
    test_loader_known = Data.test_loader
    
    


    classid_all = [i for i in range(0, 100)]
    Data = CIFAR100_OSR(known=classid_all, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=32)
    test_loader_unknown_CIFAR100 = Data.test_loader

    #Accuracy, Accuracy_known, Accuracy_unknown, AUC, AUC_fnorm = openset_eval_contrastive_logitsnormed(encoder, classifier, test_loader_known, test_loader_unknown_CIFAR100)
    results = openset_eval_contrastive_odd(encoder, classifier, test_loader_known, test_loader_unknown_CIFAR100, name="./umaps/cifar100"+str(model_suffix))
    print (results)


    classid_all = [i for i in range(0, 100)]
    Data = SVHN_OSR(known=classid_all, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=32)
    test_loader_unknown_svhn = Data.test_loader

    results = openset_eval_contrastive_odd(encoder, classifier, test_loader_known, test_loader_unknown_svhn, name="./umaps/svhn"+str(model_suffix))
    print (results)

