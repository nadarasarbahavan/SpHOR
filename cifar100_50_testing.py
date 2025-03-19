from evaluation import openset_eval_contrastive_logits, UMAP_plot
from utils import load_ImageNet200,load_ImageNet200_versiontwo, get_smooth_labels
from mixup import *
from utils_other import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--id', default=2, type=int, help='random seed')
parser.add_argument('--model_suffix', default='_warmupstuff.pt', type=str,
                    help='Model suffix for loading encoder and classifier')
args = parser.parse_args()
model_suffix = args.model_suffix
splid_id = str(args.id) 

tehn_splits = [
        [30, 25, 1, 9, 8, 0, 46, 52, 49, 71],
        [41, 9, 49, 40, 73, 60, 48, 30, 95, 71],
        [8, 9, 49, 40, 73, 60, 48, 95, 30, 71],
        [95, 60, 30, 73, 46, 49, 68, 99, 8, 71],
        [33, 2, 3, 97, 46, 21, 64, 63, 88, 43]
        ]

fifty_splits = [[27, 94, 29, 77, 88, 26, 69, 48, 75, 5, 59, 93, 39, 57, 45, 40, 78, 20, 98, 47, 66, 70, 91, 76, 41, 83, 99, 32, 53, 72, 2, 95, 21, 73, 84, 68, 35, 11, 55, 60, 30, 25, 1, 9, 8, 0, 46, 52, 49, 71],
        [65, 97, 86, 24, 45, 67, 2, 3, 91, 98, 79, 29, 62, 82, 33, 76, 0, 35, 5, 16, 54, 11, 99, 52, 85, 1, 25, 66, 28, 84, 23, 56, 75, 46, 21, 72, 55, 68, 8, 69, 41, 9, 49, 40, 73, 60, 48, 30, 95, 71],
        [20, 83, 65, 97, 94, 2, 93, 16, 67, 29, 62, 33, 24, 98, 5, 86, 35, 54, 0, 91, 52, 66, 85, 84, 56, 11, 1, 76, 25, 55, 21, 99, 72, 41, 23, 75, 28, 68, 69, 46, 8, 9, 49, 40, 73, 60, 48, 95, 30, 71],
        [92, 82, 77, 64, 5, 33, 62, 56, 70, 0, 20, 28, 67, 14, 84, 53, 91, 29, 85, 2, 52, 83, 75, 35, 11, 21, 72, 98, 55, 1, 41, 76, 25, 66, 69, 9, 48, 54, 40, 23, 95, 60, 30, 73, 46, 49, 68, 99, 8, 71],
        [47, 6, 19, 0, 62, 93, 59, 65, 54, 70, 34, 55, 23, 38, 72, 76, 53, 31, 78, 96, 77, 27, 92, 18, 82, 50, 98, 32, 1, 75, 83, 4, 51, 35, 80, 11, 74, 66, 36, 42, 33, 2, 3, 97, 46, 21, 64, 63, 88, 43]]



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_root = './data/tinyimagenet/train/'

    TOTAL_CLASS_NUM = 200
    classid_all = [i for i in range(0, TOTAL_CLASS_NUM)]

    batch_size = 128

    model_folder_path = './saved_models/'
    encoder = torch.load(model_folder_path + f'cifar100_50_encoder_{splid_id}{model_suffix}')
    classifier = torch.load(model_folder_path + f'cifar100_50_realclassifierlinearh_{splid_id}{model_suffix}')
    
    classid_known = encoder.classid_list
    classid_unknown = list(set(classid_all) - set(classid_known))

    print ("for the cifar +10")

    Data = CIFAR10_OSR(known=classid_known, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=32)
    test_loader_known = Data.test_loader
    
    classid_all = [i for i in range(0, 100)]
    Data = CIFAR100_OSR(known=tehn_splits[int(splid_id)], dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=32)
    test_loader_unknown_ten = Data.test_loader

    
    _, _, _, AUROC = openset_eval_contrastive_logits(encoder, classifier, test_loader_known, test_loader_unknown_ten, name=args.model_suffix+str("normal"))

    print("==> Known Class: ", classid_known)
    print("==> Unknown Class: ", tehn_splits[int(splid_id)])
    print('unknown detection AUC = {:.3f}%'.format(AUROC * 100))

    print ("for the cifar +50")

    
    classid_all = [i for i in range(0, 100)]
    Data = CIFAR100_OSR(known=fifty_splits[int(splid_id)], dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=32)
    test_loader_unknown_fifty = Data.test_loader

    
    _, _, _, AUROC = openset_eval_contrastive_logits(encoder, classifier, test_loader_known, test_loader_unknown_fifty,name=args.model_suffix+str("normuncertain"))
        
    print("==> Known Class: ", classid_known)
    print("==> Unknown Class: ", fifty_splits[int(splid_id)])
    print('unknown detection AUC = {:.3f}%'.format(AUROC * 100))
