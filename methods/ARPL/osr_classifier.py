import os
import argparse
import datetime
import time
import pandas as pd
import importlib

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn

from methods.ARPL.arpl_models import gan
from methods.ARPL.arpl_models.arpl_models import classifier32ABN
from methods.ARPL.arpl_models.wrapper_classes import TimmResNetWrapper, TimmResNet50Detached, TimmResNet50
from methods.ARPL.arpl_utils import save_networks
from methods.ARPL.core import train,trainmixup, train_cs, test, FeatureTrainer ,scoretester

from utils.utils import init_experiment, seed_torch, str2bool, get_default_hyperparameters
from utils.schedulers import get_scheduler
from data.open_set_datasets import get_class_splits, get_datasets
from models.model_utils import get_model

from config import exp_root, save_dir

from utils.utils import strip_state_dict
from methods.ARPL.test_utils import EvaluateOpenSet , ModelTemplate

import timm




import pickle
class EnsembleModelEntropy(ModelTemplate):

    def __init__(self, all_models, mode='entropy', num_classes=4, use_softmax=False):

        super(ModelTemplate, self).__init__()

        self.all_models = all_models
        self.max_ent = torch.log(torch.Tensor([num_classes])).item()
        self.mode = mode
        self.use_softmax = use_softmax

    def entropy(self, preds):

        logp = torch.log(preds + 1e-5)
        entropy = torch.sum(-preds * logp, dim=-1)

        return entropy

    def forward(self, imgs):

        all_closed_set_preds = []

        for m in self.all_models:

            closed_set_preds = m(imgs, return_features=False)

            if self.use_softmax:
                closed_set_preds = torch.nn.Softmax(dim=-1)(closed_set_preds)

            all_closed_set_preds.append(closed_set_preds)

        closed_set_preds = torch.stack(all_closed_set_preds).mean(dim=0)

        if self.mode == 'entropy':
            open_set_preds = self.entropy(closed_set_preds)
        elif self.mode == 'max_softmax':
            open_set_preds = -closed_set_preds.max(dim=-1)[0]

        else:
            raise NotImplementedError

        return closed_set_preds, open_set_preds

class EnsembleModelScore(ModelTemplate):

    def __init__(self, all_models, score_function, mode='entropy', num_classes=4, use_softmax=False):

        super(ModelTemplate, self).__init__()

        self.all_models = all_models
        self.max_ent = torch.log(torch.Tensor([num_classes])).item()
        self.mode = mode
        self.use_softmax = use_softmax
        self.scoringfunction = score_function


    def forward(self, imgs):

        all_closed_set_preds = []
        all_open_set_preds = []

        for m in self.all_models:

            features, closed_set_preds = m(imgs, return_features=True)
            openset_scores = self.scoringfunction.infer(features.cpu(), closed_set_preds.cpu())

            if self.use_softmax:
                closed_set_preds = torch.nn.Softmax(dim=-1)(closed_set_preds)

            all_closed_set_preds.append(closed_set_preds)
            all_open_set_preds.append(-openset_scores)

        closed_set_preds = torch.stack(all_closed_set_preds).mean(dim=0)
        open_set_preds = torch.stack(all_open_set_preds).mean(dim=0)


        return closed_set_preds, open_set_preds
    

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='cub', help="")
parser.add_argument('--out-num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--image_size', type=int, default=64)

# optimization
parser.add_argument('--optim', type=str, default=None, help="Which optimizer to use {adam, sgd}")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate for model")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="LR regularisation on weights")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--max-epoch', type=int, default=10)
parser.add_argument('--max_epoch_stageone', type=int, default=10)
parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts')
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num_restarts', type=int, default=2, help='How many restarts for cosine_warm_restarts schedule')
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing."
                                                                        "No smoothing if None or 0")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='classifier32')
parser.add_argument('--resnet50_pretrain', type=str, default='places_moco',
                        help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
parser.add_argument('--feat_dim', type=int, default=128, help="Feature vector dim, only for classifier32 at the moment")

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=None)
parser.add_argument('--notpretrained', action='store_false', help="if used, it will not use the tIMM imagenet reptrained network")

# misc
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--split_train_val', default=False, type=str2bool,
                        help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--use_default_parameters', default=False, type=str2bool,
                    help='Set to True to use optimized hyper-parameters from paper', metavar='BOOL')
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')
parser.add_argument('--gpus', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--checkpt_freq', type=int, default=20)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
parser.add_argument('--train_feat_extractor', default=True, type=str2bool,
                        help='Train feature extractor (only implemented for renset_50_faces)', metavar='BOOL')
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--use_softmax_in_eval', default=False, type=str2bool,
                        help='Do we use softmax or logits for evaluation', metavar='BOOL')
#supcon
parser.add_argument('--proj_dim', type=int, default=128)
parser.add_argument('--exp_id', type=str, default='(19.05.2021_|_30.963)')
from config import save_dir, osr_split_dir, root_model_path, root_criterion_path
import glob
def get_optimizer(args, params_list):

    if args.optim is None:

        if options['dataset'] == 'tinyimagenet':
            optimizer = torch.optim.Adam(params_list, lr=args.lr)
        else:
            optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    elif args.optim == 'sgd':

        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    elif args.optim == 'adam':

        optimizer = torch.optim.Adam(params_list, lr=args.lr)

    else:

        raise NotImplementedError

    return optimizer


def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# TODO: Args and options are largely duplicates: tidy up
def main_worker(options, args):

    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in options['gpus']) #options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpus']))
        cudnn.benchmark = False
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # -----------------------------
    # DATALOADERS
    # -----------------------------
    trainloader = dataloaders['train']
    trainloader_training_without_aug = dataloaders['datasets_training_without_aug']
    testloader = dataloaders['val']
    outloader = dataloaders['test_unknown']

    # Define experiment IDs
    exp_ids = [
        args.exp_id,
    ]
    all_paths_combined = []
    for i in exp_ids:
        # Format the templates first
        model_pattern = root_model_path.format(i, args.dataset, args.dataset, str(int(args.max_epoch_stageone)-1), "*")
        criterion_pattern = root_criterion_path.format(i, args.dataset, args.dataset, str(int(args.max_epoch_stageone)-1), "*")
        
        # Run glob to find the actual files
        model_files = glob.glob(model_pattern)
        criterion_files = glob.glob(criterion_pattern)
        
        # Print them clearly for debugging
        print(f"--- Experiment ID: {i} ---")
        print(f"Model files found: {model_files}")
        print(f"Criterion files found: {criterion_files}")
        
        all_paths_combined.append([model_files, criterion_files])
    # -----------------------------
    # MODEL
    # -----------------------------
    print("Creating model: {}".format(options['model']))
    if options['cs'] and args.loss == 'ARPLoss':
        if args.model == 'classifier32':
            net = classifier32ABN(num_classes=len(args.train_classes), feat_dim=args.feat_dim)
        else:
            raise NotImplementedError

    else:
        if args.model == 'timm_resnet50':
            if args.notpretrained:
                print ("Not pretrained","Classifier not detached")
                model = timm.create_model('resnet50', num_classes=len(args.train_classes),pretrained=False)
                net = TimmResNet50(model)
            else:
                print ("Pretrained","Classifier not detached")
                model = timm.create_model('resnet50', num_classes=len(args.train_classes),pretrained=True)
                net = TimmResNet50(model)
        elif args.model == 'timm_resnet50_repl':
            if args.notpretrained:
                print ("Not pretrained","Classifier detached")
                model = timm.create_model('resnet50', num_classes=len(args.train_classes),pretrained=False)
                net = TimmResNet50Detached(model)
            else:
                print ("Pretrained","Classifier detached")
                model = timm.create_model('resnet50', num_classes=len(args.train_classes),pretrained=True)
                net = TimmResNet50Detached(model)

        elif args.model == 'classifier32':
            net = classifier32ABN(num_classes=len(args.train_classes), feat_dim=args.feat_dim)
        else:
            wrapper_class = None
    
    # try:
    #     state_dict = strip_state_dict(torch.load(all_paths_combined[0][0][0]))
    #     state_dict_proj = strip_state_dict(torch.load(all_paths_combined[0][0][1]))
    # except Exception as e1:
    #     try:
    #         state_dict = strip_state_dict(torch.load(all_paths_combined[0][0][1]))
    #         state_dict_proj = strip_state_dict(torch.load(all_paths_combined[0][0][0]))
    #     except Exception as e2:
    #         raise RuntimeError(f"Both paths failed:\n1: {e1}\n2: {e2}")
    

    # try:
    #     net.load_state_dict(state_dict)
    # except Exception as e1:
    #     net.load_state_dict(state_dict_proj)



    feat_dim = args.feat_dim



    # Loss
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu':  use_gpu
        }
    )


    options['loss'] = 'Softmax'



    if use_gpu:
        net = nn.DataParallel(net, device_ids=options['gpus']).cuda()


    model_path = os.path.join(args.log_dir, 'arpl_models', options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    params_list = [{'params': net.parameters()}]
    
    # Get base network and criterion
    optimizer = get_optimizer(args=args, params_list=params_list)
    options['loss'] = 'Softmax'
    Loss = importlib.import_module('methods.ARPL.loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)


    # -----------------------------
    # GET SCHEDULER
    # ----------------------------
    scheduler = get_scheduler(optimizer, args)

    start_time = time.time()

    # -----------------------------
    # TRAIN
    # -----------------------------

    options.update(
    {
        'temp': 1,
        'label_smoothing':  0
    }
    )

    osr_path = os.path.join(osr_split_dir, '{}_osr_splits.pkl'.format(args.dataset))

    with open(osr_path, 'rb') as f:
        class_info = pickle.load(f)

    train_classes = class_info['known_classes']
    open_set_classes = class_info['unknown_classes']

    FT= FeatureTrainer(net, criterion, optimizer, trainloader)
    FT.extract_features()
    for epoch in range(options['max_epoch']):



        FT.train_classifier_epoch(epoch=epoch,args=args)
        # -----------------------------
        # STEP SCHEDULER
        # ----------------------------
        if args.scheduler == 'plateau' or args.scheduler == 'warm_restarts_plateau':
            scheduler.step(results['ACC'], epoch)
        elif args.scheduler == 'multi_step':
            scheduler.step()
        else:
            scheduler.step(epoch=epoch)


    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    score_functions = [ "knn5","msp"] 
    for score_func in score_functions:
        _, scoring_function = scoretester(net, criterion, trainloader_training_without_aug, testloader, outloader, epoch=None, score_func=score_func, **options)


                                                                                            
        with torch.no_grad():


            for difficulty in ('Easy', 'Hard'):

                # ------------------------
                # DATASETS
                # ------------------------
                args.train_classes, args.open_set_classes = train_classes, open_set_classes[difficulty]

                if difficulty == 'Hard' and args.dataset != 'imagenet':
                    args.open_set_classes += open_set_classes['Medium']

                datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                        image_size=args.image_size, balance_open_set_eval=False,
                                        split_train_val=False, open_set_classes=args.open_set_classes)

                # ------------------------
                # DATALOADERS
                # ------------------------
                dataloaderstest = {}
                for k, v, in datasets.items():
                    shuffle = True if k == 'train' else False
                    dataloaderstest[k] = DataLoader(v, batch_size=args.batch_size,
                                                shuffle=shuffle, sampler=None, num_workers=args.num_workers)

                # ------------------------
                # MODEL
                # ------------------------
                print('Running Score Function: ',score_func, "Data Split: ",difficulty)

                model = EnsembleModelScore(all_models=[net], mode="max_softmax", num_classes=len(args.train_classes),score_function=scoring_function)
                
                # ------------------------
                # EVALUATE
                # ------------------------
                evaluate = EvaluateOpenSet(model=model, known_data_loader=dataloaderstest['test_known'],
                                        unknown_data_loader=dataloaderstest['test_unknown'], device=next(net.parameters()).device, save_dir=save_dir)

                # Make predictions on test sets
                evaluate.predict()

                preds = evaluate.evaluate(evaluate, normalised_ap=False)


    return results

if __name__ == '__main__':

    args = parser.parse_args()
    
    # ------------------------
    # Update parameters with default hyperparameters if specified
    # ------------------------
    if args.use_default_parameters:
        print('NOTE: Using default hyper-parameters...')
        args = get_default_hyperparameters(args)

    args.exp_root = exp_root
    args.epochs = args.max_epoch
    img_size = args.image_size
    results = dict()

    for i in range(1):

        # ------------------------
        # INIT
        # ------------------------
        if args.feat_dim is None:
            args.feat_dim = 128 if args.model == 'classifier32' else 2048

        args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx,
                                                                     cifar_plus_n=args.out_num)

        img_size = args.image_size

        args.save_name = '{}_{}_{}'.format(args.model, args.seed, args.dataset)
        runner_name = os.path.dirname(__file__).split("/")[-2:]
        args = init_experiment(args, runner_name=runner_name)

        # ------------------------
        # SEED
        # ------------------------
        seed_torch(args.seed)

        # ------------------------
        # DATASETS
        # ------------------------
        datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes, balance_open_set_eval=True,
                                split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                                args=args)

        datasets_training_without_aug = get_datasets(args.dataset, transform="pure", train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes, balance_open_set_eval=True,
                                split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                                args=args)

        # ------------------------
        # RANDAUG HYPERPARAM SWEEP
        # ------------------------
        if args.transform == 'rand-augment':
            if args.rand_aug_m is not None:
                if args.rand_aug_n is not None:
                    datasets['train'].transform.transforms[0].m = args.rand_aug_m
                    datasets['train'].transform.transforms[0].n = args.rand_aug_n

        # ------------------------
        # DATALOADER
        # ------------------------
        dataloaders = {}
        for k, v, in datasets.items():
            shuffle = True if k == 'train' else False
            dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                        shuffle=shuffle, sampler=None, num_workers=args.num_workers)
        dataloaders["datasets_training_without_aug"] = DataLoader(datasets_training_without_aug["test_known"], batch_size=args.batch_size,
                                        shuffle=shuffle, sampler=None, num_workers=args.num_workers)
        # ------------------------
        # SAVE PARAMS
        # ------------------------
        options = vars(args)
        options.update(
            {
                'item':     i,
                'known':    args.train_classes,
                'unknown':  args.open_set_classes,
                'img_size': img_size,
                'dataloaders': dataloaders,
                'num_classes': len(args.train_classes)
            }
        )

        dir_name = '{}_{}'.format(options['model'], options['loss'])
        dir_path = os.path.join('/'.join(args.log_dir.split("/")[:-2]), 'results', dir_name)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if options['dataset'] == 'cifar-10-100':
            file_name = '{}_{}.csv'.format(options['dataset'], options['out_num'])
            if options['cs']:
                file_name = '{}_{}_cs.csv'.format(options['dataset'], options['out_num'])
        else:
            file_name = options['dataset'] + '.csv'
            if options['cs']:
                file_name = options['dataset'] + 'cs' + '.csv'

        print('result path:', os.path.join(dir_path, file_name))
        # ------------------------
        # TRAIN
        # ------------------------
        res = main_worker(options, args)

        # ------------------------
        # LOG
        # ------------------------
        res['split_idx'] = args.split_idx
        res['unknown'] = args.open_set_classes
        res['known'] = args.train_classes
        res['ID'] = args.log_dir.split("/")[-1]
        results[str(args.split_idx)] = res
