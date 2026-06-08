import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm
from methods.ARPL.core import evaluation
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score

def test(net, criterion, testloader, outloader, epoch=None, **options):

    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels, idx in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logits, _ = criterion(x, y)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()

                if options['use_softmax_in_eval']:
                    logits = torch.nn.Softmax(dim=-1)(logits)

                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels, idx) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):

                x, y = net(data, True)

                logits, _ = criterion(x, y)

                if options['use_softmax_in_eval']:
                    logits = torch.nn.Softmax(dim=-1)(logits)

                _pred_u.append(logits.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    
    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    # Average precision
    ap_score = average_precision_score([0] * len(_pred_k) + [1] * len(_pred_u),
                                       list(-np.max(_pred_k, axis=-1)) + list(-np.max(_pred_u, axis=-1)))

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.
    results['AUPR'] = ap_score * 100

    return results

def test_accuracy(net, criterion, testloader, outloader, epoch=None, **options):

    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels, idx in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logits, _ = criterion(x, y)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()

                if options['use_softmax_in_eval']:
                    logits = torch.nn.Softmax(dim=-1)(logits)

                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())


    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    # Out-of-Distribution detction evaluation
    results = {}
    results['AUROC'] = 0
    results['ACC'] = acc
    results['OSCR'] = 0 #_oscr_socre * 100.
    results['AUPR'] = 0 #ap_score * 100

    return results






import torch
import numpy as np
import scipy
import sys
from tqdm import tqdm
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import scipy
import os



def trainknn(net, criterion, trainloader, epoch=None, **options):
    net.eval()
    torch.cuda.empty_cache()

    _pred_logits, _pred_features, _labels = [], [], []

    with torch.no_grad():
        for data, labels, idx in trainloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            features, logits = net(data, True)
            
            l2_norm = torch.norm(features, p=2, dim=1, keepdim=True)  # shape: [num_samples, 1]
            l2_norm[l2_norm == 0] = 1  # avoid division by zero

            # features = features / l2_norm


            predictions = logits.data.max(1)[1]
            
            if options['use_softmax_in_eval']:
                logits = torch.nn.Softmax(dim=-1)(logits)

            _pred_logits.append(logits.data.cpu().numpy())
            _pred_features.append(features.data.cpu().numpy())
            _labels.append(labels.data.cpu().numpy())

    # Stack lists into tensors
    _pred_logits = torch.from_numpy(np.concatenate(_pred_logits, axis=0))
    _pred_features = torch.from_numpy(np.concatenate(_pred_features, axis=0))
    _labels = torch.from_numpy(np.concatenate(_labels, axis=0))

    # Filter training samples
    filtered_train_mask = _labels == torch.max(_pred_logits, dim=1).indices
    filtered_train_logits = _pred_logits[filtered_train_mask]
    filtered_train_FV = _pred_features[filtered_train_mask]
    
    filtered_labels = _labels[filtered_train_mask]
    class_counts = torch.bincount(filtered_labels)
    #print("Filtered class counts:", class_counts)
    class_counts = torch.bincount(_labels)
    #print("unFiltered class counts:", class_counts)

    # Fit GHOST
    #GHOST_model = GHOST(filtered_train_logits.cpu(), filtered_train_FV.cpu())
    return filtered_train_FV, filtered_train_logits, filtered_labels, _pred_features, _pred_logits, _labels


def trainknnprojector(net, projector, criterion, trainloader, epoch=None, **options):
    net.eval()
    torch.cuda.empty_cache()

    _pred_logits, _pred_features, _labels = [], [], []

    with torch.no_grad():
        for data, labels, idx in trainloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            features, logits = net(data, True)
            features = projector(features)


            predictions = logits.data.max(1)[1]
            
            if options['use_softmax_in_eval']:
                logits = torch.nn.Softmax(dim=-1)(logits)

            _pred_logits.append(logits.data.cpu().numpy())
            _pred_features.append(features.data.cpu().numpy())
            _labels.append(labels.data.cpu().numpy())

    # Stack lists into tensors
    _pred_logits = torch.from_numpy(np.concatenate(_pred_logits, axis=0))
    _pred_features = torch.from_numpy(np.concatenate(_pred_features, axis=0))
    _labels = torch.from_numpy(np.concatenate(_labels, axis=0))

    # Filter training samples
    filtered_train_mask = _labels == torch.max(_pred_logits, dim=1).indices
    filtered_train_logits = _pred_logits[filtered_train_mask]
    filtered_train_FV = _pred_features[filtered_train_mask]
    
    filtered_labels = _labels[filtered_train_mask]
    class_counts = torch.bincount(filtered_labels)
    #print("Filtered class counts:", class_counts)
    class_counts = torch.bincount(_labels)
    #print("unFiltered class counts:", class_counts)

    # Fit GHOST
    #GHOST_model = GHOST(filtered_train_logits.cpu(), filtered_train_FV.cpu())
    return filtered_train_FV, filtered_train_logits, filtered_labels

import numpy as np

import faiss

from copy import deepcopy
def knn_score(feas_train, feas, k=10, min=False):

    feas_train = deepcopy(np.array(feas_train))
    feas = deepcopy(np.array(feas))

    index = faiss.IndexFlatIP(feas_train.shape[-1])
    index.add(feas_train)
    D, I = index.search(feas, k)

    if min:
        scores = np.array(D.min(axis=1))
    else:
        scores = np.array(D.mean(axis=1))

    return scores




    
import torch
from abc import ABC, abstractmethod
from typing import Dict

class OODDetector(ABC):

    @abstractmethod
    def setup(self, args, train_model_outputs: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def infer(self, model_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

class NNGuideOODDetector(OODDetector):
    def __init__(self, knn_k=5):
        self.knn_k = knn_k

    def setup(self, logits_train, feas_train, train_labels):

        feas_train = F.normalize(feas_train, p=2, dim=1)


        confs_train = torch.logsumexp(logits_train, dim=1)


        self.scaled_feas_train = feas_train * confs_train[:, None]

    def infer(self, features, logits):
        logits = logits #model_outputs['logits']
        feas = F.normalize(features, p=2, dim=1)

        confs = torch.logsumexp(logits, dim=1)

        guidances = knn_score(self.scaled_feas_train, feas, k=self.knn_k)

        scores = torch.from_numpy(guidances).to(confs.device)*confs
        return scores

class EnergyOODDetector(OODDetector):

    def setup(self, filtered_train_logits, filtered_train_FV, filtered_labels):
        pass

    def infer(self, feas, logits):

        return torch.logsumexp(logits, dim=1)




class MLS(OODDetector):
    """
    Max-Logit-Score (MLS) OOD detector.
    Computes the difference between the top-1 and top-2 logits.
    Higher difference → higher confidence → likely in-distribution.
    """

    def setup(self, filtered_train_logits=None, filtered_train_FV=None, filtered_labels=None):
        pass

    def infer(self, feas, logits):
        """
        Args:
            logits (torch.Tensor): [batch_size, num_classes] model logits

        Returns:
            torch.Tensor: [batch_size] MLS scores
        """
        top2_vals, _ = torch.topk(logits, k=2, dim=1)
        
        diff = top2_vals[:, 0] 
        
        return diff

class MSP(OODDetector):
    """
    Max-Logit-Score (MLS) OOD detector.
    Computes the difference between the top-1 and top-2 logits.
    Higher difference → higher confidence → likely in-distribution.
    """

    def setup(self, filtered_train_logits=None, filtered_train_FV=None, filtered_labels=None):
        # Optional setup logic (not needed for simple MLS)
        pass

    def infer(self, feas, logits):
        """
        Args:
            logits (torch.Tensor): [batch_size, num_classes] model logits

        Returns:
            torch.Tensor: [batch_size] MLS scores
        """

        probs = torch.softmax(logits, dim=1)
        max_prob, _ = probs.max(dim=1)

        return max_prob




class PostMax:
    def __init__(self, norm: int = 2):
        """
        PostMax scoring using Generalized Pareto Distribution (GPD).

        Parameters
        ----------
        norm : int
            Norm type (e.g. 1, 2, or np.inf) for feature normalization.
        """
        self.norm = norm
        self.trained_pareto_models = None

    def score(self, norm_logits: np.ndarray):
        """
        Compute probabilities from normalized logits using a fitted GPD.

        Parameters
        ----------
        model : dict
            GPD parameters {'shape', 'loc', 'scale'}.
        norm_logits : np.ndarray
            Normalized logits for samples.

        Returns
        -------
        torch.Tensor
            Probabilities mapped via GPD CDF.
        """
        #probs = scipy.stats.genpareto.cdf(norm_logits, shape=model['shape'], loc=model['loc'], scale=model['scale'])
        probs = scipy.stats.genpareto.cdf(
                norm_logits,
                self.trained_pareto_models['shape'],          # shape parameter (c)
                loc=self.trained_pareto_models['loc'],
                scale=self.trained_pareto_models['scale']
            )
        
        return torch.from_numpy(probs)

    def train(self, labels: torch.Tensor, features: torch.Tensor, logits: torch.Tensor):
        """
        Fit GPD to normalized logits.

        Parameters
        ----------
        labels : torch.Tensor
            Ground-truth labels [N].
        features : torch.Tensor
            Feature embeddings [N, D].
        logits : torch.Tensor
            Class logits [N, C].

        Returns
        -------
        dict
            Fitted GPD parameters.
        """
        assert labels.shape[0] == features.shape[0] == logits.shape[0], "Tensors must have the same batch dimension."

        mask = labels == torch.argmax(logits, dim=1)
        filt_labels = labels[mask]
        filt_features = features[mask]
        filt_logits = logits[mask]

        norm_logits = []
        for cls_id in range(filt_logits.shape[1]):
            cls_features = filt_features[filt_labels == cls_id]
            cls_logits = filt_logits[filt_labels == cls_id]

            if cls_features.numel() == 0:
                continue

            max_cls_logits = cls_logits[:, cls_id]
            norm_cls_logits = max_cls_logits / torch.norm(cls_features, p=self.norm, dim=1)
            norm_logits.append(norm_cls_logits)

        norm_logits = torch.cat(norm_logits, dim=0).numpy()
        shape, loc, scale = scipy.stats.genpareto.fit(norm_logits)
        self.trained_pareto_models = {'shape': shape, 'loc': loc, 'scale': scale}
        return 

    def evaluate(self, model: dict, features: torch.Tensor, logits: torch.Tensor, pct: float = 1.0):
        """
        Evaluate model and compute scores.

        Parameters
        ----------
        model : dict
            GPD parameters.
        labels : torch.Tensor
            Ground-truth labels [N].
        features : torch.Tensor
            Feature embeddings [N, D].
        logits : torch.Tensor
            Class logits [N, C].
        pct : float, default=1.0
            Fraction of dataset to evaluate (e.g. 0.5 for 50%).

        Returns
        -------
        torch.Tensor
            Tensor of shape [M, 3] with (label, prediction, probability).
        """
        #assert labels.shape[0] == features.shape[0] == logits.shape[0], "Tensors must have the same batch dimension."

        # if pct < 1.0:
        #     num_imgs = int(labels.shape[0] * pct)
        #     labels, features, logits = labels[:num_imgs], features[:num_imgs], logits[:num_imgs]

        max_logits, preds = torch.max(logits, dim=1)
        norm_logits = max_logits / torch.norm(features, p=self.norm, dim=1)
        probs = self.score(norm_logits.numpy())

        # Stack results into [N, 3]
        #results = torch.stack((labels, preds, probs), dim=1)
        
        return probs

class PostMaxOODDetector(OODDetector):
    """
    Wrapper for PostMax to make it builder-compatible (like MahalanobisOODDetector).
    """

    def setup(self, filtered_train_logits, filtered_train_FV, filtered_labels):
        self.postmax = PostMax()  # use your existing PostMax class

        # Fit the GPD model on correctly classified samples
        self.model = self.postmax.train(
            labels=filtered_labels,
            features=filtered_train_FV,
            logits=filtered_train_logits
        )
        

    def infer(self, feas, logits):
        """
        Compute PostMax scores (probabilities) for given features and logits.
        """
        device = feas.device
        #max_logits, _ = torch.max(logits, dim=1)
        #norm_logits = max_logits / torch.norm(feas, p=self.postmax.norm, dim=1)
        probs = self.postmax.evaluate(model=self.model,features=feas, logits=logits )  #features: torch.Tensor, logits: torch.Tensor, pct: float = 1.0
        return probs.to(device)






class KNNOODDetector(OODDetector):
    def __init__(self, knn_k=5):
        self.knn_k = knn_k

    def setup(self, filtered_train_logits, filtered_train_FV, filtered_labels):
        feas_train = filtered_train_FV #train_model_outputs['feas']
        feas_train = F.normalize(filtered_train_FV, p=2, dim=1)

        self.feas_train = feas_train

    def infer(self, feas, logits):

        feas = F.normalize(feas, p=2, dim=1)
        
        #print (self.feas_train.size(), feas.size())
        scores = knn_score(self.feas_train, feas, k=self.knn_k, min=True)
        scores = torch.from_numpy(scores).to(feas.device)
        return scores




def testnnguide(net, criterion, trainloader, testloader, outloader, epoch=None, **options):
    filtered_train_FV, filtered_train_logits, filtered_labels = trainknn(net, criterion, trainloader, epoch=None, **options)

    NNGUIDE = PostMaxOODDetector() #MahalanobisOODDetector() #NNGuideOODDetector()
    NNGUIDE.setup(filtered_train_logits, filtered_train_FV, filtered_labels)

    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels, idx in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logits, _ = criterion(x, y)
                predictions = logits.data.max(1)[1]
                
                l2_norm = torch.norm(x, p=2, dim=1, keepdim=True)  # shape: [num_samples, 1]
                l2_norm[l2_norm == 0] = 1  # avoid division by zero

                # x = x / l2_norm



                total += labels.size(0)
                correct += (predictions == labels.data).sum()

                # if options['use_softmax_in_eval']:
                #     logits = torch.nn.Softmax(dim=-1)(logits)
                logits = NNGUIDE.infer(x.cpu(), y.cpu())
                #logits = knn_score(filtered_train_FV, x.cpu(), k=5)
                #logits = GHOST_model.ReScore(y.cpu(),x.cpu())

                _pred_k.append(logits)
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels, idx) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):

                x, y = net(data, True)
                l2_norm = torch.norm(x, p=2, dim=1, keepdim=True)  # shape: [num_samples, 1]
                l2_norm[l2_norm == 0] = 1  # avoid division by zero

                # x = x / l2_norm
                #logits = knn_score(filtered_train_FV, x.cpu(), k=5)
                #logits = NNGUIDE(filtered_train_FV,x.cpu())
                logits = NNGUIDE.infer(x.cpu(), y.cpu())
                #logits = GHOST_model.ReScore(y,x)
                _pred_u.append(logits)

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    
    # Out-of-Distribution detction evaluation
    #x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(_pred_k, _pred_u)['Bas']
    
    # OSCR
    _oscr_socre = 0 #evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    # # Average precision
    ap_score = average_precision_score([0] * len(_pred_k) + [1] * len(_pred_u),
                                       list(_pred_k) + list(_pred_u))
    
    fpr, tpr, thresholds = roc_curve([0] * len(_pred_k) + [1] * len(_pred_u), list(_pred_k) + list(_pred_u))
    target_tpr = 0.95
    idx = np.argmax(tpr >= target_tpr)
    fpr95 = fpr[idx]
    results['ACC'] = acc
    results['OSCR'] = fpr95 * 100.
    results['AUPR'] = ap_score * 100

    return results, NNGUIDE





class KNNOODDetector_updated(OODDetector):
    def __init__(self, knn_k=5):
        self.knn_k = knn_k

    def setup(self, filtered_train_logits, filtered_train_FV, filtered_labels):
        feas_train = filtered_train_FV #train_model_outputs['feas']
        feas_train = F.normalize(filtered_train_FV, p=2, dim=1)

        self.feas_train = feas_train

    def infer(self, feas, logits):

        feas = F.normalize(feas, p=2, dim=1)
        
        #print (self.feas_train.size(), feas.size())
        scores = knn_score(self.feas_train, feas, k=self.knn_k, min=False)
        scores = torch.from_numpy(scores).to(feas.device)
        return scores
    

import torch
import numpy as np
from sklearn.covariance import EmpiricalCovariance



SCOREFUNCTIONS ={
    "energy":EnergyOODDetector(),
    "nnguide5": NNGuideOODDetector(knn_k=5),
    "knn5":KNNOODDetector(knn_k=5),
    "mls":MLS(),
    "msp":MSP()
}



def scoretester(net, criterion, trainloader, testloader, outloader, epoch=None, score_func="None", **options):
    filtered_train_FV, filtered_train_logits, filtered_labels, train_FV, train_logits, train_labels = trainknn(net, criterion, trainloader, epoch=None, **options)


    scoringfunction = SCOREFUNCTIONS[score_func] #PostMaxOODDetector() #MahalanobisOODDetector() #NNGuideOODDetector()
    if score_func in ["postmax","ghost","pureknn1","vmf"]:
        print ("Filtering ...")
        scoringfunction.setup(filtered_train_logits, filtered_train_FV, filtered_labels)
    else:
        print ("No Filtering ...")
        scoringfunction.setup(train_logits, train_FV, train_labels)      

    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels, idx in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logits, _ = criterion(x, y)
                predictions = logits.data.max(1)[1]
                
                l2_norm = torch.norm(x, p=2, dim=1, keepdim=True)  # shape: [num_samples, 1]
                l2_norm[l2_norm == 0] = 1  # avoid division by zero

                # x = x / l2_norm



                total += labels.size(0)
                correct += (predictions == labels.data).sum()

                # if options['use_softmax_in_eval']:
                #     logits = torch.nn.Softmax(dim=-1)(logits)
                logits = scoringfunction.infer(x.cpu(), y.cpu())
                #logits = knn_score(filtered_train_FV, x.cpu(), k=5)
                #logits = GHOST_model.ReScore(y.cpu(),x.cpu())

                _pred_k.append(logits)
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels, idx) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):

                x, y = net(data, True)
                l2_norm = torch.norm(x, p=2, dim=1, keepdim=True)  # shape: [num_samples, 1]
                l2_norm[l2_norm == 0] = 1  # avoid division by zero

                # x = x / l2_norm
                #logits = knn_score(filtered_train_FV, x.cpu(), k=5)
                #logits = NNGUIDE(filtered_train_FV,x.cpu())
                logits = scoringfunction.infer(x.cpu(), y.cpu())
                #logits = GHOST_model.ReScore(y,x)
                _pred_u.append(logits)

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    
    # Out-of-Distribution detction evaluation
    #x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(_pred_k, _pred_u)['Bas']
    
    # OSCR
    _oscr_socre = 0 #evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    # # Average precision
    ap_score = average_precision_score([0] * len(_pred_k) + [1] * len(_pred_u),
                                       list(_pred_k) + list(_pred_u))
    
    fpr, tpr, thresholds = roc_curve([0] * len(_pred_k) + [1] * len(_pred_u), list(_pred_k) + list(_pred_u))
    target_tpr = 0.95
    idx = np.argmax(tpr >= target_tpr)
    fpr95 = fpr[idx]
    results['ACC'] = acc
    results['OSCR'] = fpr95 * 100.
    results['AUPR'] = ap_score * 100

    return results, scoringfunction




def testknn(net, criterion, trainloader, testloader, outloader, epoch=None, **options):
    filtered_train_FV, filtered_train_logits, filtered_labelsFV = trainknn(net, criterion, trainloader, epoch=None, **options)
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels, idx in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logits, _ = criterion(x, y)
                predictions = logits.data.max(1)[1]
                
                l2_norm = torch.norm(x, p=2, dim=1, keepdim=True)  # shape: [num_samples, 1]
                l2_norm[l2_norm == 0] = 1  # avoid division by zero

                # x = x / l2_norm



                total += labels.size(0)
                correct += (predictions == labels.data).sum()

                # if options['use_softmax_in_eval']:
                #     logits = torch.nn.Softmax(dim=-1)(logits)
                
                logits = knn_score(filtered_train_FV, x.cpu(), k=5)
                #logits = GHOST_model.ReScore(y.cpu(),x.cpu())

                _pred_k.append(logits)
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels, idx) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):

                x, y = net(data, True)
                l2_norm = torch.norm(x, p=2, dim=1, keepdim=True)  # shape: [num_samples, 1]
                l2_norm[l2_norm == 0] = 1  # avoid division by zero

                # x = x / l2_norm
                logits = knn_score(filtered_train_FV, x.cpu(), k=5)
                #logits = GHOST_model.ReScore(y,x)
                _pred_u.append(logits)

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    
    # Out-of-Distribution detction evaluation
    #x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(_pred_k, _pred_u)['Bas']
    
    # OSCR
    # _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    # # Average precision
    # ap_score = average_precision_score([0] * len(_pred_k) + [1] * len(_pred_u),
    #                                    list(-np.max(_pred_k, axis=-1)) + list(-np.max(_pred_u, axis=-1)))

    results['ACC'] = acc
    results['OSCR'] = 0 #_#oscr_socre * 100.
    results['AUPR'] = 0 #ap_score * 100

    return results
