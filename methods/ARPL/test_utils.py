import torch
import numpy as np
import os

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score

from tqdm import tqdm

def normalised_average_precision(y_true, y_pred):

    from sklearn.metrics.ranking import _binary_clf_curve

    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred,
                                             pos_label=None,
                                             sample_weight=None)

    n_pos = np.array(y_true).sum()
    n_neg = (1 - np.array(y_true)).sum()

    precision = tps * n_pos / (tps * n_pos + fps * n_neg)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    precision, recall, thresholds = np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]

    return -np.sum(np.diff(recall) * np.array(precision)[:-1])

def find_nearest(array, value):

    array = np.asarray(array)
    length = len(array)
    abs_diff = np.abs(array - value)

    t_star = abs_diff.min()
    equal_arr = (abs_diff == t_star).astype('float32') + np.linspace(start=0, stop=0.1, num=length)

    idx = equal_arr.argmax()

    return array[idx], idx


def acc_at_t(preds, labels, t):

    pred_t = np.copy(preds)
    pred_t[pred_t > t] = 1
    pred_t[pred_t <= t] = 0

    acc = accuracy_score(labels, pred_t.astype('int32'))

    return acc


def closed_set_acc(preds, labels):

    preds = preds.argmax(axis=-1)
    acc = accuracy_score(labels, preds)

    print('Closed Set Accuracy: {:.3f}'.format(acc))

    return acc


def tar_at_far_and_reverse(fpr, tpr, thresholds):

    # TAR at FAR
    tar_at_far_all = {}
    for t in thresholds:
        tar_at_far_all[t] = None

    for t in thresholds:
        _, idx = find_nearest(fpr, t)
        tar_at_far = tpr[idx]
        tar_at_far_all[t] = tar_at_far

        print(f'TAR @ FAR {t}: {tar_at_far}')

    # FAR at TAR
    far_at_tar_all = {}
    for t in thresholds:
        far_at_tar_all[t] = None

    for t in thresholds:
        _, idx = find_nearest(tpr, t)
        far_at_tar = fpr[idx]
        far_at_tar_all[t] = far_at_tar

        print(f'FAR @ TAR {t}: {far_at_tar}')


def acc_at_95_tpr(open_set_preds, open_set_labels, thresholds, tpr):

    # Error rate at 95% TAR
    _, idx = find_nearest(tpr, 0.95)
    t = thresholds[idx]
    acc_at_95 = acc_at_t(open_set_preds, open_set_labels, t)
    print(f'Error Rate at TPR 95%: {1 - acc_at_95}')

    return acc_at_95


def compute_auroc(open_set_preds, open_set_labels):

    auroc = roc_auc_score(open_set_labels, open_set_preds)
    print(f'AUROC: {auroc}')

    return auroc


def compute_aupr(open_set_preds, open_set_labels, normalised_ap=False):

    if normalised_ap:
        aupr = normalised_average_precision(open_set_labels, open_set_preds)
    else:
        aupr = average_precision_score(open_set_labels, open_set_preds)
    print(f'AUPR: {aupr}')

    return aupr


def compute_oscr(x1, x2, pred, labels):

    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    x1, x2 = -x1, -x2

    # x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    # pred = np.argmax(pred_k, axis=1)

    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)
    FPR_sorted, CCR_sorted = zip(*ROC)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    print(f'OSCR: {OSCR}')

    return OSCR, (FPR_sorted, CCR_sorted)

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from cycler import cycler
import torch
import random

def OSA(gt: torch.Tensor, pred: torch.Tensor, prob: torch.Tensor, thresh: float, algo_name: str):
    """
    Compute Open-Set Accuracy (OSA) and Unknown Rejection Rate (URR).

    Parameters
    ----------
    gt : torch.Tensor
        Ground truth class labels (1D tensor).
    pred : torch.Tensor
        Predicted class labels (1D tensor).
    prob : torch.Tensor
        Prediction probabilities (1D tensor or 2D with shape [N, C]).
        If 2D, only the first column is used.
    thresh : float
        Operational Threshold.
        If provided, compute OSA at this threshold, otherwise compute operational threshold that maximizes OSA.
    algo_name : str
        Algorithm name.

    Returns
    -------
    If `thresh` is provided:
        tuple
            (total_accuracy, URR, algo_name, thresh_idx), osa_score
            - total_accuracy : torch.Tensor
                OSA across thresholds.
            - URR : torch.Tensor
                Unknown Rejection Rate across thresholds.
            - algo_name : str
                Algorithm name.
            - thresh_idx : int
                Index of the operational threshold.
            - osa_score : float
                OSA achieved at the given threshold.
    If `thresh` is not provided:
        float
            The operational threshold that maximizes OSA.
    """
    
    if len(prob.shape)!= 1:
        prob = prob[:,0]
        
    prob, indices = torch.sort(prob, descending=True)
    pred_class, gt = pred[indices], gt[indices]
    
    # Get unique consecutive values and their counts
    unique_prob, counts = torch.unique_consecutive(torch.flip(prob, [0]), return_counts=True)
    # Reverse them back to original order
    unique_prob, counts = torch.flip(unique_prob, [0]), torch.flip(counts, [0])
    
    # Get labels
    pred_class_unique = pred_class.flatten().unique()
    knowns_idxs = torch.isin(gt, pred_class_unique)
    unknowns_idxs = ~knowns_idxs
    
    # Get denominator for accuracy and Unknown Rejection Rate (URR)
    num_knowns = knowns_idxs.sum().float()
    num_unknowns = unknowns_idxs.sum().float()
    
    all_unknowns = torch.cumsum(unknowns_idxs, dim=-1).float()
    URR = all_unknowns / num_unknowns
    
    correct = torch.any(gt[:, None] == pred_class, dim=1)
    correct = torch.cumsum(correct, dim=-1)
    
    knowns_acc = correct / num_knowns
    threshold_indices = torch.cumsum(counts, dim=-1) - 1
    
    total_accuracy = ((knowns_acc[threshold_indices] * num_knowns) + ((1 - URR[threshold_indices]) * num_unknowns)) / (num_unknowns + num_knowns)
    URR = 1 - URR[threshold_indices]
    
    # If threshold is given, compute OSA and idx
    if thresh:
        thresh_mask = unique_prob < thresh
        thresh_idx = torch.argmax(thresh_mask)
        osa_score = total_accuracy[thresh_idx]
        print('Max OSA', osa_score.item())
        
        return (total_accuracy, URR, thresh_idx), osa_score.item()
    
    # If threshold is not known, compute and return it.
    else:
        total_accuracy_flipped = torch.flip(total_accuracy, [0])
        max_idx = (total_accuracy_flipped.shape[0]-1) - torch.argmax(total_accuracy_flipped)
        print('Max OSA achieved with threshold:', unique_prob[max_idx].item())
        
        return unique_prob[max_idx].item()
    
import numpy as np

def compute_oscr_OP(x1, x2, pred, labels, fpr_targets=(0.01, 0.05)):
    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :param fpr_targets: tuple of FPR operating points to report
    :return:
        OSCR,
        (FPR_curve, CCR_curve),
        dict of CCR@FPR
    """

    x1, x2 = -x1, -x2

    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1

    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)

    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    CCR = [0 for _ in range(n + 2)]
    FPR = [0 for _ in range(n + 2)]

    idx = predict.argsort()
    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        CCR[k] = float(CC) / float(len(x1))
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    ROC = sorted(zip(FPR, CCR), reverse=True)
    FPR_sorted, CCR_sorted = zip(*ROC)

    FPR_sorted = np.array(FPR_sorted)
    CCR_sorted = np.array(CCR_sorted)

    # ---- OSCR (area) ----
    OSCR = 0
    for j in range(n + 1):
        h = FPR_sorted[j] - FPR_sorted[j + 1]
        w = (CCR_sorted[j] + CCR_sorted[j + 1]) / 2.0
        OSCR += h * w

    # ---- CCR @ fixed FPR ----
    # Sort ascending for interpolation
    order = np.argsort(FPR_sorted)
    FPR_interp = FPR_sorted[order]
    CCR_interp = CCR_sorted[order]

    ccr_at_fpr = {}
    for target in fpr_targets:
        value = np.interp(target, FPR_interp, CCR_interp)
        ccr_at_fpr[target] = value
        print(f"CCR @ {int(target*100)}% FPR: {value:.4f}")

    print(f"OSCR: {OSCR:.4f}")

    return  ccr_at_fpr[0.05]


# def plot_OSA(to_plot: List[Tuple[torch.Tensor, torch.Tensor, str, int]], log: bool = False, filename: Optional[str] = None, title: Optional[str] = None):
#     """
#     Plot Open-Set Accuracy curve.

#     Parameters
#     ----------
#     to_plot : list of tuples
#         Each tuple: (knowns_accuracy, URR, algo_name, thresh_idx)
#     log : bool, default=False
#         Use log scale on x-axis.
#     filename : str, optional
#         If provided, saves the figure (adds .pdf if no extension).
#     title : str, optional
#         Plot title.
#     """

#     # Cycling of colors + markers (tab10 colors + 7 distinct markers)
#     prop_cycle = (cycler(color=plt.cm.tab10.colors) * cycler(marker=['o', 's', '^', 'D', 'v', 'x', '*']))

#     fig, ax = plt.subplots()
#     ax.set_prop_cycle(prop_cycle)

#     if title:
#         fig.suptitle(title, fontsize=20)

#     for knowns_accuracy, URR, algo_name, thresh_idx in to_plot:
#         knowns_acc_flipped = torch.flip(knowns_accuracy, [0])
#         max_idx = (knowns_acc_flipped.shape[0] - 1) - torch.argmax(knowns_acc_flipped)
        
#         # Draw curve (randomized marker placement)
#         markevery = random.randint(int(URR.shape[0] * 0.1), int(URR.shape[0] * 0.2))
#         line, = ax.plot(URR, knowns_accuracy, label=algo_name, markevery=markevery)
        
#         # Extract the assigned color from this line
#         color = line.get_color()

#         # Operational threshold
#         ax.plot(URR[thresh_idx], knowns_accuracy[thresh_idx],marker="*", markersize=15, markeredgecolor=color, markerfacecolor=color)

#         # Oracle performance
#         ax.plot(URR[max_idx], knowns_accuracy[max_idx], marker="D", markersize=7, markeredgecolor=color, markerfacecolor="None")

#     # Log scale if desired
#     if log:
#         ax.set_xscale("log")

#     ax.set_ylim([0, 1])
#     ax.set_ylabel("Open-Set Accuracy", fontsize=18, labelpad=10)
#     ax.set_xlabel("Unknown Rejection Rate", fontsize=18, labelpad=10)

#     # Legends
#     test_thresh = mlines.Line2D([], [], marker='D', linestyle='None', markeredgecolor='Black', markerfacecolor='None', markersize=7, label='Oracle Performance')
#     val_thresh = mlines.Line2D([], [], color='Black', marker='*', linestyle='None',markersize=15, label='Operational Performance')

#     # First legend: oracle/operational
#     star_legend = ax.legend(handles=[test_thresh, val_thresh], loc="lower right", ncol=1, fontsize=10, frameon=False)
#     ax.add_artist(star_legend)

#     # Second legend: curve labels
#     ax.legend(loc="lower left", ncol=1, fontsize=12, frameon=False)

#     # Save plot
#     if filename:
#         if "." not in filename:
#             filename = f"{filename}.pdf"
#         fig.savefig(filename, bbox_inches="tight")

#     plt.close()
    
def compute_openauc(x1, x2, pred, labels):
    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    x1, x2, correct = x1.tolist(), x2.tolist(), (pred == labels).tolist()
    m_x2 = max(x2) + 1e-5
    y_score = [value if hit else m_x2 for value, hit in zip(x1, correct)] + x2
    y_true = [0] * len(x1) + [1] * len(x2)
    open_auc = roc_auc_score(y_true, y_score)
    print('OpenAUC:', open_auc)
    return open_auc


class EvaluateOpenSet():

    def __init__(self, model, save_dir, known_data_loader, unknown_data_loader, device=None):

        self.model = model
        self.known_data_loader = known_data_loader
        self.unknown_data_loader = unknown_data_loader
        self.save_dir = save_dir

        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.device = device

        # Init empty lists for saving labels and preds
        self.closed_set_preds = {0: [], 1: []}
        self.open_set_preds = {0: [], 1: []}

        self.closed_set_labels = {0: [], 1: []}
        self.open_set_labels = {0: [], 1: []}

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def predict(self):

        with torch.no_grad():
            for open_set_label, loader in enumerate((self.known_data_loader, self.unknown_data_loader)):

                if open_set_label:
                    print('Forward pass through Open Set test set...')
                else:
                    print('Forward pass through Closed Set test set...')

                for batch_idx, batch in enumerate(tqdm(loader)):

                    imgs, labels, idxs = [x.to(self.device) for x in batch]

                    # Model forward
                    output = self.model(imgs)
                    closed_set_preds, open_set_preds = [x.cpu().numpy().tolist() for x in output]

                    # Update preds and labels
                    self.closed_set_preds[open_set_label].extend(closed_set_preds)
                    self.open_set_preds[open_set_label].extend(open_set_preds)

                    self.closed_set_labels[open_set_label].extend(labels.cpu().numpy().tolist())
                    self.open_set_labels[open_set_label].extend([open_set_label] * len(labels))

        # Save to disk
        save_names = ['closed_set_preds.pt', 'open_set_preds.pt', 'closed_set_labels.pt', 'open_set_labels.pt']
        save_lists = [self.closed_set_preds, self.open_set_preds, self.closed_set_labels, self.open_set_labels]

        for name, x in zip(save_names, save_lists):

            path = os.path.join(self.save_dir, name)
            torch.save(x, path)

    @staticmethod
    def evaluate(self, load=True, preds=None, normalised_ap=False, returnplot=False):

        if load:
            save_names = ['closed_set_preds.pt', 'open_set_preds.pt', 'closed_set_labels.pt', 'open_set_labels.pt']

            closed_set_preds, open_set_preds, closed_set_labels, open_set_labels = \
                [torch.load(os.path.join(self.save_dir, name)) for name in save_names]

        else:

            closed_set_preds, open_set_preds, closed_set_labels, open_set_labels = preds

        fpr95=100*np.sum(open_set_preds[1] > np.percentile(open_set_preds[0], 5)) / len(open_set_preds[1])
        print ("FPR95",fpr95)
        open_set_preds = np.array(open_set_preds[0] + open_set_preds[1])
        open_set_labels = np.array(open_set_labels[0] + open_set_labels[1])

        # ----------------------------
        # CLOSED SET EVALUATION
        # ----------------------------

        test_acc = closed_set_acc(np.array(closed_set_preds[0]), np.array(closed_set_labels[0]))


        # ----------------------------
        # OPEN SET EVALUATION
        # ----------------------------

        fpr, tpr, thresh = roc_curve(open_set_labels, open_set_preds, drop_intermediate=False)
        acc_95 = acc_at_95_tpr(open_set_preds, open_set_labels, thresh, tpr)
        auroc = compute_auroc(open_set_preds, open_set_labels)
        

        
        aupr = compute_aupr(open_set_preds, open_set_labels, normalised_ap=normalised_ap)

        # OSCR calcs
        open_set_preds_known_cls = open_set_preds[~open_set_labels.astype('bool')]
        open_set_preds_unknown_cls = open_set_preds[open_set_labels.astype('bool')]
        closed_set_preds_pred_cls = np.array(closed_set_preds[0]).argmax(axis=-1)
        labels_known_cls = np.array(closed_set_labels[0])

        # print(type(open_set_preds))
        # print(len(open_set_preds) if hasattr(open_set_preds, '__len__') else 'No length')


        oscr, list_oscr = compute_oscr(open_set_preds_known_cls, open_set_preds_unknown_cls, closed_set_preds_pred_cls, labels_known_cls)
        open_auc = compute_openauc(open_set_preds_known_cls, open_set_preds_unknown_cls, closed_set_preds_pred_cls, labels_known_cls)
        operatingpoint = compute_oscr_OP(open_set_preds_known_cls, open_set_preds_unknown_cls, closed_set_preds_pred_cls, labels_known_cls, fpr_targets=(0.01, 0.05)) 
        if returnplot:
            #rocstuff = fpr, tpr, thresh
            return (test_acc, acc_95, auroc, aupr, oscr, fpr95, operatingpoint, list_oscr)
        else:
            return (test_acc, acc_95, auroc, aupr, oscr, fpr95, operatingpoint, None)
        
        #

    def osa_metric_get_threshold(self, gt, pred, prob, thresh=None, algo_name="Model"):
            """
            Integrated OSA function. 
            Note: I've adapted the logic to work as a class method.
            """
            # Ensure inputs are tensors for the logic below
            gt = torch.as_tensor(gt)
            pred = torch.as_tensor(pred)
            prob = torch.as_tensor(prob)

            if len(prob.shape) != 1:
                prob = prob[:, 0]
                
            prob, indices = torch.sort(prob, descending=True)
            pred_class, gt = pred[indices], gt[indices]
            
            unique_prob, counts = torch.unique_consecutive(torch.flip(prob, [0]), return_counts=True)
            unique_prob, counts = torch.flip(unique_prob, [0]), torch.flip(counts, [0])
            
            pred_class_unique = pred_class.flatten().unique()
            knowns_idxs = torch.isin(gt, pred_class_unique)
            unknowns_idxs = ~knowns_idxs
            
            num_knowns = knowns_idxs.sum().float()
            num_unknowns = unknowns_idxs.sum().float()
            
            all_unknowns = torch.cumsum(unknowns_idxs, dim=-1).float()
            URR = all_unknowns / num_unknowns
            
            # Check if ground truth matches the predicted class
            # (Assuming pred is the class index)
            correct = (gt == pred_class).float() 
            correct_cumsum = torch.cumsum(correct, dim=-1)
            
            knowns_acc = correct_cumsum / num_knowns
            threshold_indices = torch.cumsum(counts, dim=-1) - 1
            
            total_accuracy = ((knowns_acc[threshold_indices] * num_knowns) + 
                            ((1 - URR[threshold_indices]) * num_unknowns)) / (num_unknowns + num_knowns)
            URR_final = 1 - URR[threshold_indices]
            
            if thresh:
                thresh_mask = unique_prob < thresh
                thresh_idx = torch.argmax(thresh_mask.byte()) 
                osa_score = total_accuracy[thresh_idx]
                return osa_score.item()
            else:
                total_accuracy_flipped = torch.flip(total_accuracy, [0])
                max_idx = (total_accuracy_flipped.shape[0]-1) - torch.argmax(total_accuracy_flipped)
                return unique_prob[max_idx].item(), total_accuracy[max_idx].item()
            


class EvaluateOpenSetInline(EvaluateOpenSet):

    def __init__(self, *args, **kwargs):

        super(EvaluateOpenSetInline, self).__init__(*args, **kwargs)

    def predict_and_eval(self):

        self.model.eval()

        print('Testing Open Set...')

        with torch.no_grad():
            for open_set_label, loader in enumerate((self.known_data_loader, self.unknown_data_loader)):
                for batch_idx, batch in enumerate(tqdm(loader)):

                    imgs, labels, idxs = [x.to(self.device) for x in batch]

                    # Model forward
                    output = self.model(imgs)
                    closed_set_preds, open_set_preds = [x.cpu().numpy().tolist() for x in output]

                    # Update preds and labels
                    self.closed_set_preds[open_set_label].extend(closed_set_preds)
                    self.open_set_preds[open_set_label].extend(open_set_preds)

                    self.closed_set_labels[open_set_label].extend(labels.cpu().numpy().tolist())
                    self.open_set_labels[open_set_label].extend([open_set_label] * len(labels))

        open_set_preds = np.array(self.open_set_preds[0] + self.open_set_preds[1])
        open_set_labels = np.array(self.open_set_labels[0] + self.open_set_labels[1])

        # ----------------------------
        # CLOSED SET EVALUATION
        # ----------------------------

        test_acc = closed_set_acc(np.array(self.closed_set_preds[0]), np.array(self.closed_set_labels[0]))

        # ----------------------------
        # OPEN SET EVALUATION
        # ----------------------------

        fpr, tpr, thresh = roc_curve(open_set_labels, open_set_preds, drop_intermediate=False)
        acc_95 = acc_at_95_tpr(open_set_preds, open_set_labels, thresh, tpr)
        auroc = compute_auroc(open_set_preds, open_set_labels)

        return (test_acc, acc_95, auroc)

class ModelTemplate(torch.nn.Module):

    def forward(self, imgs):
        """
        :param imgs:
        :return: Closed set and open set predictions on imgs
        """
        pass

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


class EvaluateOpenSetScoringFunction(): 

    def __init__(self, model, save_dir, known_data_loader, unknown_data_loader, scoringfunction, device=None):

        self.model = model
        self.known_data_loader = known_data_loader
        self.unknown_data_loader = unknown_data_loader
        self.save_dir = save_dir

        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.device = device

        # Init empty lists for saving labels and preds
        self.closed_set_preds = {0: [], 1: []}
        self.open_set_preds = {0: [], 1: []}

        self.closed_set_labels = {0: [], 1: []}
        self.open_set_labels = {0: [], 1: []}

        self.scoringfunction = scoringfunction
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def predict(self):

        with torch.no_grad():
            for open_set_label, loader in enumerate((self.known_data_loader, self.unknown_data_loader)):

                if open_set_label:
                    print('Forward pass through Open Set test set...')
                else:
                    print('Forward pass through Closed Set test set...')

                for batch_idx, batch in enumerate(tqdm(loader)):

                    imgs, labels, idxs = [x.to(self.device) for x in batch]

                    # Model forward
                    features, logits = self.model(imgs)
                    scores = self.scoringfunction.infer(features.cpu(), logits.cpu())

                    output = self.model(imgs)
                    closed_set_preds, open_set_preds = [x.cpu().numpy().tolist() for x in output]


                    # Update preds and labels
                    self.closed_set_preds[open_set_label].extend(closed_set_preds)
                    self.open_set_preds[open_set_label].extend(open_set_preds)

                    self.closed_set_labels[open_set_label].extend(labels.cpu().numpy().tolist())
                    self.open_set_labels[open_set_label].extend([open_set_label] * len(labels))

        # Save to disk
        save_names = ['closed_set_preds.pt', 'open_set_preds.pt', 'closed_set_labels.pt', 'open_set_labels.pt']
        save_lists = [self.closed_set_preds, self.open_set_preds, self.closed_set_labels, self.open_set_labels]

        for name, x in zip(save_names, save_lists):

            path = os.path.join(self.save_dir, name)
            torch.save(x, path)

    @staticmethod
    def evaluate(self, load=True, preds=None, normalised_ap=False):

        if load:
            save_names = ['closed_set_preds.pt', 'open_set_preds.pt', 'closed_set_labels.pt', 'open_set_labels.pt']

            closed_set_preds, open_set_preds, closed_set_labels, open_set_labels = \
                [torch.load(os.path.join(self.save_dir, name)) for name in save_names]

        else:

            closed_set_preds, open_set_preds, closed_set_labels, open_set_labels = preds


        open_set_preds = np.array(open_set_preds[0] + open_set_preds[1])
        open_set_labels = np.array(open_set_labels[0] + open_set_labels[1])

        fpr95=np.sum(open_set_preds[1] > np.percentile(open_set_preds[0], 5)) / len(open_set_preds[1])
        #fpr95=np.sum(novel > np.percentile(known, 5)) / len(novel)
        # ----------------------------
        # CLOSED SET EVALUATION
        # ----------------------------

        test_acc = closed_set_acc(np.array(closed_set_preds[0]), np.array(closed_set_labels[0]))

        # ----------------------------
        # OPEN SET EVALUATION
        # ----------------------------

        fpr, tpr, thresh = roc_curve(open_set_labels, open_set_preds, drop_intermediate=False)
        acc_95 = acc_at_95_tpr(open_set_preds, open_set_labels, thresh, tpr)
        auroc = compute_auroc(open_set_preds, open_set_labels)
        aupr = compute_aupr(open_set_preds, open_set_labels, normalised_ap=normalised_ap)

        

        # OSCR calcs
        open_set_preds_known_cls = open_set_preds[~open_set_labels.astype('bool')]
        open_set_preds_unknown_cls = open_set_preds[open_set_labels.astype('bool')]
        closed_set_preds_pred_cls = np.array(closed_set_preds[0]).argmax(axis=-1)
        labels_known_cls = np.array(closed_set_labels[0])

        oscr = compute_oscr(open_set_preds_known_cls, open_set_preds_unknown_cls, closed_set_preds_pred_cls, labels_known_cls)

        return (test_acc, acc_95, auroc, aupr, oscr, fpr95)

if __name__ == '__main__':

    from sklearn.metrics.ranking import precision_recall_curve

    np.random.seed(0)

    y_true = [0] * 40 + [1] * 60
    y_pred = np.random.uniform(size=(100,))

    def _binary_uninterpolated_average_precision(
            y_true, y_score):
        precision, recall, _ = precision_recall_curve(
            y_true, y_score, None, None)
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

    ap = average_precision_score(y_true, y_pred)
    ap1 = _binary_uninterpolated_average_precision(y_true, y_pred)
    ap2 = normalised_average_precision(y_true, y_pred)

    print(ap)
    print(ap1)
    print(ap2)


