import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import auc, roc_auc_score, f1_score, roc_curve
import torch.nn.functional as F

def closedset_eval(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            prediction = model.predict_closed(images)
            correct += (prediction == labels).sum().item()
            total += len(images)

    Accuracy = correct / total
    return Accuracy




def closedset_eval(encoder, classifier, known_test_loader,normed=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.eval()
    classifier.eval()
    total_known = 0
    correct_known = 0

    label_binary = []
    label_logits = []
    known_logits = []

    labels_all = np.asarray([])
    predictions_all = np.asarray([])

    for images, labels in known_test_loader:
        images = images.to(device)

        labels_np = labels.numpy()
        labels = labels.to(device)
        labels_all = np.concatenate((labels_all, labels_np))

        features = encoder.get_feature(images)
        if normed:
            features = F.normalize(features, dim=1)
            
        prediction, logits = classifier.predict_closed(features)
        for i in range(len(images)):
            total_known += 1
            if prediction[i] == labels[i]:
                correct_known += 1
        predictions_all = np.concatenate((predictions_all, prediction.cpu().detach().numpy()))

    Accuracy_known = correct_known / total_known
    return Accuracy_known

def openset_eval_contrastive(model, classifier, known_test_loader, unknown_test_loader, temperature=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    classifier.eval()
    total_known = 0
    total_unknown = 0
    correct_known = 0
    correct_unknown = 0

    label_binary = []
    label_prob = []
    known_prob = []
    unknown_prob = []


    for images, labels in known_test_loader:
        images = images.to(device)

        labels = labels.to(device)

        features = model.get_feature(images)
        prediction, probabilities = classifier.predict(features, temperature)
        for i in range(len(images)):
            total_known += 1
            if prediction[i] == labels[i]:
                correct_known += 1
            label_prob.append(np.max(probabilities[i, :]))
            known_prob.append(np.max(probabilities[i, :]))
            label_binary.append(1)

    print('mean prob of known classes:{:.3f}, std:{:.3f}'.format(np.mean(known_prob), np.std(known_prob)))

    for images, labels in unknown_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        features = model.get_feature(images)
        prediction, probabilities = classifier.predict(features, temperature)

        for i in range(len(images)):
            total_unknown += 1
            if prediction[i] == -1:
                correct_unknown += 1
            label_prob.append(np.max(probabilities[i, :]))
            unknown_prob.append(np.max(probabilities[i, :]))
            label_binary.append(0)
    print('mean prob of unknown classes:{:.3f}, std:{:.3f}'.format(np.mean(unknown_prob), np.std(unknown_prob)))

    AUC = roc_auc_score(label_binary, label_prob)
    Accuracy_known = correct_known / total_known
    Accuracy_unknown = correct_unknown / total_unknown
    Accuracy = (correct_known + correct_unknown) / (total_known + total_unknown)
    return Accuracy, Accuracy_known, Accuracy_unknown, AUC


import os
import numpy as np
import sklearn.metrics as sk
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



def openset_eval_contrastive_logits(model, classifier, known_test_loader, unknown_test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    classifier.eval()
    total_known = 0
    total_unknown = 0
    correct_known = 0
    correct_unknown = 0

    label_binary = []
    label_logits = []
    known_logits = []
    unknown_logits = []


    for images, labels in known_test_loader:
        images = images.to(device)
        labels = labels.to(device)

        features = model.get_feature(images)
        prediction, logits = classifier.predict_logits(features)
        for i in range(len(images)):
            total_known += 1
            if prediction[i] == labels[i]:
                correct_known += 1
            label_logits.append(np.max(logits[i, :]))
            known_logits.append(np.max(logits[i, :]))
            label_binary.append(1)

    print('mean logits of known classes:{:.3f}, std:{:.3f}'.format(np.mean(known_logits), np.std(known_logits)))

    for images, labels in unknown_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        features = model.get_feature(images)
        prediction, logits = classifier.predict_logits(features)

        for i in range(len(images)):
            total_unknown += 1
            if prediction[i] == -1:
                correct_unknown += 1
            label_logits.append(np.max(logits[i, :]))
            unknown_logits.append(np.max(logits[i, :]))
            label_binary.append(0)
    print('mean logits of unknown classes:{:.3f}, std:{:.3f}'.format(np.mean(unknown_logits), np.std(unknown_logits)))

    AUC = roc_auc_score(label_binary, label_logits)

    Accuracy_known = correct_known / total_known
    Accuracy_unknown = correct_unknown / total_unknown
    Accuracy = (correct_known + correct_unknown) / (total_known + total_unknown)

    return Accuracy, Accuracy_known, Accuracy_unknown, AUC

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def openset_eval_contrastive_logitsnormed(model, classifier, known_test_loader, unknown_test_loader,name=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    classifier.eval()
    total_known = 0
    total_unknown = 0
    correct_known = 0
    correct_unknown = 0

    label_binary = []
    label_logits = []

    feature_norm = []

    known_logits = []
    unknown_logits = []


    for images, labels in known_test_loader:
        images = images.to(device)
        labels = labels.to(device)

        features = model.get_feature(images)
        feature_norm.extend(torch.norm(features, dim=1).cpu().tolist())
        features = F.normalize(features, dim=1)
        prediction, logits = classifier.predict_logits(features)
        for i in range(len(images)):
            total_known += 1
            if prediction[i] == labels[i]:
                correct_known += 1
            label_logits.append(np.max(logits[i, :]))
            known_logits.append(np.max(logits[i, :]))
            label_binary.append(1)

    print('mean logits of known classes:{:.3f}, std:{:.3f}'.format(np.mean(known_logits), np.std(known_logits)))

    for images, labels in unknown_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        features = model.get_feature(images)
        feature_norm.extend(torch.norm(features, dim=1).cpu().tolist())
        features = F.normalize(features, dim=1)
        prediction, logits = classifier.predict_logits(features)

        for i in range(len(images)):
            total_unknown += 1
            if prediction[i] == -1:
                correct_unknown += 1
            label_logits.append(np.max(logits[i, :]))
            unknown_logits.append(np.max(logits[i, :]))
            label_binary.append(0)
    print('mean logits of unknown classes:{:.3f}, std:{:.3f}'.format(np.mean(unknown_logits), np.std(unknown_logits)))

    AUC = roc_auc_score(label_binary, label_logits)
    fpr, tpr, _ = roc_curve(label_binary, label_logits)
    auroc = auc(fpr, tpr) * 100.0
    
    # Compute TNR at TPR 95%
    tpr_95_idx = np.where(tpr >= 0.95)[0][0] if np.any(tpr >= 0.95) else -1
    tnr_at_tpr95 = (1 - fpr[tpr_95_idx]) * 100.0 if tpr_95_idx != -1 else 0.0
    
    # Compute Detection Accuracy (DTACC)
    dtacc = 100.0 * max(0.5 * (tpr + (1 - fpr)))
    
    # Compute AUIN & AUOUT
    precision_in = tpr / (tpr + fpr + 1e-10)
    precision_out = (1 - fpr) / ((1 - fpr) + (1 - tpr) + 1e-10)
    auin = 100.0 * auc(tpr, precision_in)
    auout = 100.0 * auc(1 - fpr, precision_out)
    
    # Store results
    results_angle = {
        'TNR': tnr_at_tpr95,
        'AUROC': auroc,
        'DTACC': dtacc,
        'AUIN': auin,
        'AUOUT': auout
    }
    print ("Normalized Classifier",results_angle)




    AUC_fnorm = roc_auc_score(label_binary, feature_norm)


    # Compute ROC Curve
    fpr, tpr, _ = roc_curve(label_binary, feature_norm)
    auroc = auc(fpr, tpr) * 100.0
    
    # Compute TNR at TPR 95%
    tpr_95_idx = np.where(tpr >= 0.95)[0][0] if np.any(tpr >= 0.95) else -1
    tnr_at_tpr95 = (1 - fpr[tpr_95_idx]) * 100.0 if tpr_95_idx != -1 else 0.0
    
    # Compute Detection Accuracy (DTACC)
    dtacc = 100.0 * max(0.5 * (tpr + (1 - fpr)))
    
    # Compute AUIN & AUOUT
    precision_in = tpr / (tpr + fpr + 1e-10)
    precision_out = (1 - fpr) / ((1 - fpr) + (1 - tpr) + 1e-10)
    auin = 100.0 * auc(tpr, precision_in)
    auout = 100.0 * auc(1 - fpr, precision_out)
    
    # Store results
    results_fnorm = {
        'TNR': tnr_at_tpr95,
        'AUROC': auroc,
        'DTACC': dtacc,
        'AUIN': auin,
        'AUOUT': auout
    }
    print ("Feature Norm",results_fnorm)


    Accuracy_known = correct_known / total_known
    Accuracy_unknown = correct_unknown / total_unknown
    Accuracy = (correct_known + correct_unknown) / (total_known + total_unknown)
    print('Accuracy Known and unknown:{:.3f}, std:{:.3f}'.format(Accuracy_known, Accuracy_unknown, Accuracy))

    return Accuracy, Accuracy_known, Accuracy_unknown, AUC, AUC_fnorm

import torch.nn.functional as F
def softmax_numpy(logits, axis=1):
    # Subtract the maximum value for numerical stability.
    logits_stable = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits_stable)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)

def openset_eval_contrastive_odd(model, classifier, known_test_loader, unknown_test_loader, name=None, feature_norm = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    classifier.eval()
    
    total_known = 0
    total_unknown = 0
    correct_known = 0
    correct_unknown = 0

    label_binary = []  # 1 for known, 0 for unknown

    label_logits = []  # Max logits for each sample
    known_logits = []  # Max logits for known samples
    unknown_logits = []  # Max logits for unknown samples

    label_softmaxlogits = []  # Max logits for each sample
    known_softmaxlogits = []  # Max logits for known samples
    unknown_softmaxlogits = []  # Max logits for unknown samples


    known_norms = []
    unknown_norms = []

    
    # Evaluate known samples
    for images, labels in known_test_loader:
        images = images.to(device)
        labels = labels.to(device)

        features = model.get_feature(images)

        
        prediction, logits = classifier.predict_logits(features)
        softmaxpreds = softmax_numpy(logits)

        for i in range(len(images)):
            total_known += 1
            if prediction[i] == labels[i]:
                correct_known += 1
            logit_max = np.max(logits[i, :])
            label_logits.append(logit_max)
            known_logits.append(logit_max)
            label_binary.append(1)


    # Evaluate unknown samples
    for images, labels in unknown_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        features = model.get_feature(images)
        prediction, logits = classifier.predict_logits(features)

        for i in range(len(images)):
            total_unknown += 1
            if prediction[i] == -1:
                correct_unknown += 1
            logit_max = np.max(logits[i, :])


            label_logits.append(logit_max)
            unknown_logits.append(logit_max)

            label_binary.append(0)
            unknown_norms.append(feature_norm)
    
    # Compute ROC Curve
    fpr, tpr, _ = roc_curve(label_binary, label_logits)
    auroc = auc(fpr, tpr) * 100.0
    
    # Compute TNR at TPR 95%
    tpr_95_idx = np.where(tpr >= 0.95)[0][0] if np.any(tpr >= 0.95) else -1
    tnr_at_tpr95 = (1 - fpr[tpr_95_idx]) * 100.0 if tpr_95_idx != -1 else 0.0
    
    # Compute Detection Accuracy (DTACC)
    dtacc = 100.0 * max(0.5 * (tpr + (1 - fpr)))
    
    # Compute AUIN & AUOUT
    precision_in = tpr / (tpr + fpr + 1e-10)
    precision_out = (1 - fpr) / ((1 - fpr) + (1 - tpr) + 1e-10)
    auin = 100.0 * auc(tpr, precision_in)
    auout = 100.0 * auc(1 - fpr, precision_out)
    
    # Store results
    results = {
        'TNR': tnr_at_tpr95,
        'AUROC': auroc,
        'DTACC': dtacc,
        'AUIN': auin,
        'AUOUT': auout
    }
    
    # Print results
    print('      ', end='')
    for mtype in ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']:
        print(f' {mtype:6s}', end='')
    print('')
    print('Base  ', end='')
    for mtype in ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']:
        print(f' {results[mtype]:6.3f}', end='')
    print('')
    


    normlabels = [1] * len(known_logits) + [0] * len(unknown_logits)
    scores = known_logits + unknown_logits


    if name is not None:
        plt.figure()
        sns.kdeplot(known_logits, label='Known', fill=True, alpha=0.5)
        sns.kdeplot(unknown_logits, label='Unknown', fill=True, alpha=0.5)

        plt.legend()
        plt.title(f'Maximum Logits Distribution')
        plt.xlabel('Maximum Logit')
        plt.ylabel('Density')
        plt.savefig(f'{name}_logit_distribution.png')
        plt.close()
    return results



def openset_eval_contrastive_ALUN(model, classifier, known_test_loader, unknown_test_loader, name=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    classifier.eval()
    
    total_known = 0
    total_unknown = 0
    correct_known = 0
    correct_unknown = 0

    label_binary = []  # 1 for known, 0 for unknown
    label_logits = []  # Max logits for each sample
    known_logits = []  # Max logits for known samples
    unknown_logits = []  # Max logits for unknown samples

    # Store features and labels for alignment and uniformity calculation
    known_features = []
    known_labels = []
    unknown_features = []

    # Evaluate known samples
    for images, labels in known_test_loader:
        images = images.to(device)
        labels = labels.to(device)

        features = model.get_feature(images)
        prediction, logits = classifier.predict_logits(features)
        
        for i in range(len(images)):
            total_known += 1
            if prediction[i] == labels[i]:
                correct_known += 1
            logit_max = np.max(logits[i, :])
            label_logits.append(logit_max)
            known_logits.append(logit_max)
            label_binary.append(1)

            # Store features and labels for alignment and uniformity
            known_features.append(features[i].cpu().detach().numpy())
            known_labels.append(labels[i].cpu().detach().numpy())

    print('Mean logits of known classes: {:.3f}, Std: {:.3f}'.format(np.mean(known_logits), np.std(known_logits)))

    # Evaluate unknown samples
    for images, labels in unknown_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        features = model.get_feature(images)
        prediction, logits = classifier.predict_logits(features)

        for i in range(len(images)):
            total_unknown += 1
            if prediction[i] == -1:
                correct_unknown += 1
            logit_max = np.max(logits[i, :])
            label_logits.append(logit_max)
            unknown_logits.append(logit_max)
            label_binary.append(0)

            # Store features for uniformity (unknown classes)
            unknown_features.append(features[i].cpu().detach().numpy())

    # Compute alignment and uniformity
    known_features = np.array(known_features)
    known_labels = np.array(known_labels)

    # Alignment: Average distance between features of the same class
    alignment = 0.0
    unique_labels = np.unique(known_labels)
    for label in unique_labels:
        class_features = known_features[known_labels == label]
        n = len(class_features)
        if n > 1:
            class_features = class_features/np.linalg.norm(class_features, axis=1)[:, None]
            pairwise_distances = np.linalg.norm(class_features[:, None] - class_features, axis=2)
            alignment += np.sum(pairwise_distances) / (n * (n - 1))
    alignment /= len(unique_labels)

    # Uniformity: Average distance between class centers
    class_centers = []
    for label in unique_labels:
        class_features = known_features[known_labels == label]
        class_centers.append(np.mean(class_features, axis=0))
    class_centers = np.array(class_centers)
    class_centers = class_centers/np.linalg.norm(class_centers, axis=1)[:, None]
    pairwise_distances = np.linalg.norm(class_centers[:, None] - class_centers, axis=1)
    uniformity = np.sum(pairwise_distances) / (len(class_centers) * (len(class_centers) - 1))

    
    return alignment, uniformity

def openset_eval_F1(model, known_test_loader, unknown_test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total_known = 0
    total_unknown = 0
    correct_known = 0
    correct_unknown = 0

    labels_all = np.asarray([])
    predictions_all = np.asarray([])

    with torch.no_grad():
        for images, labels in known_test_loader:
            images = images.to(device)

            labels_np = labels.numpy()
            labels = labels.to(device)
            labels_all = np.concatenate((labels_all, labels_np))

            prediction, probilities, dists = model.predict(images)
            for i in range(len(images)):
                total_known += 1
                if prediction[i] == labels[i]:
                    correct_known += 1
            predictions_all = np.concatenate((predictions_all, prediction.cpu().detach().numpy()))


        for images, labels in unknown_test_loader:
            images = images.to(device)
            labels_np = -1 * np.ones(len(labels))
            labels_all = np.concatenate((labels_all, labels_np))

            prediction, probilities, dists = model.predict(images)

            for i in range(len(images)):
                total_unknown += 1
                if prediction[i] == -1:
                    correct_unknown += 1

            predictions_all = np.concatenate((predictions_all, prediction.cpu().detach().numpy()))

    F1Score = f1_score(labels_all, predictions_all, average='macro')

    Accuracy_known = correct_known / total_known
    Accuracy_unknown = correct_unknown / total_unknown
    Accuracy = (correct_known + correct_unknown) / (total_known + total_unknown)
    return Accuracy, Accuracy_known, Accuracy_unknown, F1Score



def openset_eval_F1_contrastive(encoder, classifier, known_test_loader, unknown_test_loader, disturb_rate=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.eval()
    classifier.eval()
    total_known = 0
    total_unknown = 0
    correct_known = 0
    correct_unknown = 0

    label_binary = []
    label_logits = []
    known_logits = []
    unknown_logits = []

    labels_all = np.asarray([])
    predictions_all = np.asarray([])

    for images, labels in known_test_loader:
        images = images.to(device)

        labels_np = labels.numpy()
        labels = labels.to(device)
        labels_all = np.concatenate((labels_all, labels_np))

        features = encoder.get_feature(images)
        prediction, logits = classifier.predict_logits(features)
        for i in range(len(images)):
            total_known += 1
            if prediction[i] == labels[i]:
                correct_known += 1
            label_logits.append(np.max(logits[i, :]))
            known_logits.append(np.max(logits[i, :]))
            label_binary.append(1)
        predictions_all = np.concatenate((predictions_all, prediction.cpu().detach().numpy()))

    for images, labels in unknown_test_loader:
        images = images.to(device)
        labels_np = -1 * np.ones(len(labels))
        labels_all = np.concatenate((labels_all, labels_np))
        features = encoder.get_feature(images)
        prediction, logits = classifier.predict_logits(features)

        for i in range(len(images)):
            total_unknown += 1
            if prediction[i] == -1:
                correct_unknown += 1
            label_logits.append(np.max(logits[i, :]))
            unknown_logits.append(np.max(logits[i, :]))
            label_binary.append(0)

        predictions_all = np.concatenate((predictions_all, prediction.cpu().detach().numpy()))

    F1Score = f1_score(labels_all, predictions_all, average='macro')

    Accuracy_known = correct_known / total_known
    Accuracy_unknown = correct_unknown / total_unknown
    Accuracy = (correct_known + correct_unknown) / (total_known + total_unknown)
    return Accuracy, Accuracy_known, Accuracy_unknown, F1Score




import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE

def UMAP_plot(model, classifier, known_test_loader, unknown_test_loader, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    classifier.eval()
    total_known = 0
    total_unknown = 0
    correct_known = 0
    correct_unknown = 0

    label_binary = []
    label_logits = []
    known_features = []
    unknown_features = []
    labels_known = []
    labels_unknown = []



    # Process known samples
    for images, labels in known_test_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            features = model.get_feature(images) #.cpu().numpy()
            features_numpy = features.cpu().numpy()
            
            #features_numpy = features_numpy / np.linalg.norm(features_numpy, axis=1, keepdims=True)
            
            prediction, logits = classifier.predict_logits(features)

        for i in range(len(images)):
            total_known += 1
            if prediction[i] == labels[i]:
                correct_known += 1
            label_logits.append(np.max(logits[i, :]))
            label_binary.append(1)
            known_features.append(features_numpy[i])
            labels_known.append(labels[i].item())

    print('Mean logits of known classes: {:.3f}, std: {:.3f}'.format(np.mean(label_logits), np.std(label_logits)))
    unique_labels = np.unique(labels_known)
    label_to_number_mapping = {label: idx+1 for idx, label in enumerate(unique_labels)}
    print (label_to_number_mapping)
    # Process unknown samples
    for images, labels in unknown_test_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            features = model.get_feature(images) #.cpu().numpy()
            features_numpy = features.cpu().numpy()
            
            #features_numpy = features_numpy / np.linalg.norm(features_numpy, axis=1, keepdims=True)

            prediction, logits = classifier.predict_logits(features)

        for i in range(len(images)):
            total_unknown += 1
            if prediction[i] == -1:  # Assuming -1 indicates unknown
                correct_unknown += 1
            label_logits.append(np.max(logits[i, :]))
            label_binary.append(0)
            unknown_features.append(features_numpy[i])
            labels_unknown.append(-1)  # Label for unknown class

    print('Mean logits of unknown classes: {:.3f}, std: {:.3f}'.format(np.mean(label_logits), np.std(label_logits)))


    all_features = np.vstack([known_features  , unknown_features]) 
    all_labels = np.array(labels_known + labels_unknown) 

    # Perform UMAP
    reducer = umap.UMAP(n_components=2, random_state=42) 
    embeddings = reducer.fit_transform(all_features)

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab20')  
    for label in np.unique(all_labels):
        if label == -1:  
            plt.scatter(
                embeddings[all_labels == label, 0],
                embeddings[all_labels == label, 1],
                c='black', label='Unknown',marker='^', alpha=0.5
            )
        else:  
            color = cmap(label_to_number_mapping[label] % len(cmap.colors))
            plt.scatter(
                embeddings[all_labels == label, 0],
                embeddings[all_labels == label, 1],
                label=f'Class {label}', alpha=0.6, c=[color]
            )
    plt.title("t-SNE Visualization of Features from Known and Unknown Classes")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(False)
    plt.savefig(f"{model_name}_adjustedUMAP_plot.png", dpi=300, bbox_inches="tight")


