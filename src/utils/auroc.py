import numpy as np


def calculate_auroc(scores, labels):
    """
    Compute AUROC given prediction scores and ground truth labels.
    
    Args:
        scores (List[float]): Model output scores (higher = more likely AI).
        labels (List[int]): Ground truth labels (1 = AI-generated, 0 = human).
        
    Returns:
        float: Area Under the ROC Curve (AUROC).
    """
    # Convert to numpy arrays
    scores = np.array(scores)
    labels = np.array(labels)

    # Sort by descending scores
    desc_score_indices = np.argsort(-scores)
    scores = scores[desc_score_indices]
    labels = labels[desc_score_indices]

    # Total positives and negatives
    P = np.sum(labels == 1)
    N = np.sum(labels == 0)

    tpr_list = []
    fpr_list = []

    tp = 0
    fp = 0

    for i in range(len(scores)):
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / P if P > 0 else 0)
        fpr_list.append(fp / N if N > 0 else 0)

    # Sort by FPR for integration
    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)

    # Trapezoidal integration of TPR over FPR
    auroc = np.trapz(tpr_array, fpr_array)
    return auroc
