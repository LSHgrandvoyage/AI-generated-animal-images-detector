from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def compute_metrics(y_true, y_pred, y_prob):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    if len(y_prob.shape) != 1:
        y_prob = y_prob[:, 1]

    metrics = {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision" : precision_score(y_true, y_pred),
        "recall" : recall_score(y_true, y_pred),
        "f1-score" : f1_score(y_true, y_pred),
        "roc-auc" : roc_auc_score(y_true, y_prob)
    }

    return metrics
