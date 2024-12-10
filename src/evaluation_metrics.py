from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np


def evaluate_models(y_test, similarities, num_labels):
    """Calcula AUC e acurácia para as métricas de similaridade."""
    results = []
    for metric, predictions_proba in similarities.items():
        auc = calculate_auc_ovr(y_test, predictions_proba, num_labels)
        accuracy = calculate_accuracy(y_test, predictions_proba)
        results.append({"Metric": metric, "AUC": auc, "Accuracy": accuracy})
    return results


def calculate_auc_ovr(y_true, y_pred_proba, num_labels):
    one_hot_true = np.eye(num_labels)[y_true]
    return roc_auc_score(one_hot_true, y_pred_proba, multi_class="ovr")


def calculate_accuracy(y_true, y_pred_proba):
    y_pred = np.argmax(y_pred_proba, axis=1)
    return accuracy_score(y_true, y_pred)
