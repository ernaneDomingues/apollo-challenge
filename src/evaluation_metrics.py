from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

def calculate_auc_ovr(y_true, y_pred_proba, num_labels):
    """
    Calcula o AUC-ROC (One-vs-Rest) para cada classe.
    """
    one_hot_true = np.eye(num_labels)[y_true]
    return roc_auc_score(one_hot_true, y_pred_proba, multi_class="ovr")

def calculate_accuracy(y_true, y_pred_proba):
    """
    Calcula a acurácia baseada nas probabilidades previstas.
    """
    y_pred = np.argmax(y_pred_proba, axis=1)  # Classe com maior probabilidade
    return accuracy_score(y_true, y_pred)
