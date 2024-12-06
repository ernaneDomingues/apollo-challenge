from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    top_k_accuracy_score,
)
import matplotlib.pyplot as plt


def plot_roc_curve(true_labels, predictions_cosine, predictions_euclidean):
    fpr_cos, tpr_cos, _ = roc_curve(true_labels, predictions_cosine)
    fpr_euc, tpr_euc, _ = roc_curve(true_labels, predictions_euclidean)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_cos, tpr_cos, label="Cosine Distance")
    plt.plot(fpr_euc, tpr_euc, label="Euclidean Distance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


def plot_roc_auc(true_labels, predictions_cosine, predictions_euclidean):
    fpr_cos, tpr_cos, _ = roc_auc_score(true_labels, predictions_cosine)
    fpr_euc, tpr_euc, _ = roc_auc_score(true_labels, predictions_euclidean)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_cos, tpr_cos, label="Cosine Distance")
    plt.plt(fpr_euc, tpr_euc, label="Euclidean Distance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Roc Auc Score")
    plt.legend()
    plt.show()


def plot_accuracy_score(true_labels, predictions_cosine, predictions_euclidean):
    fpr_cos, tpr_cos, _ = accuracy_score(true_labels, predictions_cosine)
    fpr_euc, tpr_euc, _ = accuracy_score(true_labels, predictions_euclidean)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_cos, tpr_cos, label="Cosine Distance")
    plt.plt(fpr_euc, tpr_euc, label="Euclidean Distance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Accuracy Score")
    plt.legend()
    plt.show()


def plot_top_k(true_labels, predictions_cosine, predictions_euclidean):
    fpr_cos, tpr_cos, _ = top_k_accuracy_score(true_labels, predictions_cosine)
    fpr_euc, tpr_euc, _ = top_k_accuracy_score(true_labels, predictions_euclidean)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_cos, tpr_cos, label="Cosine Distance")
    plt.plt(fpr_euc, tpr_euc, label="Euclidean Distance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Top K Score")
    plt.legend()
    plt.show()
