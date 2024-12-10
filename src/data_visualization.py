import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


def plot_class_distribution(labels, label_encoder):
    """Plota a distribuição das classes nos dados."""
    classes, counts = np.unique(labels, return_counts=True)
    class_names = label_encoder.inverse_transform(classes)

    plt.bar(class_names, counts, color="skyblue")
    plt.title("Distribuição das Classes")
    plt.xlabel("Classes")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_sample_images(X, labels, label_encoder, n_samples=5):
    """Mostra algumas amostras de imagens e suas classes."""
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 5))
    for i in range(n_samples):
        image = X[i].reshape(32, 10)  # Adapte o reshape ao tamanho da imagem original
        label = label_encoder.inverse_transform([labels[i]])[0]
        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(f"Classe: {label}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_test, y_pred_proba, num_labels, label_encoder):
    """Plota as curvas ROC para cada classe."""
    plt.figure(figsize=(10, 8))
    for i in range(num_labels):
        fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=f"{label_encoder.inverse_transform([i])[0]} (AUC = {roc_auc:.2f})",
        )
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Curvas ROC")
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(y_test, y_pred, label_encoder):
    """Plota a matriz de confusão."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=label_encoder.classes_
    )
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Matriz de Confusão")
    plt.show()
