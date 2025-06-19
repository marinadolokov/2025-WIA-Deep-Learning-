import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix
)
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from pathlib import Path


def predict_multilabel(model, dataloader, device=None, threshold=0.5):
    """
    Generate multilabel predictions and probabilities.
    Returns y_true, y_pred, y_score.
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)
    model.eval()
    all_trues, all_probs = [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            # handle tuple inputs
            if isinstance(x, (tuple, list)):
                inputs = [xi.to(device) for xi in x]
                logits = model(*inputs)
            else:
                inputs = x.to(device)
                logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_trues.append(y.cpu().numpy().astype(int))
    y_true = np.vstack(all_trues)
    y_score = np.vstack(all_probs)
    y_pred = (y_score >= threshold).astype(int)
    return y_true, y_pred, y_score


def predict_multiclass(model, dataloader, device=None, all_labels=None):
    """
    Generate multiclass predictions and probabilities.
    If all_labels is provided, ensures consistent class mapping.
    Returns y_true, y_pred, y_score.
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)
    model.eval()
    all_trues, all_scores = [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            if isinstance(x, (tuple, list)):
                inputs = [xi.to(device) for xi in x]
                logits = model(*inputs)
            else:
                inputs = x.to(device)
                logits = model(inputs)
            scores = torch.softmax(logits, dim=1).cpu().numpy()
            all_scores.append(scores)
            all_trues.append(y.cpu().numpy())
    y_true = np.concatenate(all_trues)
    y_score = np.vstack(all_scores)
    y_pred = np.argmax(y_score, axis=1)

    # handle all_labels if provided: map classes to integer indices
    if all_labels is not None:
        # ensure y_true contains only known labels
        assert set(np.unique(y_true)).issubset(set(all_labels)), \
            "y_true contains labels not in all_labels"
        # optionally, could reorder score columns to match all_labels
    return y_true, y_pred, y_score


# --- Metric and plot helpers ---

def print_classification(y_true, y_pred, labels=None, all_labels=None):
    """Print classification report, optionally with full label set."""
    if all_labels is not None:
        print(classification_report(y_true, y_pred, labels=all_labels, target_names=labels))
    else:
        print(classification_report(y_true, y_pred, target_names=labels))


def plot_multilabel_confusion(y_true, y_pred, labels=None):
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    for i, cm in enumerate(mcm):
        disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
        fig, ax = plt.subplots()
        disp.plot(ax=ax, values_format='d')
        title = labels[i] if labels else f"Label {i}"
        ax.set_title(f"Confusion Matrix: {title}")
        plt.show()

def plot_multilabel_confusion_normalized(y_true, y_pred, labels=None):
    """
    Plot per-class confusion matrices for multilabel tasks,
    with true‐label normalization (each row sums to 1).
    """
    # generate raw counts
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    for i, cm in enumerate(mcm):
        # normalize: divide each row (actual class) by its sum
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)  # replace 0/0 with 0

        disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                                      display_labels=[0, 1])
        fig, ax = plt.subplots()
        disp.plot(ax=ax, values_format='.2f', cmap='viridis')
        title = labels[i] if labels else f"Label {i}"
        ax.set_title(f"Normalized Confusion Matrix: {title}")
        plt.show()


def plot_precision_recall_multilabel(y_true, y_score, labels=None):
    n = y_true.shape[1]
    plt.figure()
    for i in range(n):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        ap = average_precision_score(y_true[:, i], y_score[:, i])
        name = labels[i] if labels else f"Class {i}"
        plt.plot(recall, precision, label=f"{name} (AP={ap:.2f})")
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curves'); plt.legend(); plt.grid(); plt.show()


def plot_roc_multilabel(y_true, y_score, labels=None):
    n = y_true.shape[1]
    plt.figure()
    for i in range(n):
        # skip if no positive samples
        if y_true[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        auc = roc_auc_score(y_true[:, i], y_score[:, i])
        name = labels[i] if labels else f"Class {i}"
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('ROC Curves'); plt.legend(); plt.grid(); plt.show()


def plot_binary_any(y_true, y_pred, y_score, positive_label='positive'):
    y_true_any = (y_true.sum(axis=1) > 0).astype(int)
    y_pred_any = (y_pred.sum(axis=1) > 0).astype(int)
    y_score_any = y_score.max(axis=1)
    print(classification_report(y_true_any, y_pred_any, target_names=['negative', positive_label]))
    cm = confusion_matrix(y_true_any, y_pred_any, normalize='true')
    disp = ConfusionMatrixDisplay(cm, display_labels=['negative', positive_label])
    fig, ax = plt.subplots(); disp.plot(ax=ax); plt.title('Binary Confusion Matrix'); plt.show()
    pr = PrecisionRecallDisplay.from_predictions(y_true_any, y_score_any); plt.title('Binary PR Curve'); plt.show()
    roc = RocCurveDisplay.from_predictions(y_true_any, y_score_any); plt.title('Binary ROC Curve'); plt.show()


# --- Full report functions ---

def full_report_multilabel(y_true, y_pred, y_score, labels=None, plot=False):
    print("=== Multilabel Classification Report ===")
    print_classification(y_true, y_pred, labels=labels)
    if plot:
        plot_multilabel_confusion_normalized(y_true, y_pred, labels)
        plot_precision_recall_multilabel(y_true, y_score, labels)
        plot_roc_multilabel(y_true, y_score, labels)
        plot_binary_any(y_true, y_pred, y_score)


def full_report_multiclass(y_true, y_pred, y_score, labels=None, all_labels=None, plot=False):
    """
    all_labels: full list of possible classes from training.
    labels: display names corresponding to all_labels.
    """
    print("=== Multiclass Classification Report ===")
    print_classification(y_true, y_pred, labels=labels, all_labels=all_labels)

    # confusion matrix only for present labels
    present = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=present, normalize='true')
    disp = ConfusionMatrixDisplay(cm, display_labels=[labels[i] if labels else str(i) for i in present])
    fig, ax = plt.subplots(); disp.plot(ax=ax, cmap='Blues'); plt.title('Confusion Matrix'); plt.show()

    if plot:
        # multiclass ROC/PR: one-vs-rest on present classes
        y_true_bin = label_binarize(y_true, classes=all_labels if all_labels else present)
        # filter present
        class_indices = [i for i, c in enumerate(all_labels if all_labels else present) if c in present]
        y_true_bin = y_true_bin[:, class_indices]
        y_score_sel = y_score[:, class_indices]
        # ROC
        plt.figure()
        for idx, c in enumerate(class_indices):
            fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_score_sel[:, idx])
            auc = roc_auc_score(y_true_bin[:, idx], y_score_sel[:, idx])
            label = labels[c] if labels else str(c)
            plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.2f})")
        plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR')
        plt.title('Multiclass ROC (present classes)'); plt.legend(); plt.show()
        # PR
        plt.figure()
        for idx, c in enumerate(class_indices):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, idx], y_score_sel[:, idx])
            ap = average_precision_score(y_true_bin[:, idx], y_score_sel[:, idx])
            label = labels[c] if labels else str(c)
            plt.plot(recall, precision, label=f"{label} (AP={ap:.2f})")
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Multiclass PR (present classes)'); plt.legend(); plt.show()


def plot_history(history: dict):
    """
    Plot training history containing at least 'train_loss' and 'val_loss'.
    Any other series in the dict will be plotted in separate subplots.

    Args:
        history: dict of lists, e.g. {
            'train_loss': [...],
            'val_loss': [...],
            'val_micro_f1': [...],
            ...
        }
    """
    # Number of epochs
    epochs = range(1, len(history['train_loss']) + 1)

    # 1) Loss plot
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Other metrics
    metric_keys = [k for k in history.keys() if k not in ('train_loss', 'val_loss')]
    if not metric_keys:
        return

    fig, axes = plt.subplots(len(metric_keys), 1, figsize=(6, 4 * len(metric_keys)))
    if len(metric_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, metric_keys):
        ax.plot(epochs, history[key], label=key.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key)
        ax.set_title(key.replace('_', ' ').title())
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()