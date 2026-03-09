import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()
    return cm


def plot_roc_curve(y_true, y_score, class_names=None, save_path=None):
    """
    绘制ROC曲线（多分类或二分类）
    """
    if len(np.unique(y_true)) == 2:
        # 二分类
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)
    else:
        # 多分类
        n_classes = len(np.unique(y_true))
        y_true_onehot = np.zeros((len(y_true), n_classes))
        for i in range(len(y_true)):
            y_true_onehot[i, y_true[i]] = 1

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            label = f'Class {i} (AUC = {roc_auc:.2f})' if class_names is None else f'{class_names[i]} (AUC = {roc_auc:.2f})'
            plt.plot(fpr, tpr, lw=2, label=label)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_precision_recall_curve(y_true, y_score, save_path=None):
    """
    绘制PR曲线
    """
    if len(np.unique(y_true)) == 2:
        # 二分类
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                 label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
    else:
        # 多分类
        n_classes = len(np.unique(y_true))
        y_true_onehot = np.zeros((len(y_true), n_classes))
        for i in range(len(y_true)):
            y_true_onehot[i, y_true[i]] = 1

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_score[:, i])
            avg_precision = average_precision_score(y_true_onehot[:, i], y_score[:, i])
            plt.plot(recall, precision, lw=2, label=f'Class {i} (AP = {avg_precision:.2f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Multi-class Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    绘制训练历史
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss曲线
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy曲线
    axes[1].plot(epochs, train_accs, 'b-', label='Training Accuracy')
    axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def visualize_attention_maps(model, images, device, save_path=None):
    """
    可视化注意力图（如果模型支持）
    """
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        # 这里需要根据具体模型实现获取注意力图
        # 示例：获取模型的注意力权重
        if hasattr(model, 'get_attention_maps'):
            attention_maps = model.get_attention_maps(images)

            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            axes = axes.ravel()

            for i in range(min(8, len(images))):
                axes[i].imshow(images[i].cpu().permute(1, 2, 0).numpy())
                axes[i].axis('off')
                axes[i].set_title(f'Image {i + 1}')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.close()

    return None