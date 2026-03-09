import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score

def compute_precision_at_recall(y_true, y_score, recall_level=0.95):
    """计算指定召回率下的精度"""
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # 对于多分类问题，需要进行独热编码处理
    num_classes = y_score.shape[1]
    y_true_onehot = np.zeros((len(y_true), num_classes))
    for i in range(len(y_true)):
        y_true_onehot[i, y_true[i]] = 1

    # 为每个类别计算P@R
    precisions = []
    for i in range(num_classes):
        if np.sum(y_true_onehot[:, i]) > 0:  # 该类有样本
            precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_score[:, i])
            # 找到最接近目标召回率的点
            idx = np.argmin(np.abs(recall - recall_level))
            precisions.append(precision[idx])
        else:
            precisions.append(0)  # 该类无样本

    # 返回平均P@R
    return np.mean(precisions)


def compute_macro_auc(y_true, y_score, num_classes):
    """计算多分类问题的macro AUC"""
    # 对真实标签进行独热编码
    y_true_onehot = np.zeros((len(y_true), num_classes))
    for i in range(len(y_true)):
        y_true_onehot[i, y_true[i]] = 1

    # 计算每个类别的AUC，然后取平均
    auc_scores = []
    for i in range(num_classes):
        if np.sum(y_true_onehot[:, i]) > 0 and np.sum(y_true_onehot[:, i]) < len(y_true):
            try:
                auc_scores.append(roc_auc_score(y_true_onehot[:, i], y_score[:, i]))
            except ValueError:
                auc_scores.append(0.5)
        else:
            auc_scores.append(0.5)

    return np.mean(auc_scores)


def calculate_harmonic_mean(metrics_dict, metric_names):
    """
    计算给定指标的调和平均

    参数:
        metrics_dict: 包含各种指标值的字典
        metric_names: 需要计算调和平均的指标名称列表

    返回:
        调和平均值
    """
    values = [metrics_dict[name] for name in metric_names if name in metrics_dict]
    if not values:
        return 0

    # 计算调和平均: n / (1/x1 + 1/x2 + ... + 1/xn)
    n = len(values)
    if 0 in values:
        return 0

    harmonic_mean = n / sum(1.0 / v for v in values)
    return harmonic_mean