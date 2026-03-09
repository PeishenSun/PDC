import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
from torch.utils.data import DataLoader
from PIL import Image

from ..config.config import Config
from ..data.datasets import ImageDataset
from ..data.transforms import get_data_transforms
from ..data.loader import load_crack_dataset
from ..models.builders import build_multi_branch_model
from ..utils.logger import setup_logger, log_config


def test_model(cfg, model_path, model_name="best_model"):
    """测试模型并生成详细报告，支持二分类模式和裂缝数据集"""
    # 设置设备
    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 设置日志
    logger = setup_logger(cfg, log_file_name=f"test_log_{model_name}.txt")
    log_config(logger, cfg)

    # 数据加载
    logger.info("加载测试数据集...")
    _, test_transform = get_data_transforms(cfg)

    # 检查是否使用裂缝数据集
    if hasattr(cfg, 'crack_path') and cfg.crack_path is not None:
        # 从已经划分好的测试集使用
        _, _, test_dataset = load_crack_dataset(cfg, logger)
        logger.info("使用裂缝数据集的测试集")
    else:
        # 原有的测试集加载逻辑
        # 先检查测试集
        temp_test_dataset = ImageDataset(
            root_dir=cfg.test_data_path,
            transform=None,
            is_binary=False
        )
        orig_test_classes = temp_test_dataset.classes
        logger.info(f"原始测试集类别：{orig_test_classes}")

        # 确保二分类模式下，正常类别存在于测试集
        if cfg.is_binary and cfg.normal_class_name not in orig_test_classes:
            logger.warning(
                f"警告：指定的正常类别 '{cfg.normal_class_name}' 不在测试集中！可用类别: {orig_test_classes}")

        # 使用正确的二分类设置创建测试集
        test_dataset = ImageDataset(
            root_dir=cfg.test_data_path,
            transform=test_transform,
            is_binary=cfg.is_binary,
            normal_class_name=cfg.normal_class_name
        )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False
    )

    num_classes = len(test_dataset.classes)
    logger.info(f"测试样本数量: {len(test_dataset)}")
    logger.info(f"测试集类别数量: {num_classes}")
    logger.info(f"测试集类别名称: {test_dataset.classes}")

    # 测试集类别分布
    test_counts = test_dataset.get_class_counts()
    logger.info("测试集类别分布:")
    for cls, count in test_counts.items():
        logger.info(f"  {cls}: {count}")

    # 构建模型
    logger.info(f"构建模型，使用{cfg.train_mode}模式...")
    model = build_multi_branch_model(
        pretrained=False,
        backbone_type=cfg.backbone_type,
        num_classes=num_classes,
        patch_head_hidden_dim=cfg.patch_head_hidden_dim,
        dropout_rate=cfg.dropout_rate,
        fusion_method=cfg.fusion_method,
        alpha_glb_pool=cfg.alpha_glb_pool,
        mode=cfg.train_mode,
        transformer_num_layers=cfg.transformer_num_layers,
        convnext_init="IMAGENET"
    )
    model = model.to(device)

    # 加载模型权重
    logger.info(f"加载模型权重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 兼容CPU/GPU保存的模型
    try:
        model.load_state_dict(checkpoint['model'])
    except RuntimeError:
        # 尝试移除 state_dict 中的 'module.' 前缀
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model.load_state_dict(new_state_dict)
    except Exception as e:
        logger.error(f"加载模型权重失败: {str(e)}")
        # 尝试部分加载
        strict_load = False
        logger.warning(f"尝试以非严格模式加载模型权重")
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        missing, unexpected = model.load_state_dict(new_state_dict, strict=strict_load)
        logger.warning(f"缺失参数: {len(missing)}, 意外参数: {len(unexpected)}")

    # 测试模型
    model.eval()

    all_labels = []
    all_final_preds = []
    all_label_preds = []
    all_feature_preds = []
    all_final_probs = []

    # 如果是二分类模式，也保存原始类别名称
    if cfg.is_binary and hasattr(test_dataset, 'binary_mapping') and hasattr(test_dataset, 'orig_classes'):
        logger.info("二分类模式: 将保存原始类别标签用于详细分析")
        all_orig_labels = []
        orig_to_binary = test_dataset.binary_mapping
        binary_to_orig = {}
        for orig_cls, binary_cls in orig_to_binary.items():
            if binary_cls not in binary_to_orig:
                binary_to_orig[binary_cls] = []
            binary_to_orig[binary_cls].append(orig_cls)

        logger.info(f"二分类映射: {orig_to_binary}")
        logger.info(f"原始类别: {test_dataset.orig_classes}")

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_dataloader):
            if cfg.is_binary and hasattr(test_dataset, 'binary_mapping'):
                if batch_idx % 10 == 0:
                    logger.info(f"测试进度: {batch_idx}/{len(test_dataloader)}")

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 前向传播，获取各分支的预测
            final_preds, label_preds, feature_preds, _, _ = model(inputs)

            # 计算softmax后的概率分布
            final_probs = F.softmax(final_preds, dim=1)

            # 将预测结果和标签添加到列表中
            all_labels.extend(labels.cpu().numpy())
            all_final_preds.extend(torch.argmax(final_preds, dim=1).cpu().numpy())
            all_final_probs.extend(final_probs.cpu().numpy())

            # 根据模型模式收集分支预测
            if cfg.train_mode in ['label_branch', 'dual_branch', 'label_branch_enhan',
                                  'dual_2conv'] and label_preds is not None:
                all_label_preds.extend(torch.argmax(label_preds, dim=1).cpu().numpy())

            if cfg.train_mode in ['feature_branch', 'dual_branch', 'dual_2conv'] and feature_preds is not None:
                all_feature_preds.extend(torch.argmax(feature_preds, dim=1).cpu().numpy())

    # 计算并打印详细的评估指标
    target_names = test_dataset.classes

    def print_and_log_metrics(preds, labels, title, target_names, logger):
        logger.info(f"---------- {title} ----------")
        logger.info(classification_report(labels, preds, target_names=target_names, digits=4))
        # 计算混淆矩阵
        cm = confusion_matrix(labels, preds)
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
        plt.title(f"{title} 混淆矩阵")
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")

        # 保存混淆矩阵
        cm_path = os.path.join(cfg.output_dir, cfg.project_name, f"{model_name}_{title}_confusion_matrix.png")
        plt.savefig(cm_path)
        logger.info(f"{title} 混淆矩阵已保存至: {cm_path}")
        plt.close()

    # 打印融合模型的评估指标
    print_and_log_metrics(all_final_preds, all_labels, "融合模型", target_names, logger)

    # 根据模型模式打印分支评估指标
    if cfg.train_mode in ['label_branch', 'dual_branch', 'label_branch_enhan', 'dual_2conv'] and len(
            all_label_preds) > 0:
        print_and_log_metrics(all_label_preds, all_labels, "标签分支", target_names, logger)

    if cfg.train_mode in ['feature_branch', 'dual_branch', 'dual_2conv'] and len(all_feature_preds) > 0:
        print_and_log_metrics(all_feature_preds, all_labels, "特征分支", target_names, logger)

    # 如果是二分类模式，还可以保存更详细的预测结果
    if cfg.is_binary:
        logger.info(f"二分类模式下保存详细结果")

        # PR曲线
        plt.figure(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(
            np.array(all_labels) == 1,
            np.array(all_final_probs)[:, 1]
        )
        plt.plot(recall, precision, 'b-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)

        # 计算AP
        avg_precision = average_precision_score(
            np.array(all_labels) == 1,
            np.array(all_final_probs)[:, 1]
        )
        plt.text(0.5, 0.5, f'AP: {avg_precision:.4f}', horizontalalignment='center',
                 verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)

        pr_curve_path = os.path.join(cfg.output_dir, cfg.project_name, f"{model_name}_pr_curve.png")
        plt.savefig(pr_curve_path)
        logger.info(f"PR曲线已保存至: {pr_curve_path}")
        plt.close()

        # ROC曲线
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(
            np.array(all_labels) == 1,
            np.array(all_final_probs)[:, 1]
        )
        plt.plot(fpr, tpr, 'b-', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid(True)

        # 计算AUC
        roc_auc = roc_auc_score(
            np.array(all_labels) == 1,
            np.array(all_final_probs)[:, 1]
        )
        plt.text(0.5, 0.3, f'AUC: {roc_auc:.4f}', horizontalalignment='center',
                 verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)

        roc_curve_path = os.path.join(cfg.output_dir, cfg.project_name, f"{model_name}_roc_curve.png")
        plt.savefig(roc_curve_path)
        logger.info(f"ROC曲线已保存至: {roc_curve_path}")
        plt.close()

        # 记录二分类的评估指标
        logger.info("二分类评估指标:")
        logger.info(f"准确率: {accuracy_score(all_labels, all_final_preds):.4f}")
        logger.info(f"AUC: {roc_auc:.4f}")
        logger.info(f"AP: {avg_precision:.4f}")

        # 计算最佳阈值
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = _[optimal_idx]
        logger.info(f"最佳阈值 (Youden指数): {optimal_threshold:.4f}")

        # 使用最佳阈值的预测
        optimal_preds = (np.array(all_final_probs)[:, 1] >= optimal_threshold).astype(int)
        logger.info("使用最佳阈值的评估指标:")
        logger.info(classification_report(all_labels, optimal_preds, target_names=target_names, digits=4))

    # 保存测试结果为npz文件
    results_path = os.path.join(cfg.output_dir, cfg.project_name, f"{model_name}_results.npz")
    np.savez(
        results_path,
        labels=np.array(all_labels),
        predictions=np.array(all_final_preds),
        probabilities=np.array(all_final_probs),
        class_names=np.array(target_names)
    )
    logger.info(f"测试结果已保存至: {results_path}")

    logger.info("测试完成！")