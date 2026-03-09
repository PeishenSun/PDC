import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from ..config.config import Config
from ..data.datasets import ImageDataset
from ..data.transforms import get_data_transforms
from ..data.loader import load_crack_dataset, create_cv_datasets
from ..data.cv_split import NormalClassSampler
from ..models.builders import build_multi_branch_model
from ..training.trainer import train_epoch, validate
from ..training.optimizer import Ranger21
from ..utils.logger import setup_logger, log_config
from ..utils.checkpoint import save_checkpoint
from ..training.metrics import calculate_harmonic_mean


def main(cfg):
    # 设置设备
    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 设置日志
    logger = setup_logger(cfg)
    log_config(logger, cfg)

    # 如果启用了AMP，记录到日志
    if cfg.use_amp:
        logger.info("已启用AMP（自动混合精度）训练，这将加速训练并减少内存使用")

    # 设置随机种子
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.backends.cudnn.benchmark = True
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True

    # 数据加载
    logger.info("加载数据集")
    train_transform, val_transform = get_data_transforms(cfg)

    # 检查是否使用裂缝数据集
    if hasattr(cfg, 'crack_path') and cfg.crack_path is not None:
        # 使用裂缝数据集
        logger.info(f"使用裂缝数据集路径: {cfg.crack_path}")
        cfg.is_binary = True
        cfg.normal_class_name = "normal"
        cfg.abnormal_class_name = "diseased"

        # 加载并划分裂缝数据集
        train_dataset, val_dataset, test_dataset = load_crack_dataset(cfg, logger)

        if "crack" not in cfg.project_name:
            cfg.project_name = f"{cfg.project_name}_crack"
    # 检查是否使用交叉验证
    elif cfg.use_cross_validation:
        logger.info(f"使用{cfg.n_folds}折交叉验证，当前为第{cfg.current_fold + 1}折")
        train_dataset, val_dataset = create_cv_datasets(cfg)
    else:
        # 非交叉验证模式，直接加载训练集和验证集
        train_dataset, val_dataset = load_standard_datasets(cfg, train_transform, val_transform)

    logger.info(f"类别数量: {len(train_dataset.classes) if hasattr(train_dataset, 'classes') else 'unknown'}")
    logger.info(f"类别名称: {train_dataset.classes if hasattr(train_dataset, 'classes') else 'unknown'}")
    logger.info(f"训练样本数量: {len(train_dataset)}")
    logger.info(f"验证样本数量: {len(val_dataset)}")

    # 根据模式决定是否使用 NormalClassSampler
    if cfg.train_mode == 'dual_2conv' and cfg.normal_class_ratio < 1.0:
        logger.info(f"在 dual_2conv 模式下使用正常类采样，正常类比例: {cfg.normal_class_ratio}")
        sampler = NormalClassSampler(
            train_dataset,
            normal_class_name=cfg.normal_class_name,
            normal_class_ratio=cfg.normal_class_ratio
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if cfg.num_workers > 0 else False
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if cfg.num_workers > 0 else False
        )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # 获取类别数量
    if hasattr(train_dataset, 'classes'):
        num_classes = len(train_dataset.classes)
    else:
        num_classes = len(train_dataset.dataset.classes)

    logger.info(f"模型输出类别数量: {num_classes}")

    # 构建模型
    logger.info(f"构建{cfg.train_mode}模型...")
    model = build_multi_branch_model(
        pretrained=cfg.use_pretrained,
        backbone_type=cfg.backbone_type,
        num_classes=num_classes,
        patch_head_hidden_dim=cfg.patch_head_hidden_dim,
        dropout_rate=cfg.dropout_rate,
        fusion_method=cfg.fusion_method,
        mode=cfg.train_mode,
        convnext_init=cfg.convnext_init,
        transformer_num_layers=cfg.transformer_num_layers,
        alpha_glb_pool=cfg.alpha_glb_pool,
        feature_branch_path=cfg.feature_branch_path,
        label_branch_path=cfg.label_branch_path,
        is_swin=cfg.is_swin
    )

    # 模型转移到GPU
    model = model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数量: {total_params:,}, 可训练参数: {trainable_params:,}")

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 使用Ranger21优化器
    optimizer = Ranger21(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-5,
        num_epochs=cfg.num_epochs,
        num_batches_per_epoch=len(train_dataloader),
    )

    # 训练循环
    best_f1 = 0.0
    best_acc = 0.0
    best_harmonic_mean = 0.0

    # 在保存路径中包含交叉验证信息
    if cfg.use_cross_validation:
        model_dir = os.path.join(cfg.output_dir, cfg.project_name)
        os.makedirs(model_dir, exist_ok=True)
        best_f1_model_path = os.path.join(model_dir, f"best_f1_model_fold{cfg.current_fold}.pth")
        best_acc_model_path = os.path.join(model_dir, f"best_acc_model_fold{cfg.current_fold}.pth")
        best_harmonic_mean_model_path = os.path.join(model_dir, f"best_harmonic_mean_model_fold{cfg.current_fold}.pth")
    else:
        model_dir = os.path.join(cfg.output_dir, cfg.project_name)
        os.makedirs(model_dir, exist_ok=True)
        best_f1_model_path = os.path.join(model_dir, "best_f1_model.pth")
        best_acc_model_path = os.path.join(model_dir, "best_acc_model.pth")
        best_harmonic_mean_model_path = os.path.join(model_dir, "best_harmonic_mean_model.pth")

    # 记录训练开始时间
    start_time = time.time()

    for epoch in range(cfg.num_epochs):
        # 训练
        logger.info(f"Epoch {epoch + 1}/{cfg.num_epochs}")
        train_metrics = train_epoch(model, train_dataloader, criterion, optimizer, device, cfg, logger=logger)

        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, Final Loss: {train_metrics['final_loss']:.4f}, "
            f"Sparse Loss: {train_metrics['sparse_loss']:.4f}, Acc: {train_metrics['acc']:.4f}, "
            f"F1 (macro): {train_metrics['f1_macro']:.4f}, Time: {train_metrics['time']:.2f}s"
        )

        # 验证
        val_metrics = validate(model, val_dataloader, criterion, device, cfg, logger=logger)

        # 计算调和平均
        harmonic_mean = calculate_harmonic_mean(val_metrics, cfg.metrics_for_harmonic_mean)

        # 记录验证结果和调和平均
        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, "
            f"F1 (macro): {val_metrics['f1_macro']:.4f}, P@R95: {val_metrics['p_at_r95']:.4f}, "
            f"P@R90: {val_metrics['p_at_r90']:.4f}, Macro AUC: {val_metrics['macro_auc']:.4f}, "
            f"Harmonic Mean: {harmonic_mean:.4f}"
        )

        # 保存最佳模型
        if harmonic_mean > best_harmonic_mean:
            best_harmonic_mean = harmonic_mean
            save_checkpoint(model, optimizer, epoch, val_metrics, best_harmonic_mean_model_path)
            logger.info(f"新的最佳调和平均模型已保存! Harmonic Mean: {best_harmonic_mean:.4f}")

        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            save_checkpoint(model, optimizer, epoch, val_metrics, best_f1_model_path)
            logger.info(f"新的最佳 F1 模型已保存! F1 (macro): {best_f1:.4f}")

        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            save_checkpoint(model, optimizer, epoch, val_metrics, best_acc_model_path)
            logger.info(f"新的最佳准确率模型已保存! Acc: {best_acc:.4f}")

        # 打印当前训练总时间
        elapsed_time = time.time() - start_time
        logger.info(f"训练已进行 {elapsed_time / 3600:.2f} 小时")

    logger.info("训练完成")
    return model