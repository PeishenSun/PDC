import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import torch
from .datasets import ImageDataset
from .transforms import get_data_transforms

def load_crack_dataset_kfold(cfg, logger, fold_idx, num_folds=5):
    """
    从裂缝数据集路径加载数据，并按照当前fold进行划分
    """
    logger.info(f"从裂缝数据集路径加载数据: {cfg.crack_path} (第{fold_idx + 1}/{num_folds}折)")

    # 确保路径存在
    if not os.path.exists(cfg.crack_path):
        raise ValueError(f"指定的裂缝数据集路径不存在: {cfg.crack_path}")

    # 检查diseased和normal文件夹是否存在
    diseased_dir = os.path.join(cfg.crack_path, "diseased")
    normal_dir = os.path.join(cfg.crack_path, "normal")

    if not os.path.exists(diseased_dir) or not os.path.exists(normal_dir):
        raise ValueError(f"裂缝数据集必须包含diseased和normal两个子文件夹")

    # 获取所有图像文件
    diseased_images = glob.glob(os.path.join(diseased_dir, "*.jpg")) + \
                      glob.glob(os.path.join(diseased_dir, "*.jpeg")) + \
                      glob.glob(os.path.join(diseased_dir, "*.png"))

    normal_images = glob.glob(os.path.join(normal_dir, "*.jpg")) + \
                    glob.glob(os.path.join(normal_dir, "*.jpeg")) + \
                    glob.glob(os.path.join(normal_dir, "*.png"))

    logger.info(f"找到diseased类别图像: {len(diseased_images)}张")
    logger.info(f"找到normal类别图像: {len(normal_images)}张")

    # 创建标签
    diseased_labels = [1] * len(diseased_images)
    normal_labels = [0] * len(normal_images)

    # 合并所有样本
    all_images = diseased_images + normal_images
    all_labels = diseased_labels + normal_labels

    # 创建K折交叉验证
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=cfg.seed)

    # 获取全部索引
    indices = np.arange(len(all_images))

    # 获取当前折的划分
    train_indices = []
    test_indices = []

    for i, (train_idx, test_idx) in enumerate(kf.split(indices)):
        if i == fold_idx:
            train_indices = train_idx
            test_indices = test_idx
            break

    # 划分训练集和测试集
    train_images = [all_images[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    test_images = [all_images[i] for i in test_indices]
    test_labels = [all_labels[i] for i in test_indices]

    logger.info(f"第{fold_idx + 1}折数据集划分完成:")
    logger.info(f"  训练集: {len(train_images)}张")
    logger.info(f"  测试集: {len(test_images)}张")

    # 准备数据集
    train_transform, test_transform = get_data_transforms(cfg)

    # 创建训练集
    train_samples = list(zip(train_images, train_labels))
    train_dataset = ImageDataset(
        root_dir=None,
        transform=train_transform,
        is_binary=True,
        samples=train_samples,
        class_to_idx={"normal": 0, "diseased": 1},
        classes=["normal", "diseased"]
    )

    # 创建测试集
    test_samples = list(zip(test_images, test_labels))
    test_dataset = ImageDataset(
        root_dir=None,
        transform=test_transform,
        is_binary=True,
        samples=test_samples,
        class_to_idx={"normal": 0, "diseased": 1},
        classes=["normal", "diseased"]
    )

    return train_dataset, test_dataset


def load_crack_dataset(cfg, logger):
    """
    从裂缝数据集路径加载数据，并自动划分为训练集、验证集和测试集
    """
    logger.info(f"从裂缝数据集路径加载数据: {cfg.crack_path}")

    # 确保路径存在
    if not os.path.exists(cfg.crack_path):
        raise ValueError(f"指定的裂缝数据集路径不存在: {cfg.crack_path}")

    # 检查diseased和normal文件夹是否存在
    diseased_dir = os.path.join(cfg.crack_path, "diseased")
    normal_dir = os.path.join(cfg.crack_path, "normal")

    if not os.path.exists(diseased_dir) or not os.path.exists(normal_dir):
        raise ValueError(f"裂缝数据集必须包含diseased和normal两个子文件夹")

    # 获取所有图像文件
    diseased_images = glob.glob(os.path.join(diseased_dir, "*.jpg")) + \
                      glob.glob(os.path.join(diseased_dir, "*.jpeg")) + \
                      glob.glob(os.path.join(diseased_dir, "*.png"))

    normal_images = glob.glob(os.path.join(normal_dir, "*.jpg")) + \
                    glob.glob(os.path.join(normal_dir, "*.jpeg")) + \
                    glob.glob(os.path.join(normal_dir, "*.png"))

    logger.info(f"找到diseased类别图像: {len(diseased_images)}张")
    logger.info(f"找到normal类别图像: {len(normal_images)}张")

    # 创建标签
    diseased_labels = [1] * len(diseased_images)
    normal_labels = [0] * len(normal_images)

    # 合并所有样本
    all_images = diseased_images + normal_images
    all_labels = diseased_labels + normal_labels

    # 分割数据集
    train_ratio, val_ratio, test_ratio = cfg.split_ratio

    # 调整比例
    test_ratio_adjusted = test_ratio
    remaining_ratio = train_ratio + val_ratio
    train_ratio_adjusted = train_ratio / remaining_ratio

    # 第一次分割：分出测试集
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=test_ratio_adjusted, random_state=cfg.seed, stratify=all_labels
    )

    # 第二次分割：分出训练集和验证集
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images, train_val_labels, test_size=(1 - train_ratio_adjusted),
        random_state=cfg.seed, stratify=train_val_labels
    )

    logger.info(f"数据集划分完成:")
    logger.info(f"  训练集: {len(train_images)}张")
    logger.info(f"  验证集: {len(val_images)}张")
    logger.info(f"  测试集: {len(test_images)}张")

    # 准备数据集
    train_transform, val_transform = get_data_transforms(cfg)

    # 创建训练集
    train_samples = list(zip(train_images, train_labels))
    train_dataset = ImageDataset(
        root_dir=None,
        transform=train_transform,
        is_binary=True,
        samples=train_samples,
        class_to_idx={"normal": 0, "diseased": 1},
        classes=["normal", "diseased"]
    )

    # 创建验证集
    val_samples = list(zip(val_images, val_labels))
    val_dataset = ImageDataset(
        root_dir=None,
        transform=val_transform,
        is_binary=True,
        samples=val_samples,
        class_to_idx={"normal": 0, "diseased": 1},
        classes=["normal", "diseased"]
    )

    # 创建测试集
    test_samples = list(zip(test_images, test_labels))
    test_dataset = ImageDataset(
        root_dir=None,
        transform=val_transform,
        is_binary=True,
        samples=test_samples,
        class_to_idx={"normal": 0, "diseased": 1},
        classes=["normal", "diseased"]
    )

    return train_dataset, val_dataset, test_dataset


def create_cv_datasets(cfg):
    """创建交叉验证数据集，支持二分类模式"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("准备交叉验证数据集...")

    # 获取数据变换
    train_transform, val_transform = get_data_transforms(cfg)

    # 1. 合并原始训练集和验证集
    temp_dataset = ImageDataset(
        root_dir=cfg.train_data_path,
        transform=None,
        is_binary=False
    )

    orig_classes = temp_dataset.classes
    logger.info(f"原始类别：{orig_classes}")

    # 是否启用二分类
    if cfg.is_binary:
        logger.info(f"启用二分类模式，正常类别为：{cfg.normal_class_name}，其他类别归为：{cfg.abnormal_class_name}")
        if cfg.normal_class_name not in orig_classes:
            raise ValueError(f"指定的正常类别 '{cfg.normal_class_name}' 不在数据集中！可用类别: {orig_classes}")

    # 使用正确的二分类设置创建训练集数据集
    combined_dataset = ImageDataset(
        root_dir=cfg.train_data_path,
        transform=None,
        is_binary=cfg.is_binary,
        normal_class_name=cfg.normal_class_name
    )

    val_samples = []
    if os.path.exists(cfg.val_data_path):
        val_dataset = ImageDataset(
            root_dir=cfg.val_data_path,
            transform=None,
            is_binary=cfg.is_binary,
            normal_class_name=cfg.normal_class_name
        )
        combined_samples = combined_dataset.samples + val_dataset.samples
    else:
        combined_samples = combined_dataset.samples

    # 提取样本路径和标签
    X = [sample[0] for sample in combined_samples]
    y = [sample[1] for sample in combined_samples]

    # 2. 创建折数分层划分
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    # 获取当前折的索引
    folds = list(skf.split(X, y))

    if cfg.current_fold >= cfg.n_folds:
        raise ValueError(f"当前折 {cfg.current_fold} 超出了总折数 {cfg.n_folds}")

    train_idx, val_idx = folds[cfg.current_fold]

    # 3. 根据索引创建当前折的训练集和验证集
    class_to_idx = combined_dataset.class_to_idx
    train_samples = [combined_samples[i] for i in train_idx]
    val_samples = [combined_samples[i] for i in val_idx]

    # 创建训练集
    train_dataset = ImageDataset(
        root_dir=None,
        transform=train_transform,
        is_binary=cfg.is_binary,
        normal_class_name=cfg.normal_class_name
    )
    train_dataset.samples = train_samples
    train_dataset.class_to_idx = class_to_idx
    train_dataset.classes = combined_dataset.classes
    if hasattr(combined_dataset, 'binary_mapping'):
        train_dataset.binary_mapping = combined_dataset.binary_mapping
    if hasattr(combined_dataset, 'orig_classes'):
        train_dataset.orig_classes = combined_dataset.orig_classes

    # 创建验证集
    val_dataset = ImageDataset(
        root_dir=None,
        transform=val_transform,
        is_binary=cfg.is_binary,
        normal_class_name=cfg.normal_class_name
    )
    val_dataset.samples = val_samples
    val_dataset.class_to_idx = class_to_idx
    val_dataset.classes = combined_dataset.classes
    if hasattr(combined_dataset, 'binary_mapping'):
        val_dataset.binary_mapping = combined_dataset.binary_mapping
    if hasattr(combined_dataset, 'orig_classes'):
        val_dataset.orig_classes = combined_dataset.orig_classes

    logger.info(f"交叉验证第 {cfg.current_fold + 1}/{cfg.n_folds} 折")
    logger.info(f"训练样本数: {len(train_samples)}, 验证样本数: {len(val_samples)}")
    logger.info(f"类别数量: {len(train_dataset.classes)}")
    logger.info(f"类别名称: {train_dataset.classes}")

    return train_dataset, val_dataset