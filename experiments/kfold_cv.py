import os
import time
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..config.config import Config
from ..data.loader import load_crack_dataset_kfold
from ..models.builders import build_multi_branch_model
from ..training.trainer import train_epoch, validate
from ..training.optimizer import Ranger21
from ..utils.logger import setup_logger, log_config
from ..utils.checkpoint import save_checkpoint


def run_single_fold(cfg, fold_idx, num_folds=5):
    """运行单次交叉验证折"""
    # 设置设备
    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 修改项目名称，添加折号
    original_project_name = cfg.project_name
    cfg.project_name = f"{original_project_name}_fold{fold_idx + 1}"

    # 设置日志
    logger = setup_logger(cfg)
    log_config(logger, cfg)

    # 如果启用了AMP，记录到日志
    if cfg.use_amp:
        logger.info("已启用AMP（自动混合精度）训练，这将加速训练并减少内存使用")

    # 设置随机种子
    seed = cfg.seed + fold_idx
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True

    # 数据加载
    logger.info(f"加载第{fold_idx + 1}折数据集")
    train_dataset, test_dataset = load_crack_dataset_kfold(cfg, logger, fold_idx, num_folds)

    # 创建数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if cfg.num_workers > 0 else False
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # 获取类别数量
    num_classes = len(train_dataset.classes)
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
        label_branch_path=cfg.label_branch_path
    )

    # 模型转移到GPU
    model = model.to(device)

    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()

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
    best_loss = float('inf')
    model_dir = os.path.join(cfg.output_dir, cfg.project_name)
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_model.pth")

    logger.info(f"开始第{fold_idx + 1}折训练...")
    start_time = time.time()

    for epoch in range(cfg.num_epochs):
        # 训练
        logger.info(f"Epoch {epoch + 1}/{cfg.num_epochs}")
        train_metrics = train_epoch(model, train_dataloader, criterion, optimizer, device, cfg, logger=logger)

        # 验证
        test_metrics = validate(model, test_dataloader, criterion, device, cfg, logger=logger)

        # 保存loss最小的模型
        if test_metrics['loss'] < best_loss:
            best_loss = test_metrics['loss']
            save_checkpoint(model, optimizer, epoch, test_metrics, best_model_path)
            logger.info(f"新的最佳模型已保存! Loss: {best_loss:.4f}")

    # 训练结束，记录总训练时间
    elapsed_time = time.time() - start_time
    logger.info(f"第{fold_idx + 1}折训练完成，总用时 {elapsed_time / 3600:.2f} 小时")

    # 加载最佳模型并测试
    logger.info(f"加载最佳模型: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)

    # 进行最终测试
    logger.info("进行最终测试评估...")
    final_test_metrics = validate(model, test_dataloader, criterion, device, cfg, logger=logger)

    # 还原项目名称
    cfg.project_name = original_project_name

    return final_test_metrics


def calculate_stats(results):
    """计算多次运行的平均值和标准差"""
    mean_results = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
    std_results = {k: np.std([r[k] for r in results]) for k in results[0].keys()}
    return mean_results, std_results


def main_kfold(cfg):
    """运行K折交叉验证主函数"""
    num_folds = 5
    all_results = []

    # 设置主日志
    main_logger = setup_logger(cfg, log_file_name="main_kfold_log.txt")
    main_logger.info(f"开始{num_folds}折交叉验证...")
    main_logger.info(f"训练模式: {cfg.train_mode}, 模型: {cfg.backbone_type}")

    # 运行每一折
    for fold_idx in range(num_folds):
        main_logger.info(f"======== 开始第{fold_idx + 1}/{num_folds}折 ========")
        fold_results = run_single_fold(cfg, fold_idx, num_folds)
        all_results.append(fold_results)
        main_logger.info(f"第{fold_idx + 1}折完成\n")

    # 计算平均结果和标准差
    mean_results, std_results = calculate_stats(all_results)

    # 输出最终结果
    main_logger.info("======== 最终结果 ========")
    main_logger.info(f"AUC: {mean_results['macro_auc']:.4f} ± {std_results['macro_auc']:.4f}")
    main_logger.info(f"P@R95: {mean_results['p_at_r95']:.4f} ± {std_results['p_at_r95']:.4f}")
    main_logger.info(f"ACC: {mean_results['acc']:.4f} ± {std_results['acc']:.4f}")
    main_logger.info(f"F1 (macro): {mean_results['f1_macro']:.4f} ± {std_results['f1_macro']:.4f}")
    main_logger.info(f"P@R90: {mean_results['p_at_r90']:.4f} ± {std_results['p_at_r90']:.4f}")

    # 创建一个结果表格并保存
    metrics_df = pd.DataFrame({
        'Fold': [f"Fold {i + 1}" for i in range(num_folds)] + ['Mean', 'Std'],
        'AUC': [r['macro_auc'] for r in all_results] + [mean_results['macro_auc'], std_results['macro_auc']],
        'P@R95': [r['p_at_r95'] for r in all_results] + [mean_results['p_at_r95'], std_results['p_at_r95']],
        'ACC': [r['acc'] for r in all_results] + [mean_results['acc'], std_results['acc']],
        'F1': [r['f1_macro'] for r in all_results] + [mean_results['f1_macro'], std_results['f1_macro']],
        'P@R90': [r['p_at_r90'] for r in all_results] + [mean_results['p_at_r90'], std_results['p_at_r90']]
    })

    # 保存结果表格
    results_dir = os.path.join(cfg.output_dir, f"{cfg.project_name}_results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "kfold_results.csv")
    metrics_df.to_csv(results_path, index=False)
    main_logger.info(f"结果已保存至: {results_path}")

    return mean_results, std_results