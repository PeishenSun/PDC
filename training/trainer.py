import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda.amp import autocast, GradScaler
from ..utils.meters import AverageMeter


def train_epoch(model, dataloader, criterion, optimizer, device, cfg, logger=None):
    """训练模型的一个epoch"""
    model.train()
    batch_time = AverageMeter('Batch', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    final_losses = AverageMeter('Final_Loss', ':.4e')
    sparse_losses = AverageMeter('Sparse_Loss', ':.4e')

    # 新增：如果是ada_kl或dual_2conv模式，跟踪KL权重
    if cfg.train_mode in ['ada_kl', 'dual_2conv', 'dual_2conv_49', 'dual_2conv_4914', 'label_branch_enhan_context']:
        kl_weights = AverageMeter('KL_Weight', ':.4e')

    # 收集预测和标签
    all_preds = []
    all_labels = []

    # 创建梯度缩放器用于混合精度训练
    scaler = GradScaler(enabled=cfg.use_amp)

    start = time.time()
    end = time.time()

    # 获取当前epoch和总batches数
    total_batches = len(dataloader) * cfg.num_epochs

    for i, (inputs, labels) in enumerate(dataloader):
        # 如果是ada_kl或dual_2conv模式，更新批次进度
        if cfg.train_mode in ['ada_kl', 'dual_2conv']:
            global_batch_idx = i + len(dataloader) * model.module.current_epoch if hasattr(model,
                                                                                           'module') else i + len(
                dataloader) * model.current_epoch

            # 更新模型的批次进度
            if hasattr(model, 'module'):
                model.module.update_batch_progress(global_batch_idx, len(dataloader), cfg.num_epochs)
            else:
                model.update_batch_progress(global_batch_idx, len(dataloader), cfg.num_epochs)

        # 计算数据加载时间
        data_time.update(time.time() - end)

        # 移动数据到GPU
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 清零梯度
        optimizer.zero_grad(set_to_none=True)

        # 使用混合精度训练
        with autocast(enabled=cfg.use_amp):
            # 根据模式计算损失
            if cfg.train_mode in ['ada_kl', 'dual_2conv', 'dual_2conv_49', 'dual_2conv_4914',
                                  'label_branch_enhan_context']:
                total_loss, final_loss, sparse_loss, outputs, extra_info = model.calculate_loss(
                    inputs, labels, criterion,
                    sparse_loss_weight=cfg.sparse_loss_weight,
                    feature_weight=cfg.feature_weight,
                    label_weight=cfg.label_weight,
                    kl_weight=cfg.kl_weight,
                    kl_ramp_start=cfg.kl_ramp_start,
                    kl_ramp_end=cfg.kl_ramp_end,
                    kl_schedule=cfg.kl_schedule
                )
                # 更新KL权重记录
                kl_weights.update(extra_info['adaptive_kl_weight'], inputs.size(0))
            else:
                total_loss, final_loss, sparse_loss, outputs = model.calculate_loss(
                    inputs, labels, criterion,
                    sparse_loss_weight=cfg.sparse_loss_weight,
                    feature_weight=cfg.feature_weight,
                    label_weight=cfg.label_weight,
                    kl_weight=cfg.kl_weight
                )

        # 使用梯度缩放器进行反向传播和优化
        if cfg.use_amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # 更新统计数据
        losses.update(total_loss.item(), inputs.size(0))
        final_losses.update(final_loss.item(), inputs.size(0))
        sparse_losses.update(sparse_loss.item(), inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # 收集预测结果
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds)
        all_labels.append(labels)

        # 日志记录
        if i % cfg.print_freq == 0:
            log_message = ('Batch: [{0}/{1}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Final_Loss {final_loss.val:.4f} ({final_loss.avg:.4f})\t'
                           'Sparse_Loss {sparse_loss.val:.4f} ({sparse_loss.avg:.4f})'.format(
                i, len(dataloader), batch_time=batch_time,
                data_time=data_time, loss=losses, final_loss=final_losses,
                sparse_loss=sparse_losses))

            # 如果是ada_kl或dual_2conv模式，添加KL权重日志
            if cfg.train_mode in ['ada_kl', 'dual_2conv']:
                log_message += '\tKL_Weight {kl_weight.val:.6f} ({kl_weight.avg:.6f})'.format(
                    kl_weight=kl_weights)

            if logger:
                logger.info(log_message)
            else:
                print(log_message)

    # 连接所有批次的预测和标签
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    metrics = {
        'loss': losses.avg,
        'final_loss': final_losses.avg,
        'sparse_loss': sparse_losses.avg,
        'acc': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'time': time.time() - start
    }

    # 如果是ada_kl或dual_2conv模式，添加KL权重到度量中
    if cfg.train_mode in ['ada_kl', 'dual_2conv', 'label_branch_enhan_context']:
        metrics['kl_weight'] = kl_weights.avg

    return metrics


def validate(model, dataloader, criterion, device, cfg, logger=None):
    """验证/测试模型"""
    model.eval()
    batch_time = AverageMeter('Batch', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    final_losses = AverageMeter('Final_Loss', ':.4e')
    sparse_losses = AverageMeter('Sparse_Loss', ':.4e')

    # 收集预测和标签
    all_preds = []
    all_labels = []
    all_label_preds = []
    all_feature_preds = []
    all_scores = []

    start = time.time()
    end = time.time()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=cfg.use_amp):
                # 根据模式计算损失
                if cfg.train_mode in ['ada_kl', 'dual_2conv', 'dual_2conv_49', 'dual_2conv_4914',
                                      'label_branch_enhan_context']:
                    total_loss, final_loss, sparse_loss, outputs, extra_info = model.calculate_loss(
                        inputs, labels, criterion,
                        sparse_loss_weight=cfg.sparse_loss_weight,
                        feature_weight=cfg.feature_weight,
                        label_weight=cfg.label_weight,
                        kl_weight=cfg.kl_weight,
                        kl_ramp_start=cfg.kl_ramp_start,
                        kl_ramp_end=cfg.kl_ramp_end,
                        kl_schedule=cfg.kl_schedule
                    )
                else:
                    total_loss, final_loss, sparse_loss, outputs = model.calculate_loss(
                        inputs, labels, criterion,
                        sparse_loss_weight=cfg.sparse_loss_weight,
                        feature_weight=cfg.feature_weight,
                        label_weight=cfg.label_weight,
                        kl_weight=cfg.kl_weight
                    )

            # 更新统计数据
            losses.update(total_loss.item(), inputs.size(0))
            final_losses.update(final_loss.item(), inputs.size(0))
            sparse_losses.update(sparse_loss.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            # 获取预测概率
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # 收集结果
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_scores.append(probs.cpu())

            # 获取分支预测（如果适用）
            if cfg.train_mode in ['dual_branch', 'ada_kl', 'dual_2conv', 'dual_2conv_49', 'dual_2conv_4914']:
                final_preds, label_preds, feature_preds, _, _ = model(inputs)

                if label_preds is not None:
                    all_label_preds.append(torch.argmax(label_preds, dim=1).cpu())

                if feature_preds is not None:
                    all_feature_preds.append(torch.argmax(feature_preds, dim=1).cpu())

    # 合并预测并移至CPU计算指标
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_scores = torch.cat(all_scores).numpy()

    # 计算融合模型指标
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    # 计算P@R95和P@R90
    from ..training.metrics import compute_precision_at_recall, compute_macro_auc
    p_at_r95 = compute_precision_at_recall(all_labels, all_scores, recall_level=0.95)
    p_at_r90 = compute_precision_at_recall(all_labels, all_scores, recall_level=0.90)

    # 计算macro AUC
    num_classes = len(np.unique(all_labels))
    macro_auc = compute_macro_auc(all_labels, all_scores, num_classes)

    metrics = {
        'loss': losses.avg,
        'final_loss': final_losses.avg,
        'sparse_loss': sparse_losses.avg,
        'acc': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'p_at_r95': p_at_r95,
        'p_at_r90': p_at_r90,
        'macro_auc': macro_auc,
        'time': time.time() - start
    }

    # 根据模型模式添加分支指标
    if cfg.train_mode in ['label_branch', 'dual_branch', 'ada_kl', 'label_branch_enhan', 'dual_2conv'] and len(
            all_label_preds) > 0:
        all_label_preds = torch.cat(all_label_preds).numpy()
        metrics['label_acc'] = accuracy_score(all_labels, all_label_preds)
        metrics['label_f1_macro'] = f1_score(all_labels, all_label_preds, average='macro')

    if cfg.train_mode in ['feature_branch', 'dual_branch', 'ada_kl', 'dual_2conv'] and len(all_feature_preds) > 0:
        all_feature_preds = torch.cat(all_feature_preds).numpy()
        metrics['feature_acc'] = accuracy_score(all_labels, all_feature_preds)
        metrics['feature_f1_macro'] = f1_score(all_labels, all_feature_preds, average='macro')

    return metrics