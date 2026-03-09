import os
import torch

def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """保存检查点"""
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存检查点
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'metrics': metrics
    }, save_path)

    print(f"模型检查点已保存到 {save_path}")