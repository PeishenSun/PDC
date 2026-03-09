import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多分支模型训练与测试')

    # 基本参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--train_data_path', type=str, default="/root/autodl-tmp/sps/cqubpdd/train",
                        help='训练集路径')
    parser.add_argument('--val_data_path', type=str, default="/root/autodl-tmp/sps/cqubpdd/val", help='验证集路径')
    parser.add_argument('--test_data_path', type=str, default="/root/autodl-tmp/sps/cqubpdd/test",
                        help='测试集路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--use_amp', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')

    # 模型参数
    parser.add_argument('--train_mode', type=str, default='label_branch_enhan',
                        choices=['label_branch', 'feature_branch', 'dual_branch', 'ada_kl',
                                 'label_branch_enhan', 'dual_2conv', 'dual_2conv_49',
                                 'dual_2conv_4914', 'context_aware_branch', 'dual_context_aware',
                                 'label_branch_enhan_context'],
                        help='训练模式')

    # 损失函数权重
    parser.add_argument('--sparse_loss_weight', type=float, default=1e-3, help='L1稀疏损失权重')
    parser.add_argument('--label_weight', type=float, default=0.001, help='标签分支损失权重')
    parser.add_argument('--feature_weight', type=float, default=0.999, help='特征分支损失权重')
    parser.add_argument('--kl_weight', type=float, default=0.7, help='KL散度损失权重')

    # 自适应KL参数
    parser.add_argument('--kl_ramp_start', type=float, default=0.1, help='KL开始增长的训练进度比例')
    parser.add_argument('--kl_ramp_end', type=float, default=0.8, help='KL达到最大值的训练进度比例')
    parser.add_argument('--kl_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine', 'step', 'exp'],
                        help='KL增长调度')

    # 其他参数
    parser.add_argument('--regularization_type', type=str, default='l2', choices=['l1', 'l2'], help='正则化类型')
    parser.add_argument('--gpu_id', type=int, default=1, help='使用的GPU ID')
    parser.add_argument('--output_dir', type=str, default="/root/autodl-tmp/sps/pdc1/moe/output",
                        help='输出目录')
    parser.add_argument('--project_name', type=str, default="fold1", help='项目名称')
    parser.add_argument('--normal_class_ratio', type=float, default=1, help='正常类别样本使用比率')
    parser.add_argument('--normal_class_name', type=str, default="normal", help='正常类别的名称')

    # 交叉验证参数
    parser.add_argument('--use_cross_validation', action='store_true', default=True, help='是否使用交叉验证')
    parser.add_argument('--n_folds', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--current_fold', type=int, default=1, help='当前使用的折号(0-4)')
    parser.add_argument('--test_after_training', action='store_true', default=True, help='是否在训练后测试模型')
    parser.add_argument('--alpha_glb_pool', type=float, default=0.3, help='label分支超参数')
    parser.add_argument('--transformer_num_layers', type=int, default=2, help='几层transformer解码器层')
    parser.add_argument('--metrics_for_harmonic_mean', nargs='+',
                        default=['acc', 'f1_macro', 'p_at_r95', 'p_at_r90'],
                        help='用于计算调和平均的指标列表')

    # 二分类参数
    parser.add_argument('--is_binary', action='store_true', help='是否启用二分类模式')
    parser.add_argument('--abnormal_class_name', type=str, default="defect", help='二分类模式下异常类别的显示名称')

    # 裂缝数据集参数
    parser.add_argument('--crack_path', type=str, default=None, help='裂缝数据集路径')
    parser.add_argument('--split_ratio', type=str, default='0.5,0.5',
                        help='crack数据集上训练集、测试集的划分比例')

    return parser.parse_args()