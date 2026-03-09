import os
import sys
import datetime
import argparse

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from config.arguments import parse_args
from experiments.main import main
from experiments.kfold_cv import main_kfold
from experiments.testing import test_model

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 创建配置对象并更新参数
    cfg = Config()

    # 更新配置参数
    cfg.seed = args.seed
    cfg.train_data_path = args.train_data_path
    cfg.val_data_path = args.val_data_path
    cfg.test_data_path = args.test_data_path
    cfg.batch_size = args.batch_size
    cfg.num_epochs = args.num_epochs
    cfg.use_amp = args.use_amp
    cfg.learning_rate = args.learning_rate
    cfg.train_mode = args.train_mode
    cfg.sparse_loss_weight = args.sparse_loss_weight
    cfg.label_weight = args.label_weight
    cfg.feature_weight = args.feature_weight
    cfg.kl_weight = args.kl_weight
    cfg.kl_ramp_start = args.kl_ramp_start
    cfg.kl_ramp_end = args.kl_ramp_end
    cfg.kl_schedule = args.kl_schedule
    cfg.regularization_type = args.regularization_type
    cfg.gpu_id = args.gpu_id
    cfg.output_dir = args.output_dir
    cfg.project_name = args.project_name
    cfg.normal_class_ratio = args.normal_class_ratio
    cfg.normal_class_name = args.normal_class_name
    cfg.use_cross_validation = args.use_cross_validation
    cfg.n_folds = args.n_folds
    cfg.current_fold = args.current_fold
    cfg.test_after_training = args.test_after_training
    cfg.alpha_glb_pool = args.alpha_glb_pool
    cfg.transformer_num_layers = args.transformer_num_layers

    if isinstance(args.metrics_for_harmonic_mean, list):
        cfg.metrics_for_harmonic_mean = args.metrics_for_harmonic_mean
    else:
        cfg.metrics_for_harmonic_mean = [args.metrics_for_harmonic_mean]

    # 更新二分类相关参数
    cfg.is_binary = args.is_binary
    cfg.normal_class_name = args.normal_class_name
    cfg.abnormal_class_name = args.abnormal_class_name

    # 添加二分类信息到项目名称
    if cfg.is_binary:
        binary_info = "_binary"
    else:
        binary_info = ""

    # 添加时间戳、训练模式和交叉验证信息到项目名称
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # 添加交叉验证信息到项目名称
    if cfg.use_cross_validation:
        cv_info = f"_fold{cfg.current_fold + 1}of{cfg.n_folds}"
    else:
        cv_info = ""

    # 修改项目名称，包含二分类信息
    cfg.project_name = f"{cfg.project_name}_{cfg.train_mode}_{cfg.regularization_type}{cv_info}{binary_info}_{timestamp}"

    os.makedirs(os.path.join(cfg.output_dir, cfg.project_name), exist_ok=True)

    print(f"开始训练{cfg.train_mode}模式，使用{cfg.regularization_type}正则化，时间戳: {timestamp}")
    if cfg.use_cross_validation:
        print(f"交叉验证: 第 {cfg.current_fold + 1}/{cfg.n_folds} 折")
    print(f"稀疏损失权重: {cfg.sparse_loss_weight}")
    print(f"AMP（自动混合精度）训练: {'启用' if cfg.use_amp else '禁用'}")

    # 如果启用二分类，输出相关信息
    if cfg.is_binary:
        print(f"启用二分类模式，正常类别：{cfg.normal_class_name}，其他类别归为：{cfg.abnormal_class_name}")

    # 裂缝数据集相关参数
    cfg.crack_path = args.crack_path
    if args.split_ratio:
        cfg.split_ratio = [float(x) for x in args.split_ratio.split(',')]

    if cfg.crack_path:
        # 使用裂缝数据集进行K折交叉验证
        main_kfold(cfg)
    else:
        # 使用标准数据集
        main(cfg)