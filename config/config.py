class Config:
    def __init__(self):
        # 随机种子
        self.seed = 42
        self.deterministic = False
        self.metrics_for_harmonic_mean = ['acc', 'f1_macro', 'p_at_r95', 'p_at_r90', 'macro_auc']
        self.print_freq = 20

        # 数据集路径
        self.train_data_path = "/root/sps/cqubpdd/train"
        self.val_data_path = "/root/sps/cqubpdd/val"
        self.test_data_path = "/root/sps/cqubpdd/test"

        # 图像大小
        self.image_size = 224

        # 数据增强参数
        self.colorjitter_prob = 0.5
        self.randhoriflip_prob = 0.5
        self.randvertiflip_prob = 0.5
        self.randaffine_degrees = 15
        self.randaffine_prob = 0.5

        # 训练参数
        self.num_epochs = 30
        self.num_workers = 8
        self.batch_size = 32
        self.use_amp = True

        # 优化器参数
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5

        # 模型参数
        self.backbone_type = 'convnext_tiny'
        self.use_pretrained = True
        self.patch_head_hidden_dim = 384
        self.dropout_rate = 0.5

        # 训练模式和融合设置
        self.train_mode = 'label_branch_enhan'
        self.fusion_method = 'learnable'

        # 损失函数权重
        self.sparse_loss_weight = 1e-3
        self.label_weight = 0.001
        self.feature_weight = 0.999
        self.kl_weight = 0.7

        # 自适应KL的参数
        self.kl_ramp_start = 0.1
        self.kl_ramp_end = 0.8
        self.kl_schedule = 'cosine'

        # 正则化设置
        self.regularization_type = 'l2'
        self.convnext_init = "IMAGENET"
        self.is_swin = False

        # dual_2conv模式的预训练模型路径
        self.feature_branch_path = None
        self.label_branch_path = None

        self.transformer_num_layers = 2

        # GPU
        self.gpu_id = 1
        self.alpha_glb_pool = 0.3

        # 保存路径
        self.output_dir = "/root/autodl-tmp/sps/pdc1/moe/output"
        self.project_name = "fold1"
        self.normal_class_ratio = 1

        # 交叉验证设置
        self.use_cross_validation = True
        self.n_folds = 5
        self.current_fold = 1
        self.test_after_training = True

        # 二分类相关参数
        self.is_binary = False
        self.normal_class_name = "normal"
        self.abnormal_class_name = "abnormal"

        # 裂缝数据集相关
        self.crack_path = None
        self.split_ratio = [0.5, 0.5]