import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np


class MultiBranchModel(nn.Module):
    """
    多分支模型 - 支持标签分支、特征分支、双分支模式、增强标签分支、dual_2conv和dual_2conv_49模式
    """

    def __init__(self, backbone, num_classes=8, feature_dim=768, num_patches=14,
                 patch_head_hidden_dim=384, transformer_num_heads=8,
                 transformer_num_layers=2, transformer_ff_dim=2048,
                 transformer_dropout=0.1, attention_dim=256, dropout_rate=0.5,
                 fusion_method='learnable', fusion_alpha=0.5, mode='dual_branch',
                 alpha_glb_pool=0.3):
        super(MultiBranchModel, self).__init__()

        self.alpha_glb_pool = alpha_glb_pool
        self.transformer_num_layers = transformer_num_layers
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.mode = mode

        # 根据模式设置特征数量
        if mode == 'dual_2conv_4914':
            self.num_patches = 63  # 49 + 14 = 63
        else:
            self.num_patches = num_patches

        # 更新进度跟踪
        self.current_batch = 0
        self.total_batches = 1
        self.batches_per_epoch = 1
        self.current_epoch = 0
        self.num_epochs = 1

        # 特征提取器初始化
        if mode in ['dual_2conv', 'dual_2conv_49', 'dual_2conv_4914']:
            # 特征分支的特征提取器
            from .modules import MultiScaleFeatureExtractor
            self.feature_extractor_for_feature_branch = MultiScaleFeatureExtractor(backbone)

            # 为标签分支创建一个新的backbone副本
            backbone_copy = copy.deepcopy(backbone)
            self.feature_extractor_for_label_branch = MultiScaleFeatureExtractor(backbone_copy)
        else:
            # 原有模式使用单个特征提取器
            from .modules import MultiScaleFeatureExtractor
            self.feature_extractor = MultiScaleFeatureExtractor(backbone)

        # 分支1: 标签空间分支
        from .modules import MultiScaleClassifierHeads, ClassifierNetwork, ProgressiveCNNClassifier

        # 1. 多尺度分类头
        self.multi_scale_classifier_heads = MultiScaleClassifierHeads(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=patch_head_hidden_dim,
            dropout_rate=dropout_rate
        )

        # 2. 标签聚合器
        self.label_aggregator_mlp = ClassifierNetwork(
            num_classes=num_classes,
            patches=num_patches,
            dp_rate=dropout_rate
        )

        # 分支2: 特征空间分支
        max_seq_length = 100 if mode == 'dual_2conv_4914' else (num_patches + 1)

        from .modules import TransformerEncoder, GatedAttention, MultiLayerClassifier

        # 特征空间分支的Transformer
        self.transformer = TransformerEncoder(
            embed_dim=feature_dim,
            num_heads=transformer_num_heads,
            num_layers=transformer_num_layers,
            ff_dim=transformer_ff_dim,
            max_seq_length=max_seq_length,
            dropout=transformer_dropout
        )

        self.attention = GatedAttention(
            feature_dim=feature_dim,
            attention_dim=attention_dim,
            num_heads=1
        )

        self.feature_classifier = MultiLayerClassifier(
            input_dim=feature_dim,
            hidden_dims=[384, 128],
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )

        # 增强标签分支的渐进式CNN分类器
        if mode in ['label_branch_enhan', 'dual_2conv', 'dual_2conv_49']:
            self.progressive_classifier = ProgressiveCNNClassifier(
                feature_dim=feature_dim,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                alpha_glb_pool=alpha_glb_pool,
                num_patches=num_patches
            )

        # 对于dual_2conv_4914模式
        if mode == 'dual_2conv_4914':
            self.progressive_classifier = ProgressiveCNNClassifier(
                feature_dim=feature_dim,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                alpha_glb_pool=alpha_glb_pool,
                num_patches=63
            )

        # 融合参数
        self.fusion_method = fusion_method
        self.fusion_alpha = fusion_alpha
        if fusion_method == 'learnable':
            self.branch_weights = nn.Parameter(torch.ones(2))

    # ... 其他方法 forward, calculate_loss, forward_label_branch, forward_feature_branch,
    # get_adaptive_kl_weight, init_branches_from_pretrained 等
    # 由于篇幅限制，这里只展示类结构，具体方法实现请参考原文