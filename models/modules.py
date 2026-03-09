import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from timm.models.layers import trunc_normal_


class ProgressiveCNNClassifier(nn.Module):
    def __init__(self, feature_dim=768, num_classes=8, dropout_rate=0.5, alpha_glb_pool=0.3, num_patches=14):
        super().__init__()

        self.alpha_glb_pool = alpha_glb_pool
        self.num_patches = num_patches

        # 渐进式特征转换路径
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(96, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 区域分类头
        self.region_classifier = nn.Linear(32, num_classes)

        # 全局特征增强
        self.global_enhance = nn.Sequential(
            nn.Linear(feature_dim, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Linear(192, 32),
            nn.BatchNorm1d(32),
        )

        # 全局分类头
        self.global_classifier = nn.Linear(32, num_classes)

        # 最终聚合器
        self.aggregator = nn.Sequential(
            nn.Linear(num_classes * num_patches, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def l1_regularizer(self, region_scores):
        """L1稀疏正则化"""
        batch_size = region_scores.size(0)
        if batch_size > 0:
            l1_loss = torch.abs(region_scores).sum()
            return l1_loss / batch_size
        else:
            return torch.tensor(0.0, device=region_scores.device)

    def l2_regularizer(self, region_scores):
        """L2稀疏正则化"""
        batch_size = region_scores.size(0)
        if batch_size > 0:
            l2_loss = torch.norm(region_scores, p=2)
            return l2_loss / batch_size
        else:
            return torch.tensor(0.0, device=region_scores.device)

    def forward(self, features, apply_regularization=True):
        batch_size, num_patches, feature_dim = features.shape

        # 特殊处理全局特征
        if num_patches == 49:
            global_feat = torch.mean(features, dim=1)
        else:
            global_feat = features[:, 0, :]

        global_enhanced = self.global_enhance(global_feat)
        global_scores = self.global_classifier(global_enhanced)

        # 处理所有区域特征
        all_region_features = []
        for i in range(num_patches):
            region_feat = features[:, i, :]
            transformed = self.feature_transform(region_feat)
            all_region_features.append(transformed)

        # 获取每个区域的类别分数
        region_scores = []
        for i in range(num_patches):
            score = self.region_classifier(all_region_features[i])
            region_scores.append(score)

        # 堆叠所有区域分数
        region_scores = torch.stack(region_scores, dim=1)

        # 展平区域分数用于聚合
        flattened_scores = region_scores.reshape(batch_size, -1)

        # 通过聚合器获得最终分类结果
        final_output = self.aggregator(flattened_scores)

        # 加入全局特征的影响
        alpha = self.alpha_glb_pool
        final_output = (1 - alpha) * final_output + alpha * global_scores

        return final_output, region_scores


class ClassifierNetwork(nn.Module):
    def __init__(self, num_classes, patches, dp_rate=0.5):
        super().__init__()
        input_dim = num_classes * patches
        self.cls_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(p=dp_rate),
            nn.Linear(input_dim, num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.cls_head(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        from ..utils.meters import initialize_weights
        self.apply(initialize_weights)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(context)
        return output


class FeedForward(nn.Module):
    def __init__(self, embed_dim=768, ff_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        from ..utils.meters import initialize_weights
        self.apply(initialize_weights)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, ff_dim=2048, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attn = SelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        x = residual + self.dropout(x)
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=2, ff_dim=2048,
                 max_seq_length=100, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length, dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class GatedAttention(nn.Module):
    def __init__(self, feature_dim=512, attention_dim=128, num_heads=1):
        super(GatedAttention, self).__init__()
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.attention_V = nn.Sequential(
            nn.Linear(self.feature_dim, self.attention_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.feature_dim, self.attention_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.attention_dim, self.num_heads)

        from ..utils.meters import initialize_weights
        self.apply(initialize_weights)

    def forward(self, x, apply_softmax=True):
        attention_V = self.attention_V(x)
        attention_U = self.attention_U(x)
        attention = self.attention_weights(attention_V * attention_U).transpose(1, 2)
        if apply_softmax:
            attention = F.softmax(attention, dim=2)
        return attention


class MultiLayerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[384, 128], num_classes=8, dropout_rate=0.5):
        super(MultiLayerClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

        from ..utils.meters import initialize_weights
        self.apply(initialize_weights)

    def forward(self, x):
        return self.mlp(x)


class MultiScaleClassifierHeads(nn.Module):
    """
    为不同尺度的特征设计单独的分类头
    """

    def __init__(self, feature_dim=768, num_classes=8, hidden_dim=384, dropout_rate=0.5):
        super(MultiScaleClassifierHeads, self).__init__()

        # 1. 全局特征的分类头
        self.gap_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

        # 2. 5x5窗口特征的分类头
        self.window5x5_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

        # 3. 3x3窗口特征的分类头
        self.window3x3_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

        from ..utils.meters import initialize_weights
        self.apply(initialize_weights)

    def forward(self, multi_scale_features):
        batch_size = multi_scale_features['gap'].shape[0]

        # 应用各尺度分类头
        gap_scores = self.gap_classifier(
            multi_scale_features['gap'].reshape(-1, multi_scale_features['gap'].shape[-1]))
        gap_scores = gap_scores.view(batch_size, -1, gap_scores.shape[-1])

        window5x5_scores = self.window5x5_classifier(
            multi_scale_features['window5x5'].reshape(-1, multi_scale_features['window5x5'].shape[-1]))
        window5x5_scores = window5x5_scores.view(batch_size, -1, window5x5_scores.shape[-1])

        window3x3_scores = self.window3x3_classifier(
            multi_scale_features['window3x3'].reshape(-1, multi_scale_features['window3x3'].shape[-1]))
        window3x3_scores = window3x3_scores.view(batch_size, -1, window3x3_scores.shape[-1])

        # 合并所有尺度的分数
        all_scores = torch.cat([
            gap_scores,
            window5x5_scores,
            window3x3_scores
        ], dim=1)

        return all_scores, {
            'gap': gap_scores,
            'window5x5': window5x5_scores,
            'window3x3': window3x3_scores
        }


class MultiScaleFeatureExtractor(nn.Module):
    """
    多尺度特征提取器 - 从ConvNeXt中提取不同尺度的特征
    """

    def __init__(self, backbone):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.feature_extractor = backbone

        # 检查backbone类型
        if hasattr(backbone, 'forward_features'):
            self._forward = self._forward_features_timm
        else:
            raise ValueError("不支持的backbone类型，必须支持forward_features方法")

    def _forward_features_timm(self, x):
        features = self.feature_extractor.forward_features(x)
        return features

    def extract_flat_features(self, x):
        """
        提取扁平化的49个特征向量 (7x7网格)
        """
        batch_size = x.shape[0]
        features = self._forward_features_timm(x)
        feat_dim = features.shape[1]

        # 直接将特征图重塑为 49 个特征向量
        flat_features = features.permute(0, 2, 3, 1).reshape(batch_size, -1, feat_dim)

        return flat_features

    def extract_multi_scale_features(self, x):
        """
        提取多尺度特征: GAP, 5x5窗口, 3x3窗口
        """
        batch_size = x.shape[0]
        features = self._forward_features_timm(x)
        feat_dim = features.shape[1]

        # 1. 全局平均池化 (GAP)
        gap_feature = F.adaptive_avg_pool2d(features, (1, 1))
        gap_feature = gap_feature.reshape(batch_size, 1, feat_dim)

        # 2. 使用5x5窗口
        pooled_features = F.adaptive_avg_pool2d(features, (2, 2))
        window5x5 = pooled_features.permute(0, 2, 3, 1).reshape(batch_size, 4, feat_dim)

        # 3. 使用3x3窗口
        pooled_features = F.adaptive_avg_pool2d(features, (3, 3))
        window3x3 = pooled_features.permute(0, 2, 3, 1).reshape(batch_size, 9, feat_dim)

        return {
            'gap': gap_feature,
            'window5x5': window5x5,
            'window3x3': window3x3,
        }

    def forward(self, x):
        """
        前向传播，提取并合并多尺度特征
        """
        multi_scale_features = self.extract_multi_scale_features(x)

        # 合并所有尺度的特征
        features = torch.cat([
            multi_scale_features['gap'],
            multi_scale_features['window5x5'],
            multi_scale_features['window3x3']
        ], dim=1)

        return features, multi_scale_features

    def extract_multi_scale_features_with_raw(self, x):
        """
        提取多尺度特征: GAP, 5x5窗口, 3x3窗口，以及原始的49个特征
        """
        batch_size = x.shape[0]
        features = self._forward_features_timm(x)
        feat_dim = features.shape[1]

        # 1. 提取原始的49个特征向量
        flat_features = features.permute(0, 2, 3, 1).reshape(batch_size, -1, feat_dim)

        # 2. 全局平均池化 (GAP)
        gap_feature = F.adaptive_avg_pool2d(features, (1, 1))
        gap_feature = gap_feature.reshape(batch_size, 1, feat_dim)

        # 3. 使用5x5窗口
        pooled_features = F.adaptive_avg_pool2d(features, (2, 2))
        window5x5 = pooled_features.permute(0, 2, 3, 1).reshape(batch_size, 4, feat_dim)

        # 4. 使用3x3窗口
        pooled_features = F.adaptive_avg_pool2d(features, (3, 3))
        window3x3 = pooled_features.permute(0, 2, 3, 1).reshape(batch_size, 9, feat_dim)

        return {
            'raw': flat_features,
            'gap': gap_feature,
            'window5x5': window5x5,
            'window3x3': window3x3
        }

    def forward_with_all_features(self, x):
        """
        前向传播，提取并合并多尺度特征，包括原始49个特征
        """
        multi_scale_features = self.extract_multi_scale_features_with_raw(x)

        # 合并所有尺度的特征
        features = torch.cat([
            multi_scale_features['raw'],
            multi_scale_features['gap'],
            multi_scale_features['window5x5'],
            multi_scale_features['window3x3']
        ], dim=1)

        return features, multi_scale_features