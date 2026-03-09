import torch
import torch.nn as nn
from timm.models import create_model
from .multi_branch import MultiBranchModel


def build_multi_branch_model(pretrained=True, backbone_type='convnext_tiny',
                             num_classes=8, dropout_rate=0.5, patch_head_hidden_dim=384,
                             fusion_method='learnable', mode='dual_branch',
                             convnext_init="IMAGENET", is_swin=False,
                             feature_branch_path=None, label_branch_path=None,
                             alpha_glb_pool=0.3, transformer_num_layers=2):
    """
    构建多分支模型
    """
    # 检查模式是否有效
    valid_modes = ['label_branch', 'feature_branch', 'dual_branch', 'ada_kl', 'label_branch_enhan',
                   'dual_2conv', 'dual_2conv_49', 'dual_2conv_4914', 'context_aware_branch', 'dual_context_aware',
                   'label_branch_enhan_context']
    if mode not in valid_modes:
        raise ValueError(f"无效的模式: {mode}。有效模式包括: {', '.join(valid_modes)}")

    # 根据模式设置patch数量
    if mode == 'dual_2conv_49':
        num_patches = 49
        print(f"使用 {num_patches} 个特征向量 (7x7格点)")
    elif mode == 'dual_2conv_4914':
        num_patches = 63
        print(f"使用 {num_patches} 个特征向量 (49个原始特征 + 14个多尺度特征)")
    else:
        num_patches = 14
        print(f"使用 {num_patches} 个特征向量 (多尺度池化: 1+4+9)")

    # 1. 构建骨干网络
    if is_swin:
        print(f"使用Swin Transformer (swin_tiny_patch4_window7_224) 作为backbone")
        backbone = create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0
        )
        feature_dim = 768
    else:
        if convnext_init == "IMAGENET":
            backbone = create_model(
                backbone_type,
                pretrained=pretrained,
                num_classes=0,
                global_pool='',
                pretrained_cfg_overlay={"file": "/root/autodl-tmp/sps/pdc1/moe/convnext_tiny_1k_224_ema.pth"}
            )
            print(f"使用ImageNet预训练权重初始化{backbone_type}")
        else:
            print(f"从路径加载ConvNeXt权重: {convnext_init}")
            backbone = create_model(
                backbone_type,
                pretrained=False,
                num_classes=0,
                global_pool=''
            )
            checkpoint = torch.load(convnext_init, map_location='cpu')

            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                backbone_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('feature_extractor.feature_extractor.'):
                        new_key = k.replace('feature_extractor.feature_extractor.', '')
                        backbone_state_dict[new_key] = v
                    elif k.startswith('feature_extractor.'):
                        new_key = k.replace('feature_extractor.', '')
                        backbone_state_dict[new_key] = v

                if len(backbone_state_dict) > 0:
                    missing, unexpected = backbone.load_state_dict(backbone_state_dict, strict=False)
                    print(f"加载backbone权重完成。未加载的参数: {len(missing)}, 意外的参数: {len(unexpected)}")
                else:
                    print("未找到匹配的backbone权重，使用随机初始化")
            else:
                print("未找到model字段，尝试直接加载权重")
                missing, unexpected = backbone.load_state_dict(checkpoint, strict=False)
                print(f"直接加载权重完成。未加载的参数: {len(missing)}, 意外的参数: {len(unexpected)}")

    # 2. 获取骨干网络的特征维度
    try:
        feature_dim = backbone.num_features
    except AttributeError:
        if 'convnext_tiny' in backbone_type:
            feature_dim = 768
        elif 'convnext_small' in backbone_type:
            feature_dim = 768
        elif 'convnext_base' in backbone_type:
            feature_dim = 1024
        elif 'convnext_large' in backbone_type:
            feature_dim = 1536
        elif 'resnet50' in backbone_type:
            feature_dim = 2048
        elif 'resnet101' in backbone_type:
            feature_dim = 2048
        else:
            print(f"尝试推断 {backbone_type} 的特征维度...")
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 224, 224)
                if hasattr(backbone, 'forward_features'):
                    dummy_output = backbone.forward_features(dummy_input)
                    if len(dummy_output.shape) == 4:
                        feature_dim = dummy_output.shape[1]
                    else:
                        feature_dim = dummy_output.shape[-1]
                else:
                    raise ValueError(f"无法确定 backbone '{backbone_type}' 的特征维度。")
        print(f"使用特征维度: {feature_dim} for {backbone_type}")

    # 实例化多分支模型
    model = MultiBranchModel(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_patches=num_patches,
        patch_head_hidden_dim=patch_head_hidden_dim,
        transformer_num_heads=8,
        transformer_num_layers=transformer_num_layers,
        transformer_ff_dim=2048,
        transformer_dropout=0.1,
        attention_dim=256,
        dropout_rate=dropout_rate,
        fusion_method=fusion_method,
        fusion_alpha=0.5,
        alpha_glb_pool=alpha_glb_pool,
        mode=mode
    )

    # 如果是特定模式且提供了预训练模型路径，从预训练模型初始化两个分支
    if mode in ['dual_2conv', 'dual_2conv_49', 'dual_2conv_4914'] and (feature_branch_path or label_branch_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.init_branches_from_pretrained(feature_branch_path, label_branch_path, device)

    return model