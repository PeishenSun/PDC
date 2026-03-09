import torch
import torch.nn as nn
from timm.models import create_model


def create_backbone(backbone_type='convnext_tiny', pretrained=True, convnext_init="IMAGENET"):
    """
    创建骨干网络
    """
    if convnext_init == "IMAGENET":
        backbone = create_model(
            backbone_type,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头
            global_pool=''  # 移除全局池化
        )
    else:
        # 从指定路径加载权重
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

    return backbone