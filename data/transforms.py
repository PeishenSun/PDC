from torchvision import transforms

# ImageNet均值和标准差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_data_transforms(cfg):
    """数据增强变换"""
    train_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ], p=cfg.colorjitter_prob),
        transforms.RandomHorizontalFlip(p=cfg.randhoriflip_prob),
        transforms.RandomVerticalFlip(p=cfg.randvertiflip_prob),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=cfg.randaffine_degrees)
        ], p=cfg.randaffine_prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # 测试时只需要调整大小和标准化
    test_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, test_transform