import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class ImageDataset(Dataset):
    """图像数据集，支持两种初始化方式：
    1. 通过root_dir指定根目录，自动加载所有图像
    2. 通过samples、class_to_idx和classes直接指定，用于交叉验证或自定义划分
    """

    def __init__(self, root_dir=None, transform=None, samples=None,
                 class_to_idx=None, classes=None, is_binary=False,
                 normal_class_name="normal"):
        self.root_dir = root_dir
        self.transform = transform
        self.is_binary = is_binary
        self.normal_class_name = normal_class_name
        self.binary_mapping = None

        # 直接提供了样本列表
        if samples is not None and class_to_idx is not None and classes is not None:
            self.samples = samples
            self.class_to_idx = class_to_idx
            self.classes = classes
            if self.is_binary:
                self.binary_mapping = {cls: 0 if cls == self.normal_class_name else 1
                                       for cls in self.classes}
                if hasattr(self, 'orig_classes'):
                    self.orig_classes = classes
        elif root_dir is not None:
            # 从目录结构加载
            self._load_from_directory()
        else:
            # 空数据集
            self.samples = []
            self.class_to_idx = class_to_idx if class_to_idx is not None else {}
            self.classes = classes if classes is not None else []

    def _load_from_directory(self):
        """从根目录加载所有样本"""
        orig_classes = sorted(os.listdir(self.root_dir))

        if self.is_binary:
            # 二分类模式
            self.binary_mapping = {cls: 0 if cls == self.normal_class_name else 1
                                   for cls in orig_classes}
            self.classes = ["normal", "abnormal"]
            self.class_to_idx = {"normal": 0, "abnormal": 1}
            self.orig_classes = orig_classes
        else:
            # 多分类模式
            self.classes = orig_classes
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 加载所有样本
        self.samples = []
        if self.is_binary:
            # 二分类模式
            for class_name in self.orig_classes:
                class_dir = os.path.join(self.root_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue

                binary_label = self.binary_mapping[class_name]

                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((img_path, binary_label))
        else:
            # 多分类模式
            for class_name in self.classes:
                class_dir = os.path.join(self.root_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"无法读取图片 {img_path}: {str(e)}")
            img = Image.new('RGB', (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def get_class_counts(self):
        """获取每个类别的样本数量"""
        counts = {}
        if self.is_binary:
            for _, label in self.samples:
                class_name = self.classes[label]
                counts[class_name] = counts.get(class_name, 0) + 1
        else:
            for _, label in self.samples:
                if isinstance(label, int) and 0 <= label < len(self.classes):
                    class_name = self.classes[label]
                    counts[class_name] = counts.get(class_name, 0) + 1
                else:
                    counts["unknown"] = counts.get("unknown", 0) + 1
        return counts