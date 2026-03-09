import numpy as np

class NormalClassSampler:
    """
    自定义采样器，用于控制正常类的样本比例
    只在 dual_2conv 模式下使用
    """

    def __init__(self, dataset, normal_class_name, normal_class_ratio=1.0):
        self.dataset = dataset
        self.normal_class_name = normal_class_name
        self.normal_class_ratio = normal_class_ratio

        # 获取正常类的索引
        self.normal_class_idx = self.dataset.class_to_idx.get(normal_class_name)

        if self.normal_class_idx is None:
            print(f"警告: 未找到名为 '{normal_class_name}' 的正常类，使用全部样本")
            self.indices = list(range(len(dataset)))
            return

        # 分离正常类和异常类样本的索引
        self.normal_indices = []
        self.abnormal_indices = []

        for idx, (_, label) in enumerate(self.dataset.samples):
            if label == self.normal_class_idx:
                self.normal_indices.append(idx)
            else:
                self.abnormal_indices.append(idx)

        # 对正常类样本进行采样
        if 0.0 < normal_class_ratio < 1.0:
            num_normal_to_use = int(len(self.normal_indices) * normal_class_ratio)
            np.random.shuffle(self.normal_indices)
            self.normal_indices = self.normal_indices[:num_normal_to_use]

        # 合并索引
        self.indices = self.abnormal_indices + self.normal_indices

        print(f"采样后的数据集统计: 总样本数 {len(self.indices)}, "
              f"正常类样本数 {len(self.normal_indices)} ({len(self.normal_indices) / len(self.indices):.1%}), "
              f"异常类样本数 {len(self.abnormal_indices)} ({len(self.abnormal_indices) / len(self.indices):.1%})")

    def __iter__(self):
        np.random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)