import os
import logging

def setup_logger(cfg, log_file_name="train_log.txt"):
    """设置日志记录"""
    # 创建日志目录
    log_dir = os.path.join(cfg.output_dir, cfg.project_name)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_file_name)

    # 配置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 清除现有的handlers以避免重复日志
    if logger.handlers:
        logger.handlers = []

    # 文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_config(logger, cfg):
    """记录配置参数"""
    logger.info("========== 配置信息 ==========")
    for key, value in vars(cfg).items():
        logger.info(f"{key}: {value}")
    logger.info("==============================")