"""
日志工具模块 - 提供统一的日志配置管理
"""

import os
import logging
import sys
from typing import Optional, Dict, Any

# 默认日志配置
DEFAULT_LOG_CONFIG = {
    'log_file': '/var/log/industrial-scene-optimizer/service.log',
    'log_level': 'INFO'
}

# 主日志记录器名称
MAIN_LOGGER_NAME = 'industrial_scene_optimizer'

# 已初始化的日志记录器缓存
_loggers: Dict[str, logging.Logger] = {}


def get_logger(logger_name: str = MAIN_LOGGER_NAME, config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    获取或创建日志记录器
    
    Args:
        logger_name: 日志记录器名称
        config: 日志配置字典，包含log_file和log_level
        
    Returns:
        配置好的日志记录器实例
    """
    # 检查缓存中是否已有此记录器
    if logger_name in _loggers:
        return _loggers[logger_name]
    
    # 合并配置
    merged_config = DEFAULT_LOG_CONFIG.copy()
    if config:
        merged_config.update(config)
    
    # 获取或创建日志记录器
    logger = logging.getLogger(logger_name)
    logger.propagate = False  # 防止日志传播到父记录器
    
    # 设置日志级别
    log_level = getattr(logging, merged_config['log_level'], logging.INFO)
    logger.setLevel(log_level)
    
    # 清除已有的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 确保日志目录存在
    log_dir = os.path.dirname(merged_config['log_file'])
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            print(f"无法创建日志目录: {e}", file=sys.stderr)
    
    # 创建文件处理器
    try:
        file_handler = logging.FileHandler(merged_config['log_file'], encoding='utf-8')
        file_handler.setLevel(log_level)
    except Exception as e:
        file_handler = None
        print(f"无法创建日志文件处理器: {e}", file=sys.stderr)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 添加处理器
    if file_handler:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 缓存日志记录器
    _loggers[logger_name] = logger
    
    return logger


def init_main_logger(config: Dict[str, Any]) -> logging.Logger:
    """
    初始化主日志记录器
    
    Args:
        config: 包含log_file和log_level的配置字典
        
    Returns:
        配置好的主日志记录器实例
    """
    return get_logger(MAIN_LOGGER_NAME, config)