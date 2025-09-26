#!/usr/bin/env python3
"""
恢复系统原始参数脚本

此脚本用于恢复之前备份的系统原始参数，将系统恢复到应用任何场景参数之前的状态。
"""

import os
import sys
import logging

# 检查是否作为模块运行还是作为脚本运行
try:
    # 尝试相对导入（作为包的一部分运行）
    from .param_optimizer import ParamOptimizer
except ImportError:
    # 如果失败，尝试绝对导入（直接运行脚本）
    if __name__ == "__main__" and __package__ is None:
        # 添加当前目录到Python路径
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from param_optimizer import ParamOptimizer
    else:
        raise

# 直接设置默认配置文件路径为系统配置目录
# 不使用环境变量路径，避免检测/usr/lib目录下的配置文件
default_config_file = "/etc/industrial-scene-optimizer/service_config.conf"

# 获取配置信息以确定日志文件路径
def _load_config_for_logger(config_path=default_config_file):
    """加载配置文件以获取日志设置"""
    config = {'log_file': '/var/log/industrial-scene-optimizer/service.log', 'log_level': 'INFO'}
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"')
                        if key in ['log_file', 'log_level']:
                            config[key] = value
    except Exception:
        pass
    return config

# 检查是否作为模块运行还是作为脚本运行
try:
    # 尝试相对导入（作为包的一部分运行）
    from .logger_utils import get_logger
except ImportError:
    # 如果失败，尝试绝对导入（直接运行脚本）
    if __name__ == "__main__" and __package__ is None:
        # 已经添加了路径，直接尝试绝对导入
        from logger_utils import get_logger
    else:
        raise

# 使用统一的日志记录器，传入配置以确保使用相同的日志文件
config_for_logger = _load_config_for_logger()
logger = get_logger(config=config_for_logger)

def _load_conf_file(config_path):
    """加载.conf格式配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    config = {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 忽略空行和注释行
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 解析key=value格式
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"')  # 去除可能的引号
                    
                    # 尝试将数字字符串转换为数字
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                        value = float(value)
                    
                    config[key] = value
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
    return config

# 加载配置文件
def load_config():
    """加载配置文件
    
    Returns:
        配置字典
    """
    try:
        if os.path.exists(default_config_file):
            config = _load_conf_file(default_config_file)
            return config
        else:
            logger.warning(f"未找到配置文件: {default_config_file}")
            return {}
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}

# 获取配置
service_config = load_config()

# 根据配置文件更新日志设置
if service_config:
    # 确定日志目录
    log_dir = service_config.get("log_dir", "/var/log/industrial-scene-optimizer")
    # 如果service_config中有log_file配置项，可以从中提取目录
    if "log_file" in service_config:
        log_dir = os.path.dirname(service_config["log_file"])
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 确定日志文件路径
    log_file = os.path.join(log_dir, "restore_original_params.log")
    if "log_file" in service_config:
        # 使用service_config中的日志文件路径，但保持文件名
        log_file = os.path.join(os.path.dirname(service_config["log_file"]), "restore_original_params.log")
    
    # 确保logging模块可用
    import logging
    
    # 更新日志配置
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志配置已更新，日志文件路径: {log_file}")


def restore_original_parameters():
    """恢复系统原始参数"""
    logger.info("开始恢复系统原始参数")
    
    try:
        # 创建参数优化器实例并传入配置
        optimizer = ParamOptimizer(config=service_config)
        
        # 调用恢复方法
        success = optimizer._restore_original_parameters()
        
        if success:
            logger.info("系统原始参数恢复成功")
            return 0
        else:
            logger.error("系统原始参数恢复失败")
            return 1
    except Exception as e:
        logger.error(f"恢复系统原始参数时发生未预期的错误: {e}")
        return 2


if __name__ == "__main__":
    # 检查是否有足够的权限
    if os.name == 'posix' and os.geteuid() != 0:
        logger.error("需要以root权限运行此脚本")
        print("错误: 需要以root权限运行此脚本")
        sys.exit(1)
    
    # 执行恢复操作
    exit_code = restore_original_parameters()
    sys.exit(exit_code)