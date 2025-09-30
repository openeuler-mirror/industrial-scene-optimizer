# -*- coding: utf-8 -*-
"""参数优化器 - 对场景识别结果中配置进行设置"""

import yaml
import time
import subprocess
import os
import json
import re
import sys
import logging
# 延迟导入SceneRecognizer以避免循环依赖

# 确保可以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入日志工具
from logger_utils import get_logger

# 使用统一的日志记录器
logger = get_logger()

# 全局配置变量，初始为空
service_config = {}

# 加载配置文件
def load_config(config_file=None):
    """加载.conf格式配置文件
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        if config_file and os.path.exists(config_file):
            config = {}
            with open(config_file, 'r', encoding='utf-8') as f:
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
            logger.info(f"成功加载配置文件: {config_file}")
            return config
        elif config_file:
            logger.warning(f"配置文件不存在: {config_file}")
        return {}
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}

# 初始化全局配置的函数
def init_global_config():
    """初始化全局配置"""
    global service_config
    
    # 首先从环境变量获取配置文件路径
    default_config_file = os.environ.get('SERVICE_CONFIG_PATH', None)
    
    # 如果环境变量未设置，检查系统配置目录
    if default_config_file is None or not os.path.exists(default_config_file):
        # 首先检查/etc目录下的标准配置位置
        system_etc_config_path = "/etc/industrial-scene-optimizer/service_config.conf"
        if os.path.exists(system_etc_config_path):
            default_config_file = system_etc_config_path
        else:
            # 然后检查脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            system_config_path = os.path.join(script_dir, "service_config.conf")
            if os.path.exists(system_config_path):
                default_config_file = system_config_path
            else:
                # 最后回退到脚本所在目录
                default_config_file = system_etc_config_path  # 使用标准路径作为回退，即使它不存在
    
    # 加载配置文件
    service_config = load_config(default_config_file)

# 在ParamOptimizer类中会根据传入的config参数设置日志
# 这里不再在模块级别更新日志配置，避免不必要的配置文件加载

# 初始化全局配置，但不要在模块导入时执行
# 这将在需要使用全局配置时手动调用
# init_global_config()

class ParamOptimizer:
    """参数优化器 - 对场景识别结果中配置进行设置"""
    
    def __init__(self, templates_dir=None, scene_recognizer=None, config=None):
        """初始化参数优化器
        
        Args:
            templates_dir: 场景参数模板目录
            scene_recognizer: 场景识别器实例，如果为None则创建新实例
            config: 配置字典，如果为None则使用默认配置
        """
        global logger
        
        # 初始化配置为字典，确保即使传入None也不会出错
        self.config = config if config is not None else {}
        
        # 如果配置为空，尝试使用全局配置或初始化全局配置
        if not self.config:
            if service_config:
                self.config = service_config
            else:
                try:
                    init_global_config()
                    self.config = service_config if service_config else {}
                except Exception as e:
                    logger.warning(f"初始化全局配置失败: {e}")
        
        # 根据配置更新日志设置
        if self.config:
            try:
                # 确定日志目录
                # 优先从config的log_file中提取目录，如果不存在则使用默认值
                if "log_file" in self.config:
                    log_dir = os.path.dirname(self.config["log_file"])
                else:
                    log_dir = self.config.get("log_dir", "/var/log/industrial-scene-optimizer")
                
                # 确保日志目录存在
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except Exception as e:
                    logger.warning(f"创建日志目录失败: {e}")
                    # 回退到当前目录
                    log_dir = os.path.dirname(os.path.abspath(__file__))
                
                # 确定日志文件路径
                log_file = os.path.join(log_dir, "param_optimizer.log")
                
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
            except Exception as e:
                logger.warning(f"更新日志配置时出错: {e}")
        
        # 确定模板目录
        if templates_dir is None:
            # 优先从配置文件中获取
            templates_dir = self.config.get("param_templates_dir", None)
            
            # 如果配置文件中没有，则使用脚本所在目录
            if templates_dir is None:
                try:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    templates_dir = os.path.join(script_dir, "templates")
                except Exception as e:
                    logger.warning(f"获取模板目录时出错: {e}")
                    templates_dir = "templates"  # 使用相对路径作为最后的回退
        
        # 确定数据目录用于存储参数应用历史和备份文件
        try:
            if self.config and "data_dir" in self.config:
                self.data_dir = self.config["data_dir"]
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                self.data_dir = os.path.join(script_dir, "data")
                
            # 确保数据目录存在
            try:
                os.makedirs(self.data_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"创建数据目录失败: {e}")
                # 回退到当前目录
                self.data_dir = os.path.dirname(os.path.abspath(__file__))
        except Exception as e:
            logger.warning(f"确定数据目录时出错: {e}")
            # 使用当前目录作为最后的回退
            self.data_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 设置原始参数备份文件路径（YAML格式）
        self.original_params_file = os.path.join(self.data_dir, "original_system_params.yaml")
        
        # 检测操作系统类型
        self.os_type = self._detect_os_type()
        
        # 如果未提供scene_recognizer，则在需要时创建
        self.scene_recognizer = scene_recognizer
        
        # 确保模板目录存在
        if not os.path.exists(templates_dir):
            logger.warning(f"模板目录不存在: {templates_dir}")
        
        self.templates_dir = templates_dir
        logger.info(f"参数优化器已初始化，模板目录: {templates_dir}")
        
        # 缓存加载的模板配置
        self.template_cache = {}
        
        # 参数应用历史记录
        self.applied_params_history = []
        
        # 当前应用的场景
        self.current_scene = None

    def _detect_os_type(self):
        """检测操作系统类型
        
        Returns:
            操作系统类型（'windows' 或 'linux'）
        """
        import platform
        system = platform.system().lower()
        if 'windows' in system:
            return 'windows'
        elif 'linux' in system:
            return 'linux'
        else:
            return 'unknown'
    
    def load_scene_parameters(self, scene_name):
        """加载对应场景的参数文件
        
        Args:
            scene_name: 场景名称
            
        Returns:
            参数字典或None（如果加载失败）
        """
        # 构建参数文件路径
        template_path = os.path.join(self.templates_dir, f"{scene_name}.yaml")
        
        # 检查参数文件是否存在
        if not os.path.exists(template_path):
            logger.warning(f"场景参数文件不存在: {template_path}")
            
            # 尝试创建默认参数文件
            self._create_default_scene_params(scene_name)
            
            # 再次检查文件是否存在
            if not os.path.exists(template_path):
                return None
        
        try:
            with open(template_path, 'r') as f:
                params = yaml.safe_load(f)
            
            logger.info(f"成功加载{scene_name}场景的参数文件")
            return params
        except Exception as e:
            logger.error(f"加载场景参数文件失败: {e}")
            return None
    
    def _create_default_scene_params(self, scene_name):
        """为指定场景创建默认参数文件
        
        Args:
            scene_name: 场景名称
        """
        # 定义默认参数模板
        default_params = {
            'sysctl': {
                'vm.swappiness': 10,
                'net.core.somaxconn': 1024,
                'net.ipv4.tcp_tw_reuse': 1,
                'kernel.sched_min_granularity_ns': 10000000,
                'kernel.sched_wakeup_granularity_ns': 15000000
            },
            'io_scheduler': 'deadline',
            'thread_limit': 1024,
            'description': f'默认的{scene_name}场景参数配置'
        }
        
        # 根据场景类型调整默认参数
        if scene_name == 'compute_intensive':
            # 计算密集型场景的优化参数
            default_params['sysctl'].update({
                'vm.swappiness': 5,
                'kernel.sched_min_granularity_ns': 10000000,
                'kernel.sched_wakeup_granularity_ns': 15000000,
                'kernel.sched_latency_ns': 60000000
            })
        elif scene_name == 'data_intensive':
            # 数据密集型场景的优化参数
            default_params['sysctl'].update({
                'vm.swappiness': 1,
                'vm.dirty_ratio': 40,
                'vm.dirty_background_ratio': 10,
                'kernel.sched_min_granularity_ns': 30000000,
                'kernel.sched_wakeup_granularity_ns': 60000000
            })
            default_params['io_scheduler'] = 'cfq'
        elif scene_name == 'hybrid_load':
            # 混合负载场景的优化参数
            default_params['sysctl'].update({
                'vm.swappiness': 10,
                'kernel.sched_min_granularity_ns': 20000000,
                'kernel.sched_wakeup_granularity_ns': 30000000
            })
        elif scene_name == 'light_load':
            # 轻量负载场景的优化参数
            default_params['sysctl'].update({
                'vm.swappiness': 60,
                'kernel.sched_min_granularity_ns': 10000000,
                'kernel.sched_wakeup_granularity_ns': 15000000
            })
        
        # 保存默认参数文件
        template_path = os.path.join(self.templates_dir, f"{scene_name}.yaml")
        try:
            with open(template_path, 'w') as f:
                yaml.dump(default_params, f, default_flow_style=False)
            
            logger.info(f"已创建默认的{scene_name}场景参数文件: {template_path}")
        except Exception as e:
            logger.error(f"创建默认场景参数文件失败: {e}")
    
    def _backup_original_parameters(self, params_to_backup=None):
        """备份系统的原始参数
        
        Args:
            params_to_backup: 需要备份的参数列表，如果为None则备份所有可能的参数
        
        Returns:
            备份的参数字典
        """
        logger.info("开始备份系统的原始参数")
        
        original_params = {
            'timestamp': time.time(),
            'sysctl': {},
            'io_scheduler': {},
            'thread_limit': None,
            'cpu_governor': {}
        }
        
        # 定义一些可能在不同系统上不可用的非关键参数列表
        non_critical_params = [
            'kernel.sched_migration_cost_ns', 
            'kernel.sched_latency_ns'
        ]
        
        try:
            if self.os_type == 'linux':
                # 备份sysctl参数
                if 'sysctl' in params_to_backup:
                    for key in params_to_backup['sysctl']:
                        try:
                            result = subprocess.run(['sysctl', key], check=True, capture_output=True, text=True)
                            value = result.stdout.split(' = ')[1].strip()
                            original_params['sysctl'][key] = value
                            logger.info(f"备份sysctl参数: {key}={value}")
                        except Exception as e:
                            # 对于非关键参数，记录警告但继续处理其他参数
                            if key in non_critical_params:
                                logger.warning(f"备份非关键sysctl参数失败 {key}: {e}，继续处理其他参数")
                            else:
                                logger.warning(f"备份sysctl参数失败 {key}: {e}")
                
                # 备份I/O调度器
                if 'io_scheduler' in params_to_backup:
                    devices = ['sda', 'sdb']  # 示例设备列表
                    for device in devices:
                        try:
                            device_path = f'/sys/block/{device}/queue/scheduler'
                            if os.path.exists(device_path):
                                with open(device_path, 'r') as f:
                                    scheduler = f.read().strip()
                                    # 提取当前活动的调度器（通常用[ ]包围）
                                    if '[' in scheduler:
                                        active_scheduler = scheduler[scheduler.find('[')+1:scheduler.find(']')]
                                        original_params['io_scheduler'][device] = active_scheduler
                                        logger.info(f"备份设备 {device} 的I/O调度器: {active_scheduler}")
                        except Exception as e:
                            logger.warning(f"备份设备 {device} 的I/O调度器失败: {e}")
                
                # 备份线程数限制
                if 'thread_limit' in params_to_backup:
                    try:
                        result = subprocess.run(['ulimit', '-u'], check=True, capture_output=True, text=True)
                        thread_limit = result.stdout.strip()
                        original_params['thread_limit'] = thread_limit
                        logger.info(f"备份线程数限制: {thread_limit}")
                    except Exception as e:
                        logger.warning(f"备份线程数限制失败: {e}")
                
                # 备份CPU调节器
                if 'cpu_governor' in params_to_backup:
                    for cpu_dir in os.listdir('/sys/devices/system/cpu/'):
                        if cpu_dir.startswith('cpu') and cpu_dir != 'cpu':
                            try:
                                governor_path = os.path.join('/sys/devices/system/cpu/', cpu_dir, 'cpufreq', 'scaling_governor')
                                if os.path.exists(governor_path):
                                    with open(governor_path, 'r') as f:
                                        governor = f.read().strip()
                                        original_params['cpu_governor'][cpu_dir] = governor
                                        logger.info(f"备份CPU {cpu_dir} 的调节器: {governor}")
                            except Exception as e:
                                logger.warning(f"备份CPU {cpu_dir} 的调节器失败: {e}")
            else:
                # 在非Linux系统上模拟备份
                logger.info(f"[模拟] 备份系统参数 (当前系统: {self.os_type})")
        except Exception as e:
            logger.error(f"备份系统参数过程中发生错误: {e}")
        
        # 保存备份到YAML文件
        try:
            import yaml
            with open(self.original_params_file, 'w') as f:
                yaml.dump(original_params, f, default_flow_style=False, sort_keys=False)
            logger.info(f"系统原始参数已备份到YAML文件: {self.original_params_file}")
        except ImportError:
            logger.error("未找到yaml模块，请安装pyyaml包")
        except Exception as e:
            logger.error(f"保存系统原始参数备份失败: {e}")
        
        return original_params
    
    def _restore_original_parameters(self):
        """恢复系统的原始参数
        
        Returns:
            是否成功恢复
        """
        logger.info("开始恢复系统的原始参数")
        
        # 检查备份文件是否存在
        if not os.path.exists(self.original_params_file):
            logger.warning(f"原始参数备份文件不存在: {self.original_params_file}")
            return False
        
        try:
            # 加载YAML格式的备份文件
            try:
                import yaml
                with open(self.original_params_file, 'r') as f:
                    original_params = yaml.load(f, Loader=yaml.FullLoader)
                
                logger.info(f"成功加载YAML格式原始参数备份: {self.original_params_file}")
            except ImportError:
                logger.error("未找到yaml模块，请安装pyyaml包")
                return False
            except yaml.YAMLError as e:
                logger.error(f"解析YAML备份文件失败: {e}")
                return False
            
            # 记录恢复的参数
            restored_params = {
                'timestamp': time.time(),
                'restored': {},
                'failed': {}
            }
            
            if self.os_type == 'linux':
                # 恢复sysctl参数
                if 'sysctl' in original_params and original_params['sysctl']:
                    for key, value in original_params['sysctl'].items():
                        if self._apply_sysctl_parameter(key, value):
                            restored_params['restored'][f'sysctl.{key}'] = value
                        else:
                            restored_params['failed'][f'sysctl.{key}'] = value
                
                # 恢复I/O调度器
                if 'io_scheduler' in original_params and original_params['io_scheduler']:
                    for device, scheduler in original_params['io_scheduler'].items():
                        try:
                            device_path = f'/sys/block/{device}/queue/scheduler'
                            if os.path.exists(device_path):
                                with open(device_path, 'w') as f:
                                    f.write(scheduler)
                                restored_params['restored'][f'io_scheduler.{device}'] = scheduler
                                logger.info(f"恢复设备 {device} 的I/O调度器为: {scheduler}")
                        except Exception as e:
                            logger.warning(f"恢复设备 {device} 的I/O调度器失败: {e}")
                            restored_params['failed'][f'io_scheduler.{device}'] = scheduler
                
                # 恢复线程数限制
                if 'thread_limit' in original_params and original_params['thread_limit'] is not None:
                    if self._apply_thread_limit(original_params['thread_limit']):
                        restored_params['restored']['thread_limit'] = original_params['thread_limit']
                    else:
                        restored_params['failed']['thread_limit'] = original_params['thread_limit']
                
                # 恢复CPU调节器
                if 'cpu_governor' in original_params and original_params['cpu_governor']:
                    for cpu_dir, governor in original_params['cpu_governor'].items():
                        try:
                            governor_path = os.path.join('/sys/devices/system/cpu/', cpu_dir, 'cpufreq', 'scaling_governor')
                            if os.path.exists(governor_path):
                                with open(governor_path, 'w') as f:
                                    f.write(governor)
                                restored_params['restored'][f'cpu_governor.{cpu_dir}'] = governor
                                logger.info(f"恢复CPU {cpu_dir} 的调节器为: {governor}")
                        except Exception as e:
                            logger.warning(f"恢复CPU {cpu_dir} 的调节器失败: {e}")
                            restored_params['failed'][f'cpu_governor.{cpu_dir}'] = governor
            else:
                # 在非Linux系统上模拟恢复
                logger.info(f"[模拟] 恢复系统参数 (当前系统: {self.os_type})")
                # 模拟成功恢复
                restored_params['restored']['simulated'] = 'success'
            
            # 记录恢复结果
            success_count = len(restored_params['restored'])
            failed_count = len(restored_params['failed'])
            total_count = success_count + failed_count
            
            logger.info(f"系统参数恢复完成: 成功{success_count}/{total_count}, 失败{failed_count}/{total_count}")
            
            # 如果有失败的参数，记录失败信息
            if failed_count > 0:
                logger.warning(f"以下参数恢复失败: {', '.join(restored_params['failed'].keys())}")
            
            # 清空当前场景
            self.current_scene = None
            
            # 将恢复结果添加到历史记录
            self.applied_params_history.append(restored_params)
            
            return success_count > 0  # 只要有一个参数成功恢复就返回True
        except Exception as e:
            logger.error(f"恢复系统参数过程中发生错误: {e}")
            return False
    
    def apply_scene_parameters(self, scene_name):
        """应用对应场景的优化参数
        
        Args:
            scene_name: 场景名称
            
        Returns:
            是否成功应用参数
        """
        # 加载参数
        params = self.load_scene_parameters(scene_name)
        if params is None:
            return False
        
        logger.info(f"正在准备应用{scene_name}场景的优化参数")
        
        # 1. 检查是否有备份的原始参数文件，如果有则先恢复原始参数
        if os.path.exists(self.original_params_file):
            logger.info("发现存在原始参数备份文件，先恢复原始参数")
            if not self._restore_original_parameters():
                logger.warning("恢复原始参数失败，但仍尝试继续应用新参数")
        
        # 2. 备份当前的原始参数（此时系统已经恢复到最初状态或保持当前状态）
        self._backup_original_parameters(params)
        
        # 3. 应用新的场景参数
        logger.info(f"开始应用{scene_name}场景的优化参数")
        
        # 记录应用的参数
        applied_params = {
            'timestamp': time.time(),
            'scene': scene_name,
            'applied': {},
            'failed': {}
        }
        
        try:
            # 1. 配置sysctl参数
            if 'sysctl' in params:
                for key, value in params['sysctl'].items():
                    if self._apply_sysctl_parameter(key, value):
                        applied_params['applied'][f'sysctl.{key}'] = value
                    else:
                        applied_params['failed'][f'sysctl.{key}'] = value
            
            # 2. 配置I/O调度器
            if 'io_scheduler' in params:
                if self._apply_io_scheduler(params['io_scheduler']):
                    applied_params['applied']['io_scheduler'] = params['io_scheduler']
                else:
                    applied_params['failed']['io_scheduler'] = params['io_scheduler']
            
            # 3. 配置线程数限制
            if 'thread_limit' in params:
                if self._apply_thread_limit(params['thread_limit']):
                    applied_params['applied']['thread_limit'] = params['thread_limit']
                else:
                    applied_params['failed']['thread_limit'] = params['thread_limit']
            
            # 4. 配置CPU频率（如果支持）
            if 'cpu_governor' in params:
                if self._apply_cpu_governor(params['cpu_governor']):
                    applied_params['applied']['cpu_governor'] = params['cpu_governor']
                else:
                    applied_params['failed']['cpu_governor'] = params['cpu_governor']
            
            # 5. 应用自定义脚本（如果有）
            if 'custom_scripts' in params:
                for script_name, script_path in params['custom_scripts'].items():
                    if self._execute_custom_script(script_path):
                        applied_params['applied'][f'script.{script_name}'] = script_path
                    else:
                        applied_params['failed'][f'script.{script_name}'] = script_path
            
            # 6. 配置预读缓存大小
            if 'read_ahead_kb' in params:
                if self._apply_read_ahead_kb(params['read_ahead_kb']):
                    applied_params['applied']['read_ahead_kb'] = params['read_ahead_kb']
                else:
                    applied_params['failed']['read_ahead_kb'] = params['read_ahead_kb']
            
            # 7. 配置CPU空闲状态
            if 'cpu_idle' in params and 'max_cstate' in params['cpu_idle']:
                if self._apply_cpu_idle_max_cstate(params['cpu_idle']['max_cstate']):
                    applied_params['applied']['cpu_idle.max_cstate'] = params['cpu_idle']['max_cstate']
                else:
                    applied_params['failed']['cpu_idle.max_cstate'] = params['cpu_idle']['max_cstate']
            
            # 8. 配置网络参数
            if 'net' in params:
                # 处理网络核心参数
                if 'core' in params['net']:
                    for key, value in params['net']['core'].items():
                        net_key = f'net.core.{key}'
                        if self._apply_sysctl_parameter(net_key, value):
                            applied_params['applied'][f'net.core.{key}'] = value
                        else:
                            applied_params['failed'][f'net.core.{key}'] = value
                # 处理IPv4参数
                if 'ipv4' in params['net']:
                    for key, value in params['net']['ipv4'].items():
                        net_key = f'net.ipv4.{key}'
                        if self._apply_sysctl_parameter(net_key, value):
                            applied_params['applied'][f'net.ipv4.{key}'] = value
                        else:
                            applied_params['failed'][f'net.ipv4.{key}'] = value
            
            # 9. 配置禁用的服务
            if 'services' in params and 'disabled' in params['services']:
                for service_name in params['services']['disabled']:
                    if self._disable_service(service_name):
                        applied_params['applied'][f'service.disabled.{service_name}'] = True
                    else:
                        applied_params['failed'][f'service.disabled.{service_name}'] = True
            
            # 记录参数应用结果
            self.applied_params_history.append(applied_params)
            
            # 记录当前场景
            self.current_scene = scene_name
            
            # 记录应用结果
            success_count = len(applied_params['applied'])
            failed_count = len(applied_params['failed'])
            total_count = success_count + failed_count
            
            logger.info(f"场景参数应用完成: 成功{success_count}/{total_count}, 失败{failed_count}/{total_count}")
            
            # 如果有失败的参数，记录失败信息
            if failed_count > 0:
                logger.warning(f"以下参数应用失败: {', '.join(applied_params['failed'].keys())}")
            
            return success_count > 0  # 只要有一个参数成功应用就返回True
        except Exception as e:
            logger.error(f"应用场景参数过程中发生错误: {e}")
            return False
    
    def _apply_sysctl_parameter(self, key, value):
        """应用sysctl参数
        
        Args:
            key: 参数名称
            value: 参数值
            
        Returns:
            是否成功应用
        """
        # 定义一些可能在不同系统上不可用的非关键参数列表
        non_critical_params = [
            'kernel.sched_migration_cost_ns', 
            'kernel.sched_latency_ns',
            'kernel.sched_min_granularity_ns', 
            'kernel.sched_wakeup_granularity_ns'
        ]
        
        try:
            if self.os_type == 'linux':
                # 在Linux系统上实际应用参数
                result = subprocess.run(['sysctl', '-w', f'{key}={value}'], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"设置sysctl参数: {key}={value}")
                    return True
                else:
                    # 记录失败原因
                    error_msg = result.stderr.strip() if result.stderr else str(result.returncode)
                    logger.warning(f"设置sysctl参数失败 {key}={value}: {error_msg}")
                    
                    # 对于非关键参数，我们可以更宽容地处理失败
                    # 因为这些参数可能在不同的Linux版本中不可用或需要特殊权限
                    if key in non_critical_params:
                        logger.info(f"跳过非关键参数 {key} 的设置，继续应用其他参数")
                        # 注意：这里返回True是为了不让这些非关键参数的失败影响整体统计
                        # 但在日志中我们已经记录了实际的失败情况
                        return True
                    return False
            else:
                # 在非Linux系统上模拟应用参数
                logger.info(f"[模拟] 设置sysctl参数: {key}={value} (当前系统: {self.os_type})")
                return True
        except Exception as e:
            logger.warning(f"设置sysctl参数时发生异常 {key}={value}: {e}")
            # 对于非关键参数，我们仍然可以返回True来继续其他操作
            if key in non_critical_params:
                return True
            return False
    
    def _get_block_devices(self):
        """获取Linux系统中的块设备列表，确保能够适配不同Linux系统的设备命名"""
        devices = []
        
        # 方法1：使用lsblk命令 - 推荐的现代方法
        try:
            result = subprocess.run(
                "lsblk -d -o NAME,TYPE | grep disk | awk '{print $1}'", 
                shell=True, 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0 and result.stdout:
                devices = [line.strip() for line in result.stdout.strip().split('\n')]
                devices.sort()
        except Exception as e:
            logger.warning(f"使用lsblk命令获取块设备失败: {str(e)}")
        
        # 方法2：检查/proc/partitions文件 - 更底层的方法
        if not devices and os.path.exists('/proc/partitions'):
            try:
                with open('/proc/partitions', 'r') as f:
                    lines = f.readlines()[2:]  # 跳过标题行
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            dev_name = parts[3]
                            # 排除分区，只保留主设备
                            if not re.match(r'^[a-zA-Z]+\\d+$', dev_name):
                                devices.append(dev_name)
                devices.sort()
            except Exception as e:
                logger.warning(f"读取/proc/partitions文件失败: {str(e)}")
        
        # 方法3：检查/dev目录中的常见块设备模式
        if not devices and os.path.exists('/dev'):
            try:
                # 搜索常见的块设备命名模式
                common_patterns = ['sd[a-z]+', 'hd[a-z]+', 'vd[a-z]+', 'nvme\\d+n\\d+', 'mmcblk\\d+']
                for pattern in common_patterns:
                    try:
                        result = subprocess.run(
                            f"ls -1 /dev | grep -E '^{pattern}$' | head -5",
                            shell=True, 
                            capture_output=True, 
                            text=True
                        )
                        if result.returncode == 0 and result.stdout:
                            pattern_devices = [line.strip() for line in result.stdout.strip().split('\n')]
                            for dev in pattern_devices:
                                if dev not in devices:
                                    devices.append(dev)
                    except Exception as e:
                        logger.warning(f"搜索块设备模式{pattern}失败: {str(e)}")
                devices.sort()
            except Exception as e:
                logger.warning(f"搜索块设备失败: {str(e)}")
        
        # 如果没有找到设备，使用默认设备列表
        if not devices:
            devices = ['sda', 'sdb']  # 默认设备列表
            logger.warning(f"未检测到块设备，使用默认设备列表: {devices}")
        
        return devices

    def _apply_io_scheduler(self, scheduler):
        """应用I/O调度器
        
        Args:
            scheduler: I/O调度器名称
            
        Returns:
            是否成功应用
        """
        try:
            if self.os_type == 'linux':
                # 在Linux系统上动态检测块设备
                devices = self._get_block_devices()
                success = False
                
                if not devices:
                    logger.warning(f"没有找到有效的块设备，无法设置I/O调度器: {scheduler}")
                    return False
                
                logger.info(f"尝试在以下设备上设置I/O调度器: {devices}")
                
                for device in devices:
                    try:
                        device_path = f'/sys/block/{device}/queue/scheduler'
                        if os.path.exists(device_path):
                            # 检查文件是否可写
                            if os.access(device_path, os.W_OK):
                                with open(device_path, 'w') as f:
                                    f.write(scheduler)
                                success = True
                                logger.info(f"设置设备 {device} 的I/O调度器为: {scheduler}")
                            else:
                                logger.warning(f"设备 {device} 的I/O调度器文件不可写: {device_path}")
                        else:
                            logger.warning(f"设备 {device} 的I/O调度器文件不存在: {device_path}")
                    except PermissionError:
                        logger.warning(f"设置设备 {device} 的I/O调度器权限不足，请以root用户运行")
                    except Exception as e:
                        logger.warning(f"设置设备 {device} 的I/O调度器失败: {e}")
                        continue
                
                return success
            else:
                # 在非Linux系统上模拟应用I/O调度器
                logger.info(f"[模拟] 设置I/O调度器: {scheduler} (当前系统: {self.os_type})")
                return True
        except Exception as e:
            logger.error(f"设置I/O调度器过程中发生错误: {e}")
            return False
    
    def _apply_thread_limit(self, limit):
        """应用线程数限制
        
        Args:
            limit: 线程数限制
            
        Returns:
            是否成功应用
        """
        try:
            if self.os_type == 'linux':
                # 在Linux系统上实际应用线程数限制
                subprocess.run(['ulimit', '-u', str(limit)], check=True, capture_output=True, text=True)
                logger.info(f"设置线程数限制: {limit}")
                return True
            elif self.os_type == 'windows':
                # 在Windows系统上模拟应用线程数限制
                logger.info(f"[模拟] 设置线程数限制: {limit} (Windows系统上无法直接通过命令设置)")
                return True
            else:
                # 在其他系统上模拟应用线程数限制
                logger.info(f"[模拟] 设置线程数限制: {limit} (当前系统: {self.os_type})")
                return True
        except Exception as e:
            logger.error(f"设置线程数限制失败: {e}")
            return False
    
    def _apply_cpu_governor(self, governor):
        """应用CPU调节器
        
        Args:
            governor: CPU调节器名称
            
        Returns:
            是否成功应用
        """
        try:
            if self.os_type == 'linux':
                # 在Linux系统上实际应用CPU调节器
                # 这里简化处理，实际应用中可能需要遍历所有CPU核心
                for cpu_dir in os.listdir('/sys/devices/system/cpu/'):
                    if cpu_dir.startswith('cpu') and cpu_dir != 'cpu':
                        try:
                            governor_path = os.path.join('/sys/devices/system/cpu/', cpu_dir, 'cpufreq', 'scaling_governor')
                            if os.path.exists(governor_path):
                                with open(governor_path, 'w') as f:
                                    f.write(governor)
                                logger.info(f"设置CPU {cpu_dir} 的调节器为: {governor}")
                        except Exception as e:
                            logger.warning(f"设置CPU {cpu_dir} 的调节器失败: {e}")
                            continue
                
                return True
            else:
                # 在非Linux系统上模拟应用CPU调节器
                logger.info(f"[模拟] 设置CPU调节器: {governor} (当前系统: {self.os_type})")
                return True
        except Exception as e:
            logger.error(f"设置CPU调节器失败: {e}")
            return False
            
    def _apply_read_ahead_kb(self, read_ahead_kb):
        """应用预读缓存大小设置
        
        Args:
            read_ahead_kb: 预读缓存大小（KB）
            
        Returns:
            是否成功应用
        """
        try:
            if self.os_type == 'linux':
                # 在Linux系统上实际应用预读缓存大小
                devices = self._get_block_devices()
                success = False
                
                if not devices:
                    logger.warning(f"没有找到有效的块设备，无法设置预读缓存大小: {read_ahead_kb}")
                    return False
                
                logger.info(f"尝试在以下设备上设置预读缓存大小: {devices}")
                
                for device in devices:
                    try:
                        device_path = f'/sys/block/{device}/queue/read_ahead_kb'
                        if os.path.exists(device_path):
                            # 检查文件是否可写
                            if os.access(device_path, os.W_OK):
                                with open(device_path, 'w') as f:
                                    f.write(str(read_ahead_kb))
                                success = True
                                logger.info(f"设置设备 {device} 的预读缓存大小为: {read_ahead_kb} KB")
                            else:
                                logger.warning(f"设备 {device} 的预读缓存大小文件不可写: {device_path}")
                        else:
                            logger.warning(f"设备 {device} 的预读缓存大小文件不存在: {device_path}")
                    except PermissionError:
                        logger.warning(f"设置设备 {device} 的预读缓存大小权限不足，请以root用户运行")
                    except Exception as e:
                        logger.warning(f"设置设备 {device} 的预读缓存大小失败: {e}")
                        continue
                
                return success
            else:
                # 在非Linux系统上模拟应用预读缓存大小
                logger.info(f"[模拟] 设置预读缓存大小: {read_ahead_kb} KB (当前系统: {self.os_type})")
                return True
        except Exception as e:
            logger.error(f"设置预读缓存大小过程中发生错误: {e}")
            return False
            
    def _apply_cpu_idle_max_cstate(self, max_cstate):
        """应用CPU空闲状态最大C-state设置
        
        Args:
            max_cstate: 最大C-state值
            
        Returns:
            是否成功应用
        """
        try:
            if self.os_type == 'linux':
                # 在Linux系统上实际应用CPU空闲状态设置
                # 尝试不同的路径
                cstate_paths = [
                    '/sys/devices/system/cpu/cpuidle/current_driver/max_cstate',
                    '/sys/module/intel_idle/parameters/max_cstate',
                    '/sys/module/processor/parameters/max_cstate'
                ]
                
                success = False
                
                for path in cstate_paths:
                    if os.path.exists(path):
                        try:
                            # 检查文件是否可写
                            if os.access(path, os.W_OK):
                                with open(path, 'w') as f:
                                    f.write(str(max_cstate))
                                success = True
                                logger.info(f"设置CPU最大C-state为: {max_cstate} (文件路径: {path})")
                                break
                            else:
                                logger.warning(f"CPU最大C-state文件不可写: {path}")
                        except PermissionError:
                            logger.warning(f"设置CPU最大C-state权限不足，请以root用户运行")
                        except Exception as e:
                            logger.warning(f"设置CPU最大C-state失败: {e}")
                            continue
                
                if not success:
                    logger.warning(f"未找到可写的CPU最大C-state文件路径")
                
                return success
            else:
                # 在非Linux系统上模拟应用CPU空闲状态设置
                logger.info(f"[模拟] 设置CPU最大C-state为: {max_cstate} (当前系统: {self.os_type})")
                return True
        except Exception as e:
            logger.error(f"设置CPU最大C-state过程中发生错误: {e}")
            return False
            
    def _disable_service(self, service_name):
        """禁用指定的系统服务
        
        Args:
            service_name: 服务名称
            
        Returns:
            是否成功禁用
        """
        try:
            if self.os_type == 'linux':
                # 在Linux系统上实际禁用服务
                logger.info(f"尝试禁用服务: {service_name}")
                
                # 首先检查服务是否存在
                check_cmd = ['systemctl', 'list-unit-files', service_name]
                check_result = subprocess.run(check_cmd, capture_output=True, text=True)
                
                if check_result.returncode == 0 and service_name in check_result.stdout:
                    # 服务存在，尝试禁用
                    try:
                        disable_cmd = ['systemctl', 'disable', service_name]
                        disable_result = subprocess.run(disable_cmd, check=True, capture_output=True, text=True)
                        logger.info(f"成功禁用服务: {service_name}")
                        
                        # 尝试停止服务（如果正在运行）
                        try:
                            stop_cmd = ['systemctl', 'stop', service_name]
                            subprocess.run(stop_cmd, capture_output=True, text=True)
                            logger.info(f"尝试停止服务: {service_name}")
                        except Exception as e:
                            logger.warning(f"停止服务{service_name}时发生错误: {e}")
                        
                        return True
                    except Exception as e:
                        logger.error(f"禁用服务{service_name}失败: {e}")
                        return False
                else:
                    logger.warning(f"服务{service_name}不存在或无法获取信息")
                    return True  # 如果服务不存在，也视为成功（因为不需要禁用）
            else:
                # 在非Linux系统上模拟禁用服务
                logger.info(f"[模拟] 禁用服务: {service_name} (当前系统: {self.os_type})")
                return True
        except Exception as e:
            logger.error(f"禁用服务{service_name}过程中发生错误: {e}")
            return False
    
    def _execute_custom_script(self, script_path):
        """执行自定义脚本
        
        Args:
            script_path: 脚本路径
            
        Returns:
            是否成功执行
        """
        try:
            # 检查脚本是否存在且可执行
            if os.path.exists(script_path):
                # 确保脚本有执行权限
                os.chmod(script_path, 0o755)
                
                # 执行脚本
                subprocess.run([script_path], check=True, capture_output=True, text=True)
                logger.info(f"执行自定义脚本成功: {script_path}")
                return True
            else:
                logger.error(f"自定义脚本不存在: {script_path}")
                return False
        except Exception as e:
            logger.error(f"执行自定义脚本失败: {e}")
            return False
    
    def get_applied_params_history(self):
        """获取参数应用历史
        
        Returns:
            参数应用历史列表
        """
        return self.applied_params_history
    
    def save_history_to_file(self, file_path=None):
        """将参数应用历史保存到文件
        
        Args:
            file_path: 保存文件路径，如果为None则使用配置中的data_dir
            
        Returns:
            是否成功保存
        """
        try:
            # 如果未提供文件路径，则使用配置中的data_dir
            if file_path is None:
                file_path = os.path.join(self.data_dir, "params_history.json")
            
            import json
            with open(file_path, 'w') as f:
                json.dump(self.applied_params_history, f, indent=4)
            
            logger.info(f"参数应用历史已保存到: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存参数应用历史失败: {e}")
            return False

if __name__ == "__main__":
    # 创建参数优化器实例
    optimizer = ParamOptimizer()
    
    # 注意：由于HPC优化代码已删除，实时调优功能不再可用
    logger.info("HPC优化代码已删除，ParamOptimizer现在提供参数应用和恢复功能")