#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工业场景优化器主服务脚本
功能：整体调用数据收集、转换、场景识别和参数优化模块
支持配置文件设置监控模式（0为不间断监控，1为一次监控）
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import signal
from datetime import datetime

# 导入matplotlib并配置字体设置
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 禁用matplotlib的字体查找警告
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# 设置基础字体配置，使用更通用的设置以适应不同环境
plt.rcParams['font.family'] = ['sans-serif']
# 使用安全的系统字体，避免依赖特定中文字体
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 导入各个功能模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_atune_config import AtuneConfigGenerator
# 使用原始数据转换器
from data_transformer import DataTransformer
# 导入日志工具
from logger_utils import init_main_logger
# SceneRecognizer和ParamOptimizer将在需要时延迟导入

class ServiceMain:
    def __init__(self, config_path=None):
        """初始化服务主类
        
        Args:
            config_path: 服务配置文件路径
        """
        # 优先从环境变量获取配置文件路径
        if config_path is None:
            # 首先检查系统配置目录
            system_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "service_config.conf")
            if os.path.exists(system_config_path):
                config_path = system_config_path
            else:
                # 回退到环境变量或当前目录
                config_path = os.environ.get('SERVICE_CONFIG_PATH', 'service_config.conf')
        
        # 加载配置文件
        self.config = self._load_config(config_path)
        
        # 初始化日志
        self._init_logger()
        
        # 创建数据目录
        os.makedirs(self.config['data_dir'], exist_ok=True)
        
        # 初始化采集器
        self.collector = None
        
        # 初始化转换器、识别器和优化器
        self.transformer = DataTransformer(self.config)
        self.recognizer = None
        self.optimizer = None
        
        # 初始化模块
        self._init_modules()
        
        # 运行状态
        self.running = False
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        logger.info("工业场景优化器服务已初始化完成")
        
    def _load_config(self, config_path):
        """加载配置文件
        
        Returns:
            配置字典
        """
        try:
            config = {}
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
            
            # 确保必要的配置项存在
            required_configs = [
                'monitor_mode', 'data_dir', 'log_file', 'log_level'
            ]
            
            # 对于monitor_mode=0（不间断监控模式），需要collect_interval参数
            if config.get('monitor_mode') == 0:
                if 'collect_interval' not in config:
                    raise ValueError(f"配置文件中缺少必要项: collect_interval（在不间断监控模式下）")
            
            # 验证其他必要配置项
            for key in required_configs:
                if key not in config:
                    raise ValueError(f"配置文件中缺少必要项: {key}")
            
            # 注意：interval和sample_num参数用于generate_atune_config.py，不需要在此处验证
            # 这些参数会在AtuneConfigGenerator类中被处理和使用
            
            # 确保数据目录存在
            if not os.path.exists(config['data_dir']):
                os.makedirs(config['data_dir'])
            
            return config
        except Exception as e:
            print(f"加载配置文件失败: {e}", file=sys.stderr)
            sys.exit(1)
        
    def _init_logger(self):
        """初始化日志系统"""
        global logger
        # 使用日志工具初始化主日志记录器
        logger = init_main_logger(self.config)
        
        # 获取根日志记录器并关闭其传播功能，防止日志重复
        root_logger = logging.getLogger()
        root_logger.propagate = False
        
    def _init_modules(self):
        """初始化各个功能模块"""
        try:
            logger.info("基础模块初始化成功")
        except Exception as e:
            logger.error(f"初始化模块失败: {e}")
            sys.exit(1)
            
    def _handle_signal(self, signum, frame):
        """处理终止信号"""
        logger.info(f"接收到信号 {signum}，准备停止服务...")
        self.running = False

    def _collect_data(self):
        """采集数据
        
        Returns:
            采集的数据文件路径列表
        """
        try:
            # 生成配置文件
            atune_config_path = self._generate_atune_config()
            
            data_files = []
            
            # 使用atune-collector采集数据
            atune_data_file = self._run_atune_collector(atune_config_path)
            if atune_data_file:
                data_files.append(atune_data_file)
            
            return data_files
        except Exception as e:
            logger.error(f"数据采集失败: {e}")
            return []
    
    def _generate_atune_config(self):
        """生成atune-collector配置文件
        
        Returns:
            配置文件路径或None
        """
        try:
            # 优先使用配置文件中指定的路径
            if 'atune_config' in self.config:
                atune_config_path = self.config['atune_config']
            else:
                # 使用配置目录或默认位置
                config_dir = self.config.get('config_dir', '/etc/industrial-scene-optimizer')
                atune_config_path = os.path.join(config_dir, 'collect_data.json')
            
            # 确保目录存在
            config_dir = os.path.dirname(atune_config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            # 初始化配置生成器并传入配置对象
            generator = AtuneConfigGenerator(self.config)
            
            # 首先检测系统信息
            logger.info("开始检测Linux系统信息...")
            generator.detect_system_info()
            logger.info("系统信息检测完成")
            
            # 然后生成配置
            logger.info("生成atune-collector配置文件...")
            config = generator.generate_config()
            
            # 更新配置中的输出目录
            config['output_dir'] = self.config['data_dir']
            
            # 不再设置采样数，使用默认行为
            # config['sample_num'] = self.config['max_samples']
            
            # 修复内存配置 - 确保'used'字段存在
            if 'meminfo' in config and isinstance(config['meminfo'], dict):
                if 'used' not in config['meminfo']:
                    # 计算used值（total - free - buffers - cached）
                    if 'total' in config['meminfo'] and 'free' in config['meminfo']:
                        used = config['meminfo']['total'] - config['meminfo']['free']
                        # 添加其他可能的字段
                        for field in ['buffers', 'cached', 'slab']:
                            if field in config['meminfo']:
                                used -= config['meminfo'][field]
                        config['meminfo']['used'] = max(0, used)  # 确保不小于0
            
            # 保存配置文件
            logger.info("验证生成的配置文件兼容性...")
            generator.save_config(config, atune_config_path)
            logger.info(f"自动生成atune-collector配置文件: {atune_config_path}")
            return atune_config_path
        except Exception as e:
            logger.error(f"生成atune-collector配置文件失败: {e}")
            return None
    
    def _run_atune_collector(self, config_path):
        """直接调用指定路径的collect_data.py采集数据
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            数据文件路径或None
        """
        try:
            # 直接使用用户指定的collect_data.py路径
            atune_collector_path = '/usr/lib/python3.11/site-packages/atune_collector/collect_data.py'
            
            # 检查脚本是否存在
            if not os.path.exists(atune_collector_path):
                # 如果Linux路径不存在，尝试在Windows环境下查找
                logger.warning(f"指定路径的collect_data.py不存在: {atune_collector_path}")
                atune_collector_path = self._find_atune_collector_script()
                if not atune_collector_path:
                    logger.error("无法找到collect_data.py脚本")
                    return None
            
            # 检查配置文件是否存在
            if not config_path or not os.path.exists(config_path):
                logger.error(f"配置文件不存在: {config_path}")
                return None
            
            # 构建命令参数 - 根据操作系统选择Python命令
            python_cmd = 'python' if sys.platform.startswith('win') else 'python3'
            cmd = [python_cmd, atune_collector_path, '-c', config_path]
            
            # 运行命令
            logger.info(f"开始运行atune-collector: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                logger.debug(f"atune-collector输出: {result.stdout}")
                
                # 查找生成的数据文件
                # 假设数据文件在output_dir中，文件名包含时间戳
                output_dir = self.config['data_dir']
                if os.path.exists(output_dir):
                    files = os.listdir(output_dir)
                    # 按修改时间排序，返回最新的文件
                    data_files = [f for f in files if f.endswith('.csv')]
                    if data_files:
                        data_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
                        return os.path.join(output_dir, data_files[0])
                
                return None
            except Exception as e:
                logger.error(f"运行atune-collector失败: {e}")
                logger.error(f"失败的完整命令: {' '.join(cmd)}")
                logger.error(f"错误输出: {getattr(e, 'stderr', '无详细错误信息')}")
                return None
        except Exception as e:
            logger.error(f"运行atune-collector过程中出错: {e}")
            return None

    def _find_atune_collector_script(self):
        """查找atune-collector脚本（用于回退机制）
        
        Returns:
            脚本路径或None
        """
        # 保留原有的查找逻辑作为回退机制
        atune_path = None
        
        # 1. 尝试从配置中指定的路径查找
        if 'atune_collector_path' in self.config:
            possible_path = self.config['atune_collector_path']
            if os.path.exists(possible_path):
                atune_path = possible_path
        
        # 2. 尝试标准安装路径
        if not atune_path:
            standard_paths = [
                '/usr/lib/python3.11/site-packages/atune_collector/collect_data.py',
                '/usr/lib/python3.10/site-packages/atune_collector/collect_data.py',
                '/usr/lib/python3.9/site-packages/atune_collector/collect_data.py',
                '/usr/lib/python3.8/site-packages/atune_collector/collect_data.py',
                '/usr/local/lib/python3.11/site-packages/atune_collector/collect_data.py',
            ]
            for path in standard_paths:
                if os.path.exists(path):
                    atune_path = path
                    break
        
        # 3. 尝试从Python库路径查找
        if not atune_path:
            for lib_path in sys.path:
                if 'site-packages' in lib_path:
                    possible_path = os.path.join(lib_path, 'atune_collector', 'collect_data.py')
                    if os.path.exists(possible_path):
                        atune_path = possible_path
                        break
        
        # 4. 尝试从当前目录相关路径查找
        if not atune_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join(current_dir, '..', 'atune_collector', 'collect_data.py'),
                os.path.join(current_dir, 'atune-collector', 'atune_collector', 'collect_data.py')
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    atune_path = path
                    break
        
        return atune_path

    def _transform_data(self, data_files):
        """转换数据格式
        
        Args:
            data_files: 原始数据文件路径列表
        
        Returns:
            转换后的数据文件路径
        """
        try:
            if not data_files:
                logger.warning("没有数据文件需要转换")
                return None
            
            # 过滤掉不存在的文件
            valid_files = []
            for file_path in data_files:
                if os.path.exists(file_path):
                    valid_files.append(file_path)
                else:
                    logger.warning(f"文件不存在: {file_path}")
            
            if not valid_files:
                logger.warning("没有有效的数据文件")
                return None
            
            # 获取转换后的数据
            recognition_data = self.transformer.get_recognition_data(valid_files)
            
            # 检查数据是否为None或空
            if recognition_data is None:
                logger.warning("数据转换返回了None")
                return None
            
            if hasattr(recognition_data, 'empty') and recognition_data.empty:
                logger.warning("转换后的数据为空")
                return None
            
            # 保存转换后的数据
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # 默认使用txt格式保存
            transformed_file = os.path.join(self.config['data_dir'], f'transformed_data_{timestamp}.csv')

            try:
                recognition_data.to_csv(transformed_file, index=False)
                logger.info(f"数据转换完成，保存至: {transformed_file}")
                return transformed_file
            except Exception as e:
                logger.error(f"保存转换后的数据失败: {e}")
                return None
        except Exception as e:
            logger.error(f"数据转换失败: {e}")
            return None
    
    def _recognize_scene(self, transformed_file):
        """识别场景
        
        Args:
            transformed_file: 转换后的数据文件路径
        
        Returns:
            识别出的场景名称
        """
        try:
            if not transformed_file or not os.path.exists(transformed_file):
                logger.warning("转换后的数据文件不存在")
                return None
            
            # 使用场景识别器识别场景
            scene = self.recognizer.recognize_scene(transformed_file)
            
            if scene:
                logger.info(f"场景识别结果: {scene}")
                return scene
            else:
                logger.warning("场景识别失败")
                return None
        except Exception as e:
            logger.error(f"场景识别出错: {e}")
            return None
    
    def _apply_parameters(self, scene):
        """应用场景参数
        
        Args:
            scene: 场景名称
        
        Returns:
            是否成功应用
        """
        try:
            if not scene:
                logger.warning("没有有效的场景信息，无法应用参数")
                return False
            
            # 使用参数优化器应用参数
            success = self.optimizer.apply_scene_parameters(scene)
            
            if success:
                logger.info(f"场景 '{scene}' 的参数应用成功")
                # 保存参数应用历史
                try:
                    # 安全地获取data_dir配置
                    data_dir = self.config.get('data_dir', os.path.dirname(os.path.abspath(__file__)))
                    history_file = os.path.join(data_dir, 'params_history.json')
                    self.optimizer.save_history_to_file(history_file)
                except Exception as e:
                    logger.warning(f"保存参数应用历史时出错: {e}")
            else:
                logger.warning(f"场景 '{scene}' 的参数应用失败")
            
            return success
        except Exception as e:
            logger.error(f"应用参数出错: {e}")
            return False
    
    def run_one_shot(self):
        """执行单次完整的数据收集、转换、识别和参数应用流程"""
        logger.info("开始执行一次监控模式...")
        
        try:
            # 1. 收集数据
            data_files = self._collect_data()
            
            # 2. 转换数据格式
            transformed_file = self._transform_data(data_files)
            
            if transformed_file:
                # 3. 在需要时加载模型识别场景
                if not self.recognizer:
                    model_path = self.config.get('scene_model_path', '/var/lib/industrial-scene-optimizer/models/scene_recognizer_model.pkl')
                    logger.info(f"尝试加载场景识别模型: {model_path}")
                    
                    # 确保模型目录存在
                    model_dir = os.path.dirname(model_path)
                    if model_dir and not os.path.exists(model_dir):
                        try:
                            os.makedirs(model_dir, exist_ok=True)
                            logger.info(f"已创建模型目录: {model_dir}")
                        except Exception as e:
                            logger.warning(f"创建模型目录失败: {e}")
                    
                    # 检查模型文件是否存在
                    if os.path.exists(model_path):
                        logger.info(f"模型文件存在: {model_path}")
                    else:
                        logger.warning(f"模型文件不存在: {model_path}")
                    
                    # 尝试加载模型
                    try:
                        # 延迟导入SceneRecognizer
                        from scene_recognizer import SceneRecognizer
                        self.recognizer = SceneRecognizer(model_path)
                    
                        # 检查模型是否成功加载
                        if not self.recognizer.is_model_loaded():
                            logger.warning(f"模型文件存在但加载失败: {model_path}，将尝试使用示例模式")
                            # 记录模型文件的基本信息，帮助诊断问题
                            try:
                                file_size = os.path.getsize(model_path) / 1024  # KB
                                logger.info(f"模型文件大小: {file_size:.2f} KB")
                                # 尝试以二进制模式打开文件，检查文件完整性
                                with open(model_path, 'rb') as f:
                                    header = f.read(10)  # 读取文件头
                                    logger.debug(f"模型文件头: {header}")
                            except Exception as file_e:
                                logger.warning(f"检查模型文件信息时出错: {file_e}")
                    except Exception as e:
                        logger.error(f"加载模型时发生异常: {type(e).__name__} - {str(e)}")
                        # 创建一个空的识别器实例以避免后续操作出错
                        from scene_recognizer import SceneRecognizer
                        self.recognizer = SceneRecognizer(None)
                        logger.warning("将使用示例模式继续运行")
                
                # 即使模型未加载，也继续执行流程
                if self.recognizer.is_model_loaded():
                    scene = self._recognize_scene(transformed_file)
                else:
                    # 模型未加载时，使用默认场景
                    logger.info("使用默认场景: light_load")
                    scene = "light_load"
                
                # 4. 在需要时初始化参数优化器并下发参数
                if scene and not self.optimizer:
                    # 延迟导入ParamOptimizer
                    from param_optimizer import ParamOptimizer
                    templates_dir = self.config.get('param_templates_dir', '/usr/share/industrial-scene-optimizer/templates')
                    # 将主配置对象传递给参数优化器，确保使用相同的配置文件
                    self.optimizer = ParamOptimizer(templates_dir, self.recognizer, self.config)
                
                # 5. 应用参数
                if scene:
                    self._apply_parameters(scene)
            
            logger.info("一次监控模式执行完成")
        except Exception as e:
            logger.error(f"一次监控模式执行出错: {e}")
            # 不再因为异常而中断服务，继续执行
    
    def run_continuous(self):
        """运行不间断监控模式"""
        logger.info("开始执行不间断监控模式...")
        logger.info(f"监控间隔: {self.config['collect_interval']}秒")
        
        self.running = True
        
        try:
            while self.running:
                try:
                    # 执行一次完整流程
                    self.run_one_shot()
                    
                    # 等待下一次执行
                    logger.info(f"等待{self.config['collect_interval']}秒后进行下一次监控...")
                    wait_time = 0
                    while wait_time < self.config['collect_interval'] and self.running:
                        time.sleep(1)
                        wait_time += 1
                    
                except Exception as e:
                    logger.error(f"监控循环执行出错: {e}")
                    # 继续执行，不中断循环
                    time.sleep(self.config['collect_interval'])
                    
        except KeyboardInterrupt:
            logger.info("接收到用户中断，停止监控")
        finally:
            self.running = False
            logger.info("不间断监控模式已停止")
    
    def run(self):
        """根据配置运行相应的监控模式"""
        try:
            if self.config['monitor_mode'] == 1:
                # 一次监控模式
                self.run_one_shot()
            else:
                # 不间断监控模式
                self.run_continuous()
        except Exception as e:
            logger.error(f"服务运行出错: {e}")
            # 不再强制退出，让服务能够继续运行或由systemd管理
            logger.info("服务遇到错误，但不会强制退出")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='工业场景优化器服务')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='配置文件路径')
    args = parser.parse_args()
    
    # 创建服务实例并运行
    service = ServiceMain(args.config)
    service.run()

if __name__ == "__main__":
    main()