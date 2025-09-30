#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动检测Linux环境并生成atune-collector配置文件"""

import os
import sys
import json
import subprocess
import re
import platform
from typing import List, Dict, Any

# 导入日志工具
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from logger_utils import get_logger

# 使用统一的日志记录器
logger = get_logger()

class AtuneConfigGenerator:
    """自动生成atune-collector配置文件的工具类"""
    
    def __init__(self, config=None):
        """初始化配置生成器"""
        self.system_info = {}
        self.config = config
        
        # 如果没有提供配置，尝试从默认位置加载
        if self.config is None:
            self._load_default_config()
        
        # 设置默认配置，优先从配置文件获取output_dir
        output_dir = self.config.get('raw_data_dir', None) if self.config else None
        if not output_dir:
            output_dir = self.config.get('data_dir', '/var/lib/industrial-scene-optimizer/data') if self.config else '/var/lib/industrial-scene-optimizer/data'
        
        # 按照generate_hpc_config.py的风格，将default_config定义为完整的配置结构
        self.default_config = {
            "sample_num": 25,
            "interval": 2,
            "output_dir": output_dir,
            "workload_type": "atune_collect",
            "network": "eth0",
            "block": "sda",
            "application": "firewalld,dockerd",
            "collection_items": self._get_default_collection_items()
        }
        
    def _load_default_config(self):
        """从默认位置加载配置文件"""
        try:
            # 首先从环境变量获取配置文件路径
            config_file = os.environ.get('SERVICE_CONFIG_PATH', None)
            
            # 如果环境变量未设置，使用脚本所在目录的配置文件
            if config_file is None or not os.path.exists(config_file):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                config_file = os.path.join(script_dir, "service_config.conf")
            
            # 加载配置文件
            if os.path.exists(config_file):
                self.config = self._load_conf_file(config_file)
                logger.info(f"成功加载配置文件: {config_file}")
            else:
                logger.warning(f"配置文件不存在: {config_file}")
                self.config = {}
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            self.config = {}
    
    def _load_conf_file(self, config_path):
        """加载.conf格式配置文件
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            配置字典
        """
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
        return config
        
    def _get_default_collection_items(self) -> List[Dict[str, Any]]:
        """获取默认的采集项配置，确保与atune-collector实际支持的字段一致"""
        return [
            {
                "name": "cpu",
                "module": "CPU",
                "purpose": "STAT",
                "metrics": [
                    "usr", "nice", "sys", "iowait", "irq", 
                    "soft", "steal", "guest", "util", "cutil"
                ],
                "threshold": 30
            },
            {
                "name": "storage",
                "module": "STORAGE",
                "purpose": "STAT",
                "metrics": [
                    "rs", "ws", "rMBs", "wMBs", "rrqm", "wrqm", 
                    "rareq-sz", "wareq-sz", "r_await", "w_await", "util", "aqu-sz"
                ]
            },
            {
                "name": "network",
                "module": "NET",
                "purpose": "STAT",
                "metrics": ["rxkBs", "txkBs", "rxpcks", "txpcks", "ifutil"]
            },
            {
                "name": "network-err",
                "module": "NET",
                "purpose": "ESTAT",
                "metrics": ["errs", "util"]
            },
            {
                "name": "meminfo",
                "module": "MEM",
                "purpose": "MEMINFO",
                "metrics": ["MemTotal", "MemFree", "MemAvailable", "SwapTotal", "Dirty"]
            },
            {
                "name": "mem.band",
                "module": "MEM",
                "purpose": "BANDWIDTH",
                "metrics": ["Total_Util"]
            },
            {
                "name": "perf",
                "module": "PERF",
                "purpose": "STAT",
                "metrics": [
                    "IPC", "CACHE-MISS-RATIO", "MPKI",
                    "ITLB-LOAD-MISS-RATIO", "DTLB-LOAD-MISS-RATIO", 
                    "SBPI", "SBPC"
                ]
            },
            {
                "name": "vmstat",
                "module": "MEM",
                "purpose": "VMSTAT",
                "metrics": [
                    "procs.b", "memory.swpd", "io.bo",
                    "system.in", "system.cs", "util.swap",
                    "util.cpu", "procs.r"
                ]
            },
            {
                "name": "sys.task",
                "module": "SYS",
                "purpose": "TASKS",
                "metrics": ["procs", "cswchs"]
            },
            {
                "name": "sys.ldavg",
                "module": "SYS",
                "purpose": "LDAVG",
                "metrics": ["runq-sz", "plist-sz", "ldavg-1", "ldavg-5"]
            },
            {
                "name": "file.util",
                "module": "SYS",
                "purpose": "FDUTIL",
                "metrics": ["fd-util"]
            },
            {
                "name": "process",
                "module": "PROCESS",
                "purpose": "SCHED",
                "metrics": [
                    "exec_start", "vruntime", "sum_exec_runtime", 
                    "switches", "voluntary_switches", "involuntary_switches"
                ]
            }
        ]
        
    def is_linux(self) -> bool:
        """检查当前系统是否为Linux"""
        return sys.platform.startswith('linux')
    
    def _run_command(self, cmd: str) -> str:
        """运行系统命令并返回输出结果"""
        try:
            result = subprocess.run(
                cmd, shell=True, check=True, capture_output=True, text=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.warning(f"命令执行失败: {cmd}, 错误: {e.stderr}")
            return ""
        except Exception as e:
            logger.warning(f"执行命令时出错: {cmd}, 错误: {str(e)}")
            return ""
    
    def _get_network_interfaces(self) -> List[str]:
        """获取Linux系统中的网络接口列表，确保能够适配不同Linux系统的设备命名"""
        interfaces = []
        
        # 方法1：使用ip命令 - 更现代的方法
        output = self._run_command("ip -o link show | awk -F': ' '{print $2}' | grep -v lo")
        if output:
            interfaces = [line.strip() for line in output.strip().split('\n')]
        
        # 方法2：检查/sys/class/net目录 - 更直接的方法
        if not interfaces and os.path.exists('/sys/class/net'):
            try:
                interfaces = [
                    f for f in os.listdir('/sys/class/net') 
                    if f != 'lo' and os.path.isdir(os.path.join('/sys/class/net', f))
                ]
                # 按字母顺序排序，确保一致性
                interfaces.sort()
            except Exception as e:
                logger.warning(f"读取网络接口失败: {str(e)}")
        
        # 方法3：使用ifconfig命令 - 兼容性更好的方法
        if not interfaces:
            output = self._run_command("ifconfig -a | grep '^[a-zA-Z0-9]' | awk '{print $1}' | grep -v lo")
            if output:
                interfaces = [line.strip().rstrip(':') for line in output.strip().split('\n')]
                # 按字母顺序排序
                interfaces.sort()
                
        # 方法4：特殊处理常见的网络接口命名模式
        if not interfaces and os.path.exists('/sys/class/net'):
            try:
                # 查找所有网络接口，包括以enp、wlp等现代命名方式开头的接口
                all_entries = os.listdir('/sys/class/net')
                # 过滤出非环回接口
                interfaces = [f for f in all_entries if f != 'lo']
                # 按字母顺序排序
                interfaces.sort()
            except Exception as e:
                print(f"查找网络接口失败: {str(e)}")
        
        return interfaces
    
    def _get_block_devices(self) -> List[str]:
        """获取Linux系统中的块设备列表，确保能够适配不同Linux系统的设备命名"""
        devices = []
        
        # 方法1：使用lsblk命令 - 推荐的现代方法
        output = self._run_command("lsblk -d -o NAME,TYPE | grep disk | awk '{print $1}'")
        if output:
            devices = [line.strip() for line in output.strip().split('\n')]
            # 按字母顺序排序，确保一致性
            devices.sort()
        
        # 方法2：检查/proc/partitions文件 - 更底层的方法
        if not devices and os.path.exists('/proc/partitions'):
            try:
                with open('/proc/partitions', 'r') as f:
                    lines = f.readlines()[2:]  # 跳过标题行
                    devices = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            dev_name = parts[3]
                            # 排除分区，只保留主设备
                            if not re.match(r'^[a-zA-Z]+\d+$', dev_name):
                                devices.append(dev_name)
                # 按字母顺序排序
                devices.sort()
            except Exception as e:
                logger.warning(f"读取块设备失败: {str(e)}")
        
        # 方法3：使用fdisk命令 - 兼容性更好的方法
        if not devices:
            output = self._run_command("fdisk -l | grep 'Disk /dev/' | awk -F'/' '{print $3}' | awk '{print $1}'")
            if output:
                devices = [line.strip() for line in output.strip().split('\n')]
                # 按字母顺序排序
                devices.sort()
                
        # 方法4：检查/dev目录中的常见块设备模式
        if not devices and os.path.exists('/dev'):
            try:
                # 搜索常见的块设备命名模式
                common_patterns = ['sd[a-z]+', 'hd[a-z]+', 'vd[a-z]+', 'nvme\\d+n\\d+', 'mmcblk\\d+']
                for pattern in common_patterns:
                    cmd = f"ls -1 /dev | grep -E '^{pattern}$' | head -5"
                    output = self._run_command(cmd)
                    if output:
                        pattern_devices = [line.strip() for line in output.strip().split('\n')]
                        # 将找到的设备添加到列表中（避免重复）
                        for dev in pattern_devices:
                            if dev not in devices:
                                devices.append(dev)
                # 按字母顺序排序
                devices.sort()
            except Exception as e:
                print(f"搜索块设备失败: {str(e)}")
        
        return devices
    
    def _get_running_applications(self) -> List[str]:
        """获取运行中的重要应用程序列表"""
        # 这里我们返回一些常见的系统服务作为示例
        common_services = ['firewalld', 'dockerd', 'sshd', 'nginx', 'apache2', 'mysql', 'postgresql']
        running_services = []
        
        # 检查systemd服务
        output = self._run_command("systemctl list-units --type=service --state=running | awk '{print $1}' | sed 's/\.service//g'")
        if output:
            systemd_services = [line.strip() for line in output.strip().split('\n')[1:]]  # 跳过标题
            for service in systemd_services:
                if service in common_services:
                    running_services.append(service)
        
        # 如果没有找到任何服务，返回默认值
        if not running_services:
            running_services = ['firewalld']
            print("警告: 未检测到运行中的服务，使用默认值")
        
        return running_services
        
    def _get_top_cpu_processes(self) -> List[str]:
        """获取系统中CPU利用率最高的5个进程"""
        try:
            # 使用ps命令获取CPU利用率最高的5个进程，排除ps本身和命令行参数
            # 格式说明：-eo表示自定义输出，%cpu表示CPU使用率，comm表示进程名
            # sort -k1 -nr表示按第一列（CPU使用率）降序排序
            # head -n 6获取前6行（第一行是标题），然后tail -n 5获取后5行（实际进程）
            # awk '{print $2}'获取进程名
            cmd = "ps -eo %cpu,comm --no-headers | sort -k1 -nr | head -n 6 | tail -n 5 | awk '{print $2}'"
            output = self._run_command(cmd)
            
            if output:
                # 按逗号分隔的字符串返回进程名列表
                processes = [line.strip() for line in output.strip().split('\n')]
                logger.info(f"检测到的CPU利用率最高的5个进程: {processes}")
                return processes
            else:
                logger.warning("警告: 获取CPU利用率最高的进程失败，使用默认服务列表")
                return self._get_running_applications()
        except Exception as e:
            logger.warning(f"获取CPU利用率最高的进程时出错: {str(e)}，使用默认服务列表")
            return self._get_running_applications()
    
    def _check_and_install_dependencies(self) -> None:
        """检查并安装必要的系统依赖包"""
        logger.info("检查必要的系统依赖...")
        
        # 检查sysstat包是否安装（提供sar命令）
        output = self._run_command("which sar || command -v sar")
        if not output:
            logger.info("未检测到sysstat包（包含sar命令），尝试安装...")
            
            # 检测Linux发行版和包管理器
            distro = platform.linux_distribution()[0].lower() if hasattr(platform, 'linux_distribution') else ""
            
            if "ubuntu" in distro or "debian" in distro:
                install_cmd = "apt-get update && apt-get install -y sysstat"
            elif "centos" in distro or "redhat" in distro or "fedora" in distro:
                install_cmd = "yum install -y sysstat || dnf install -y sysstat"
            else:
                logger.warning("警告: 无法确定Linux发行版，无法自动安装sysstat包")
                logger.warning("请手动安装sysstat包: sudo <包管理器> install sysstat")
                return
            
            # 尝试安装sysstat
            logger.info(f"执行安装命令: {install_cmd}")
            result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("sysstat包安装成功")
            else:
                logger.warning(f"警告: sysstat包安装失败: {result.stderr}")
                logger.warning("请手动安装sysstat包以使用网络监控功能")
        else:
            logger.info("sysstat包已安装")
    
    def detect_system_info(self) -> None:
        """检测系统信息，支持Linux和非Linux环境"""
        if not self.is_linux():
            logger.warning(f"警告: 当前系统({sys.platform})不是Linux环境")
            # 提供模拟的系统信息配置
            self._provide_mock_system_info()
            return
        
        # 系统信息检测的日志已在service_main.py中输出
        
        # 检查并安装必要的依赖
        self._check_and_install_dependencies()
        
        # 检测网络接口
        network_interfaces = self._get_network_interfaces()
        if network_interfaces:
            self.system_info['network'] = ','.join(network_interfaces[:3])  # 最多使用3个网络接口
            logger.info(f"检测到的网络接口: {self.system_info['network']}")
        else:
            # 智能选择默认网络接口名称，适配不同的Linux发行版和设备命名格式
            # 先检查系统中是否有常见的网络接口文件
            default_interfaces = ['eth0', 'enp0s3', 'ens33', 'ens160', 'ens192', 'enp1s0', 'wlan0', 'wlp2s0']
            found_interface = None
            
            # 检查/run/systemd/system判断是否使用systemd
            use_systemd = os.path.exists('/run/systemd/system')
            
            # 优先检查systemd风格的接口名称
            if use_systemd:
                # systemd风格的接口命名模式
                systemd_patterns = ['en', 'wl', 'ww', 'sl']
                # 先查看/dev目录中是否有常见的systemd风格接口
                if os.path.exists('/sys/class/net'):
                    try:
                        all_interfaces = os.listdir('/sys/class/net')
                        for iface in all_interfaces:
                            if iface != 'lo' and any(iface.startswith(pattern) for pattern in systemd_patterns):
                                found_interface = iface
                                break
                    except Exception:
                        pass
            
            # 如果没有找到systemd风格接口，尝试检查传统命名
            if not found_interface:
                for iface in default_interfaces:
                    if os.path.exists(f'/sys/class/net/{iface}'):
                        found_interface = iface
                        break
            
            # 设置默认接口
            if found_interface:
                self.system_info['network'] = found_interface
                logger.info(f"未检测到活跃网络接口，但找到系统中存在的接口: {found_interface}")
            else:
                # 如果都没找到，根据是否使用systemd选择默认模式
                if use_systemd:
                    # systemd环境优先使用ens样式（如ens160更常见于较新系统）
                    self.system_info['network'] = 'ens160'
                    logger.info("未检测到网络接口，使用常见的systemd风格默认值: ens160")
                else:
                    self.system_info['network'] = 'eth0'
                    logger.info("未检测到网络接口，使用传统默认值: eth0")
        
        # 检测块设备
        block_devices = self._get_block_devices()
        if block_devices:
            self.system_info['block'] = ','.join(block_devices[:2])  # 最多使用2个块设备
            logger.info(f"检测到的块设备: {self.system_info['block']}")
        else:
            # 尝试智能选择默认块设备名称，适配不同的存储设备类型
            # 智能检测和选择默认块设备，适配不同的存储设备类型和命名格式
            default_device = 'sda'  # 默认回退值
            
            # 定义常见的块设备类型和路径模式
            device_types = [
                {'path': '/sys/class/nvme', 'pattern': 'nvme0n1', 'desc': 'NVMe设备'},
                {'path': '/sys/class/mmc_host', 'pattern': 'mmcblk0', 'desc': 'MMC/SD卡设备'},
                {'path': '/dev/sda', 'pattern': 'sda', 'desc': '传统SATA设备'},
                {'path': '/dev/vda', 'pattern': 'vda', 'desc': '虚拟化设备'},
                {'path': '/dev/hd', 'pattern': 'hda', 'desc': 'IDE设备'}
            ]
            
            # 先检查/dev目录中实际存在的设备
            if os.path.exists('/dev'):
                try:
                    # 列出/dev目录中常见的块设备前缀
                    common_prefixes = ['sd', 'hd', 'vd', 'nvme', 'mmcblk']
                    dev_files = os.listdir('/dev')
                    
                    # 优先选择匹配常见模式的设备
                    for prefix in common_prefixes:
                        # 查找以该前缀开头的设备
                        matching_devices = [dev for dev in dev_files if dev.startswith(prefix)]
                        # 按字母顺序排序并选择第一个
                        if matching_devices:
                            matching_devices.sort()
                            # 选择第一个非分区设备（如sda而非sda1）
                            for dev in matching_devices:
                                # 简单判断是否为分区（适用于大多数情况）
                                if prefix == 'nvme':
                                    # NVMe设备格式通常是nvme0n1, nvme0n1p1等
                                    if 'p' not in dev.split('n')[1]:
                                        default_device = dev
                                        logger.info(f"在/dev目录中找到{prefix}类型设备: {default_device}")
                                        break
                                elif not re.match(r'^%s\\d+$' % prefix, dev):
                                    # 对于其他类型，排除纯数字后缀的设备（通常是分区）
                                    default_device = dev
                                    logger.info(f"在/dev目录中找到{prefix}类型设备: {default_device}")
                                    break
                            if default_device != 'sda':
                                break
                except Exception as e:
                    logger.warning(f"检查/dev目录设备时出错: {str(e)}")
            
            # 如果没有找到实际设备，基于系统特征选择默认设备类型
            if default_device == 'sda':  # 仍然是默认值，说明上面的检测没有找到设备
                try:
                    # 检查各种设备类型的特征路径
                    for device_type in device_types:
                        if os.path.exists(device_type['path']):
                            default_device = device_type['pattern']
                            logger.info(f"检测到{device_type['desc']}，使用默认值: {default_device}")
                            break
                    
                    # 特殊检查虚拟化环境
                    if default_device == 'sda' and os.path.exists('/proc/cpuinfo'):
                        with open('/proc/cpuinfo', 'r') as f:
                            cpuinfo = f.read()
                            if 'hypervisor' in cpuinfo.lower():
                                # 在虚拟机中，vda比sda更常见
                                default_device = 'vda'
                                logger.info("检测到虚拟化环境，使用默认值: vda")
                except Exception as e:
                    logger.warning(f"自动检测设备类型时出错: {str(e)}")
                    logger.warning("回退到默认块设备: sda")
            
            self.system_info['block'] = default_device
            logger.info(f"最终选择的块设备: {default_device}")
        
        # 在Linux环境下，获取CPU利用率最高的5个进程
        if self.is_linux():
            top_cpu_processes = self._get_top_cpu_processes()
            self.system_info['application'] = ','.join(top_cpu_processes)
            logger.info(f"使用CPU利用率最高的5个进程作为监控应用: {self.system_info['application']}")
        else:
            # 非Linux环境下，使用默认服务列表
            running_applications = self._get_running_applications()
            self.system_info['application'] = ','.join(running_applications)
            logger.info(f"检测到的运行中应用: {self.system_info['application']}")
        
        # 系统信息检测完成的日志已在service_main.py中输出
    
    def _provide_mock_system_info(self) -> None:
        """在非Linux环境下提供模拟的系统信息配置"""
        logger.info("在非Linux环境下提供模拟的系统信息配置")
        
        # 设置默认的网络接口和块设备
        self.system_info['network'] = 'eth0'
        self.system_info['block'] = 'sda'
        self.system_info['application'] = 'firewalld'
        
        logger.info(f"使用模拟的网络接口: {self.system_info['network']}")
        logger.info(f"使用模拟的块设备: {self.system_info['block']}")
        logger.info(f"使用模拟的应用程序: {self.system_info['application']}")
        logger.info("非Linux环境系统信息配置完成")
    
    def _calculate_used_memory(self, collection_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """添加内存使用量计算配置，确保与atune-collector的实际实现一致"""
        for item in collection_items:
            if item["name"] == "meminfo":
                # 确保不包含未在MemInfo类中定义的'used'字段，而是由atune-collector内部计算
                # 如果存在'used'字段，则移除它
                if "used" in item["metrics"]:
                    item["metrics"].remove("used")
                    logger.info("已从配置中移除'used'内存字段，该字段由atune-collector内部计算")
        return collection_items
        
    def _ensure_iostat_monitor(self, collection_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """不添加STORAGE-IOSTAT监视器配置，保持与用户要求一致"""
        return collection_items
    
    def generate_config(self) -> Dict[str, Any]:
        """生成配置文件内容，确保设备名称与系统实际设备匹配"""
        # 只提取与atune-collector相关的配置参数，避免包含多余的系统配置
        relevant_config = {}
        # 定义与atune-collector相关的配置项
        relevant_keys = ['sample_num', 'interval', 'output_dir', 'workload_type', 'network', 'block', 'application']
        
        # 从用户配置中提取相关参数
        if self.config:
            for key in relevant_keys:
                if key in self.config:
                    relevant_config[key] = self.config[key]
        
        # 合并默认配置、相关用户配置和系统信息
        config = {**self.default_config, **relevant_config, **self.system_info}
        
        # 修复内存使用量配置
        config["collection_items"] = self._calculate_used_memory(config["collection_items"])
        
        
        # 添加配置验证和兼容性检查
        # 配置验证的日志已在service_main.py中输出
        
        # 设置默认的设备名称配置（严格按照collect_data.json）
        default_devices = {
            'network': 'eth0',
            'block': 'sda',
            'application': 'firewalld,dockerd'
        }
        
        # 非Linux环境下直接使用collect_data.json中的默认设备配置
        if not self.is_linux():
            logger.info("非Linux环境下使用collect_data.json的默认设备配置")
            config.update(default_devices)
            return config
        
        # Linux环境下优先使用系统实际的网络接口，但保持与collect_data.json的兼容性
        logger.info("获取系统实际网络接口...")
        all_ifaces = self._get_network_interfaces()
        if all_ifaces:
            # 使用系统实际存在的网络接口
            config['network'] = all_ifaces[0]  # 使用第一个网络接口
            logger.info(f"已使用系统实际网络接口: {config['network']}")
        else:
            # 检查现有网络接口配置是否有效，无效则回退到默认值
            if 'network' in config:
                network_interfaces = config['network'].split(',')
                valid_interfaces = []
                for iface in network_interfaces:
                    # 检查接口是否存在于系统中
                    if os.path.exists(f'/sys/class/net/{iface}'):
                        valid_interfaces.append(iface)
                    else:
                        logger.warning(f"警告: 网络接口 {iface} 在系统中不存在")
                
                # 如果有有效的网络接口，使用它们
                if valid_interfaces:
                    config['network'] = valid_interfaces[0]
                    logger.info(f"已使用有效的网络接口: {config['network']}")
        
        # Linux环境下优先使用系统实际的块设备
        logger.info("获取系统实际块设备...")
        all_devices = self._get_block_devices()
        if all_devices:
            # 使用系统实际存在的块设备
            config['block'] = ','.join(all_devices[:2])  # 最多使用2个块设备
            logger.info(f"已使用系统实际块设备: {config['block']}")
        else:
            # 检查现有块设备配置是否有效，无效则回退到默认值
            if 'block' in config:
                block_devices = config['block'].split(',')
                valid_devices = []
                for dev in block_devices:
                    # 检查设备是否存在于系统中
                    if os.path.exists(f'/dev/{dev}'):
                        valid_devices.append(dev)
                    else:
                        logger.warning(f"警告: 块设备 {dev} 在系统中不存在")
                
                # 如果有有效的块设备，使用它们
                if valid_devices:
                    config['block'] = ','.join(valid_devices[:2])  # 最多使用2个块设备
                    logger.info(f"已使用有效的块设备: {config['block']}")
        
        # 配置文件兼容性验证完成的日志已在service_main.py中输出
        
        return config
    
    def save_config(self, config: Dict[str, Any], output_path: str = "collect_data.json") -> None:
        """保存配置文件"""
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"配置文件已保存到: {output_path}")
            logger.info(f"配置文件内容预览:\n{json.dumps(config, indent=2)[:200]}...")
        except Exception as e:
            logger.error(f"保存配置文件失败: {str(e)}")
            sys.exit(1)
    
    def run(self, output_path: str = None) -> None:
        """运行配置生成器"""
        # 如果未指定输出路径，则使用service_config.conf中配置的路径
        if output_path is None:
            try:
                # 尝试读取环境变量获取配置文件路径
                config_file = os.environ.get('SERVICE_CONFIG_PATH', None)
                if config_file is None or not os.path.exists(config_file):
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    config_file = os.path.join(script_dir, "service_config.conf")
                
                if os.path.exists(config_file):
                    # 使用已实现的_load_conf_file方法加载配置
                    service_config = self._load_conf_file(config_file)
                    if 'atune_config' in service_config:
                        output_path = service_config['atune_config']
                        logger.info(f"从service_config.conf中获取配置路径: {output_path}")
                    else:
                        output_path = "collect_data.json"
                        logger.warning("service_config.conf中未找到atune_config配置，使用默认路径")
                else:
                    output_path = "collect_data.json"
                    logger.warning(f"配置文件不存在: {config_file}，使用默认路径")
            except Exception as e:
                output_path = "collect_data.json"
                logger.error(f"读取service_config.conf失败: {str(e)}，使用默认路径")
        
        self.detect_system_info()
        config = self.generate_config()
        self.save_config(config, output_path)

if __name__ == "__main__":
    # 处理命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description='自动生成atune-collector配置文件')
    parser.add_argument('--output', '-o', type=str, default='collect_data.json',
                        help='输出配置文件路径')
    args = parser.parse_args()
    
    generator = AtuneConfigGenerator()
    generator.run(args.output)