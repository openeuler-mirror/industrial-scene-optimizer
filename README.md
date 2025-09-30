# 高性能计算自适应操作系统优化套件

# Industrial Scene Optimizer

## 项目概述

本项目基于atune-collector工具开发，是一个高性能计算(HPC)场景下的自适应操作系统优化套件，实现"开机自适应优化"与"实时动态调优"两大核心目标。通过数据采集、场景识别、参数匹配、优化执行的闭环，为不同HPC负载场景提供最优的系统参数配置。

## 软件包功能

系统采用模块化设计，将功能拆分为多个核心模块，形成完整的"数据采集→数据转换→模型训练→场景识别→配置下发"闭环流程：

### 1. 数据采集与转换
- **数据采集**：通过atune-collector采集系统性能指标（CPU、内存、网络、存储等）
- **数据转换**：将原始数据转换为标准化的场景识别数据，以5分钟平均值为一组
- **数据预处理**：数据清洗、特征工程、时间窗口聚合

### 2. 场景识别
- **实时场景识别**：根据系统性能数据自动识别当前运行场景
- **支持多种场景类型**：计算密集型、数据密集型、混合负载型、轻量负载型
- **概率预测**：提供场景识别的概率分布，增加决策可信度

### 3. 参数优化与配置下发
- **自动参数应用**：根据识别的场景自动应用最优参数配置
- **多类型参数支持**：sysctl参数、I/O调度器、线程数限制、CPU调节器、自定义脚本等
- **参数历史记录**：记录所有参数应用历史，支持查询和分析
- **系统原始参数备份与恢复**：支持一键恢复系统原始参数状态

### 4. 监控与服务管理
- **实时监控模式**：不间断监控系统状态并动态调整参数
- **单次监控模式**：执行单次场景识别和参数优化
- **Systemd服务集成**：支持开机自启和服务管理

## 项目目录结构

```
├── install.sh                       # 安装/卸载脚本
├── src/                             # 源代码目录
│   ├── service_main.py              # 主服务入口
│   ├── data_transformer.py          # 数据转换模块
│   ├── model_trainer.py             # 模型训练模块
│   ├── scene_recognizer.py          # 场景识别模块
│   ├── param_optimizer.py           # 参数优化模块
│   ├── restore_original_params.py   # 参数恢复模块
│   ├── industrial-scene-optimizer   # 命令行工具脚本
│   ├── restore_original_params      # 参数恢复命令行脚本
│   └── service_config.conf          # 服务配置文件
├── models/                          # 预训练模型文件
├── templates/                       # 场景参数模板目录
├── systemd/                         # systemd服务配置
└── README.md                        # 项目说明文档
```

## 安装与卸载

### 系统要求

- 操作系统：Linux (openeuer)
- Python 3.8 或更高版本
- atune-collector 工具（会在安装过程中自动检查并安装）
- root 用户权限（用于安装和参数配置）

### 安装步骤

项目提供了一键安装脚本，简化安装过程：

```bash
# 以root用户执行安装脚本
sudo ./install.sh install
```

安装过程会自动完成以下操作：
- 安装系统依赖（根据不同Linux发行版自动适配apt-get/dnf/yum）
- 检查并安装atune-collector模块
- 复制Python模块文件到标准Python包目录
- 创建配置目录、数据目录、日志目录
- 复制配置文件和场景参数模板
- 设置systemd服务并配置开机自启
- 复制Performance_Data.csv文件和预训练模型

### 卸载步骤

```bash
# 以root用户执行卸载脚本
sudo ./install.sh uninstall
```

卸载过程会自动完成以下操作：
- 停止并禁用systemd服务
- 删除服务文件
- 删除Python包和相关文件
- 删除配置目录、数据目录、日志目录
- 清理所有与项目相关的文件

## 主要文件路径

安装完成后，相关文件路径如下：
- 配置文件：`/etc/industrial-scene-optimizer/service_config.conf`
- 场景参数模板：`/etc/industrial-scene-optimizer/templates/`
- 数据存储：`/var/lib/industrial-scene-optimizer/data/`
- 模型文件：`/var/lib/industrial-scene-optimizer/models/`
- 日志文件：`/var/log/industrial-scene-optimizer/`
- 主服务名称：`industrial-scene-optimizer.service`
- 训练数据：：`/usr/share/industrial_scene_optimizer/Performance_Data.csv`

## 服务管理

安装完成后，可以通过以下命令管理服务：

```bash
# 启动服务
systemctl start industrial-scene-optimizer

# 查看服务状态
systemctl status industrial-scene-optimizer

# 停止服务
systemctl stop industrial-scene-optimizer

# 开机自启
systemctl enable industrial-scene-optimizer

# 禁用开机自启
systemctl disable industrial-scene-optimizer
```

## 使用方法

### 命令行工具

项目提供了命令行工具，可以直接执行场景识别和参数优化：

```bash
# 执行场景识别和参数优化（需要root权限）
industrial-scene-optimizer -c /etc/industrial-scene-optimizer/service_config.conf

# 查看帮助信息
industrial-scene-optimizer --help
```

### 参数恢复功能

系统提供了参数恢复工具，可以一键恢复系统原始参数配置：

```bash
# 恢复系统原始参数配置（需要root权限）
sudo restore_original_params
```

### 模型训练功能

系统提供了训练模型的工具，可以直接识别训练数据和训练模型
```bash
# 重新训练模型
python3 /usr/lib/python3.11/site-packages/industrial_scene_optimizer/model_trainer.py
#具体参考Performance_Data.csv文件格式
```

### 服务配置

服务配置文件位于 `/etc/industrial-scene-optimizer/service_config.conf`，主要配置项包括：

- `config_dir`: 配置文件目录
- `data_dir`: 数据存储目录
- `model_dir`: 模型存储目录
- `template_dir`: 场景参数模板目录
- `log_dir`: 日志存储目录
- `log_file`: 日志文件名
- `log_level`: 日志级别
- `monitor_interval`: 监控间隔时间（秒）

