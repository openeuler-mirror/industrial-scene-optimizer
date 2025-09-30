# 高性能计算自适应操作系统优化套件

# Industrial Scene Optimizer

## Project Overview

This project is developed based on the atune-collector tool and serves as an adaptive operating system optimization suite for High-Performance Computing (HPC) scenarios. It achieves two core objectives: "boot-time adaptive optimization" and "real-time dynamic tuning". Through a closed-loop process of data collection, scene recognition, parameter matching, and optimization execution, it provides optimal system parameter configurations for different HPC workload scenarios.

## Package Features

The system adopts a modular design, splitting its functionality into several core modules to form a complete closed-loop process of "data collection → data transformation → model training → scene recognition → configuration delivery":

### 1. Data Collection and Transformation
- **Data Collection**: Collects system performance metrics (CPU, memory, network, storage, etc.) through atune-collector
- **Data Transformation**: Converts raw data into standardized scene recognition data, grouped by 5-minute averages
- **Data Preprocessing**: Data cleaning, feature engineering, time window aggregation

### 2. Scene Recognition
- **Real-time Scene Recognition**: Automatically identifies the current running scene based on system performance data
- **Support for Multiple Scene Types**: Compute-intensive, data-intensive, hybrid workload, and light load
- **Probability Prediction**: Provides probability distribution for scene recognition to increase decision credibility

### 3. Parameter Optimization and Configuration Delivery
- **Automatic Parameter Application**: Automatically applies optimal parameter configurations based on recognized scenes
- **Support for Multiple Parameter Types**: sysctl parameters, I/O schedulers, thread limits, CPU governors, custom scripts, etc.
- **Parameter History Recording**: Records all parameter application history for query and analysis
- **System Original Parameter Backup and Restoration**: Supports one-click restoration of system original parameter state

### 4. Monitoring and Service Management
- **Real-time Monitoring Mode**: Continuously monitors system status and dynamically adjusts parameters
- **Single Monitoring Mode**: Performs single scene recognition and parameter optimization
- **Systemd Service Integration**: Supports boot auto-start and service management

## Project Directory Structure

```
├── install.sh                       # Installation/uninstallation script
├── src/                             # Source code directory
│   ├── service_main.py              # Main service entry
│   ├── data_transformer.py          # Data transformation module
│   ├── model_trainer.py             # Model training module
│   ├── scene_recognizer.py          # Scene recognition module
│   ├── param_optimizer.py           # Parameter optimization module
│   ├── restore_original_params.py   # Parameter restoration module
│   ├── industrial-scene-optimizer   # Command-line tool script
│   ├── restore_original_params      # Parameter restoration command-line script
│   └── service_config.conf          # Service configuration file
├── models/                          # Pre-trained model files
├── templates/                       # Scene parameter template directory
├── systemd/                         # systemd service configuration
└── README.md                        # Project description document
```

## Installation and Uninstallation

### System Requirements

- Operating System: Linux (openeuler)
- Python 3.8 or higher
- atune-collector tool (will be automatically checked and installed during installation)
- root user privileges (for installation and parameter configuration)

### Installation Steps

The project provides a one-click installation script to simplify the installation process:

```bash
# Execute the installation script as root user
sudo ./install.sh install
```

The installation process will automatically complete the following operations:
- Install system dependencies (automatically adapting to apt-get/dnf/yum based on different Linux distributions)
- Check and install the atune-collector module
- Copy Python module files to standard Python package directory
- Create configuration directory, data directory, and log directory
- Copy configuration files and scene parameter templates
- Set up systemd service and configure auto-start on boot
- Copy Performance_Data.csv file and pre-trained models

### Uninstallation Steps

```bash
# Execute the uninstallation script as root user
sudo ./install.sh uninstall
```

The uninstallation process will automatically complete the following operations:
- Stop and disable the systemd service
- Delete service files
- Delete Python packages and related files
- Delete configuration directory, data directory, and log directory
- Clean up all project-related files

## Main File Paths

After installation, the relevant file paths are as follows:
- Configuration file: `/etc/industrial-scene-optimizer/service_config.conf`
- Scene parameter templates: `/etc/industrial-scene-optimizer/templates/`
- Data storage: `/var/lib/industrial-scene-optimizer/data/`
- Model files: `/var/lib/industrial-scene-optimizer/models/`
- Log files: `/var/log/industrial-scene-optimizer/`
- Main service name: `industrial-scene-optimizer.service`
- Training data: `/usr/share/industrial_scene_optimizer/Performance_Data.csv`

## Service Management

After installation, you can manage the service using the following commands:

```bash
# Start the service
systemctl start industrial-scene-optimizer

# Check service status
systemctl status industrial-scene-optimizer

# Stop the service
systemctl stop industrial-scene-optimizer

# Enable auto-start on boot
systemctl enable industrial-scene-optimizer

# Disable auto-start on boot
systemctl disable industrial-scene-optimizer
```

## Usage Methods

### Command-line Tool

The project provides a command-line tool that can directly perform scene recognition and parameter optimization:

```bash
# Execute scene recognition and parameter optimization (requires root privileges)
industrial-scene-optimizer -c /etc/industrial-scene-optimizer/service_config.conf

# View help information
industrial-scene-optimizer --help
```

### Parameter Restoration Function

The system provides a parameter restoration tool that can restore system original parameter configurations with one click:

```bash
# Restore system original parameter configurations (requires root privileges)
sudo restore_original_params
```

### Model Training Function

The system provides tools for training models, which can directly recognize training data and train models
```bash
# Retrain the model
python3 /usr/lib/python3.11/site-packages/industrial_scene_optimizer/model_trainer.py
# Refer to the Performance_Data.csv file format for details
```

### Service Configuration

The service configuration file is located at `/etc/industrial-scene-optimizer/service_config.conf`, and the main configuration items include:

- `config_dir`: Configuration file directory
- `data_dir`: Data storage directory
- `model_dir`: Model storage directory
- `template_dir`: Scene parameter template directory
- `log_dir`: Log storage directory
- `log_file`: Log file name
- `log_level`: Log level
- `monitor_interval`: Monitoring interval time (seconds)

