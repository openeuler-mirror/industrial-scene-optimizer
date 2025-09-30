#!/bin/bash

# 工业场景优化器安装脚本
# 基于industrial-scene-optimizer.spec文件生成

set -e

# 源目录定义
SRC_DIR="src"
PYTHON_PACKAGE_NAME="industrial_scene_optimizer"

# 颜色定义
green='\033[0;32m'
red='\033[0;31m'
yellow='\033[0;33m'
neutral='\033[0m'

# 从配置文件中读取路径配置
if [ -f "$SRC_DIR/service_config.conf" ]; then
    # 读取.conf格式配置文件
    CONFIG_DIR=$(grep '^config_dir=' "$SRC_DIR/service_config.conf" | cut -d '=' -f 2 | tr -d ' "')
    DATA_DIR=$(grep '^data_dir=' "$SRC_DIR/service_config.conf" | cut -d '=' -f 2 | tr -d ' "')
    OPT_DIR=$(grep '^opt_dir=' "$SRC_DIR/service_config.conf" | cut -d '=' -f 2 | tr -d ' "')
    TEMPLATES_DIR=$(grep '^param_templates_dir=' "$SRC_DIR/service_config.conf" | cut -d '=' -f 2 | tr -d ' "')
    SYSTEMD_DIR=$(grep '^systemd_dir=' "$SRC_DIR/service_config.conf" | cut -d '=' -f 2 | tr -d ' "')
    SCENE_MODEL_PATH=$(grep '^scene_model_path=' "$SRC_DIR/service_config.conf" | cut -d '=' -f 2 | tr -d ' "')
    PERFORMANCE_DATA_PATH=$(grep '^performance_data_path=' "$SRC_DIR/service_config.conf" | cut -d '=' -f 2 | tr -d ' "')
    LOG_FILE=$(grep '^log_file=' "$SRC_DIR/service_config.conf" | cut -d '=' -f 2 | tr -d ' "')
    
    # 计算派生路径
    SCENE_MODEL_DIR=$(dirname "$SCENE_MODEL_PATH")
    LOG_DIR=$(dirname "$LOG_FILE")
else
    echo -e "${red}错误: 未找到 $SRC_DIR/service_config.conf 文件，无法获取路径配置${neutral}"
    exit 1
fi

# 参数解析
ACTION="install"  # 默认安装

# 显示帮助信息
show_help() {
    echo -e "${green}工业场景优化器安装脚本${neutral}"
    echo -e "用法: $0 [选项]"
    echo -e "选项:"
    echo -e "  -h, --help      显示此帮助信息"
    echo -e "  --install       安装工业场景优化器（默认）"
    echo -e "  --uninstall     卸载工业场景优化器"
    exit 0
}

# 解析命令行参数
while [ $# -gt 0 ]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --install)
            ACTION="install"
            shift
            ;;
        --uninstall)
            ACTION="uninstall"
            shift
            ;;
        *)
            echo -e "${red}未知选项: $1${neutral}"
            show_help
            ;;
    esac
done

# 检查是否以root用户运行
if [ "$(id -u)" != "0" ]; then
   echo -e "${red}此脚本需要以root用户权限运行，请使用sudo。${neutral}"
   exit 1
fi

# 根据操作类型执行不同的功能
if [ "$ACTION" = "uninstall" ]; then
    echo -e "${green}=============================${neutral}"
    echo -e "${green}工业场景优化器卸载脚本${neutral}"
    echo -e "${green}=============================${neutral}"
    
    # 停止服务
    echo -e "${green}[1/3] 停止服务...${neutral}"
    systemctl stop industrial-scene-optimizer.service 2>/dev/null || echo "服务未运行"
    systemctl disable industrial-scene-optimizer.service 2>/dev/null || echo "服务未设置开机自启"
    
    # 删除服务文件
    echo -e "${green}[2/3] 删除服务文件...${neutral}"
    rm -f ${SYSTEMD_DIR}/industrial-scene-optimizer.service
    systemctl daemon-reload
    
    # 删除Python包和文件
    echo -e "${green}[3/4] 删除Python包和文件...${neutral}"
    rm -f /usr/sbin/industrial-scene-optimizer
    rm -f /usr/sbin/restore_original_params
    
    # 获取实际的Python版本路径
    PYTHON_LIB_PATH=$(realpath /usr/lib/python3* | head -n 1)/site-packages
    rm -rf ${PYTHON_LIB_PATH}/${PYTHON_PACKAGE_NAME}
    
    # 删除创建的目录和文件
    echo -e "${green}[4/4] 删除创建的目录和文件...${neutral}"
    
    # 删除配置目录
    if [ -d "${CONFIG_DIR}" ]; then
        rm -rf ${CONFIG_DIR}
        echo "已删除配置目录: ${CONFIG_DIR}"
    fi
    
    # 删除数据目录
    if [ -d "${DATA_DIR}" ]; then
        rm -rf ${DATA_DIR}
        echo "已删除数据目录: ${DATA_DIR}"
    fi
    
    # 删除模板目录
    if [ -d "${TEMPLATES_DIR}" ]; then
        rm -rf ${TEMPLATES_DIR}
        echo "已删除模板目录: ${TEMPLATES_DIR}"
    fi
    
    # 删除日志目录
    if [ -d "${LOG_DIR}" ]; then
        rm -rf ${LOG_DIR}
        echo "已删除日志目录: ${LOG_DIR}"
    fi
    
    # 删除模型目录
    if [ -d "${SCENE_MODEL_DIR}" ]; then
        rm -rf ${SCENE_MODEL_DIR}
        echo "已删除模型目录: ${SCENE_MODEL_DIR}"
    fi
    
    # 删除性能数据文件目录
    PERFORMANCE_DATA_DIR=$(dirname "$PERFORMANCE_DATA_PATH")
    if [ -d "${PERFORMANCE_DATA_DIR}" ]; then
        rm -rf ${PERFORMANCE_DATA_DIR}
        echo "已删除性能数据文件目录: ${PERFORMANCE_DATA_DIR}"
    fi
    
    echo -e "${green}\n=============================${neutral}"
    echo -e "${green}卸载完成!${neutral}"
    echo -e "${green}=============================${neutral}"
    exit 0
else
    # 打印欢迎信息
    echo -e "${green}=============================${neutral}"
    echo -e "${green}工业场景优化器安装脚本${neutral}"
    echo -e "${green}=============================${neutral}"
fi

# 1. 安装系统依赖
 echo -e "${green}[1/7] 安装系统依赖...${neutral}"
if command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y python3 python3-pip python3-setuptools python3-pandas python3-matplotlib python3-scikit-learn python3-psutil atune-collector python3-cycler python3-pyaml linux-tools-common linux-tools-generic perf iproute2 net-tools util-linux coreutils gawk grep sed star sysstat
elif command -v dnf &> /dev/null; then
    dnf install -y python3 python3-pip python3-setuptools python3-pandas python3-matplotlib python3-scikit-learn python3-psutil atune-collector python3-dnf python3-cycler python3-pyyaml perf iproute net-tools util-linux coreutils gawk grep sed star sysstat
elif command -v yum &> /dev/null; then
    yum install -y python3 python3-pip python3-setuptools python3-pandas python3-matplotlib python3-scikit-learn python3-psutil atune-collector python3-cycler python3-pyyaml perf iproute net-tools util-linux coreutils gawk grep sed star sysstat
else
    echo -e "${red}不支持的包管理器。请手动安装以下依赖: python3 python3-pip python3-setuptools python3-pandas python3-matplotlib python3-scikit-learn python3-psutil atune-collector python3-cycler python3-pyaml iproute2 net-tools util-linux coreutils gawk grep sed star sysstat${neutral}"
fi

# 2. 安装Python包
 echo -e "${green}[2/7] 安装Python包...${neutral}"

# 获取实际的Python版本路径
PYTHON_LIB_PATH=$(realpath /usr/lib/python3* | head -n 1)/site-packages

# 创建Python包目录结构
mkdir -p ${PYTHON_LIB_PATH}/${PYTHON_PACKAGE_NAME}

# 复制所有Python模块文件到包目录
PYTHON_FILES="service_main.py data_transformer.py model_trainer.py scene_recognizer.py param_optimizer.py generate_atune_config.py performance_data_reader.py restore_original_params.py logger_utils.py"
for file in $PYTHON_FILES; do
    if [ -f "$SRC_DIR/$file" ]; then
        cp "$SRC_DIR/$file" ${PYTHON_LIB_PATH}/${PYTHON_PACKAGE_NAME}/
        echo "复制 $SRC_DIR/$file 到 ${PYTHON_LIB_PATH}/${PYTHON_PACKAGE_NAME}/"
    else
        echo -e "${red}警告: 未找到文件 $SRC_DIR/$file${neutral}"
    fi
done

# 首先尝试复制已存在的__init__.py文件
if [ -f "$SRC_DIR/__init__.py" ]; then
    cp "$SRC_DIR/__init__.py" ${PYTHON_LIB_PATH}/${PYTHON_PACKAGE_NAME}/
    echo "复制 $SRC_DIR/__init__.py 到 ${PYTHON_LIB_PATH}/${PYTHON_PACKAGE_NAME}/"
fi

# 检查系统中是否已安装atune-collector
 echo -e "${green}检查atune-collector模块...${neutral}"
if python3 -c "import atune_collector" &> /dev/null; then
    echo -e "${green}atune-collector模块已在系统中安装，跳过复制步骤${neutral}"
    echo "系统atune-collector路径: /usr/lib/python3.11/site-packages/atune_collector"
else
    echo -e "${red}错误: 未在系统中找到atune-collector模块，请先安装atune-collector包${neutral}"
    exit 1
fi

# 创建启动脚本
if [ -f "$SRC_DIR/industrial-scene-optimizer" ]; then
    cp "$SRC_DIR/industrial-scene-optimizer" /usr/sbin/
    chmod +x /usr/sbin/industrial-scene-optimizer
    echo "复制 $SRC_DIR/industrial-scene-optimizer 到 /usr/sbin/"
else
    echo -e "${red}错误: 未找到 $SRC_DIR/industrial-scene-optimizer 文件${neutral}"
    exit 1
fi

# 创建restore_original_params命令
if [ -f "$SRC_DIR/restore_original_params" ]; then
    cp "$SRC_DIR/restore_original_params" /usr/sbin/
    chmod +x /usr/sbin/restore_original_params
    echo "复制 $SRC_DIR/restore_original_params 到 /usr/sbin/"
else
    echo -e "${red}错误: 未找到 $SRC_DIR/restore_original_params 文件${neutral}"
    exit 1
fi

# 3. 创建目录结构和配置文件
 echo -e "${green}[3/7] 创建目录结构和配置文件...${neutral}"

# 创建配置目录
mkdir -p ${CONFIG_DIR}

# 创建数据目录和日志目录
mkdir -p ${DATA_DIR}
mkdir -p ${SCENE_MODEL_DIR}
mkdir -p ${LOG_DIR}

# 设置目录权限
chmod -R 755 ${DATA_DIR}
chmod -R 755 ${LOG_DIR}

# 复制配置文件
if [ -f "$SRC_DIR/service_config.conf" ]; then
    cp "$SRC_DIR/service_config.conf" ${CONFIG_DIR}/
    chmod 644 ${CONFIG_DIR}/service_config.conf
    echo "复制 $SRC_DIR/service_config.conf 到 ${CONFIG_DIR}/"
else
    echo -e "${red}警告: 未找到 $SRC_DIR/service_config.conf 文件${neutral}"
fi

# 配置文件将在首次运行时自动生成
 echo -e "${green}collect_data.json配置文件将在服务首次启动时自动生成${neutral}"

# 4. 安装参数模板文件
 echo -e "${green}[4/7] 安装参数模板文件...${neutral}"

mkdir -p ${TEMPLATES_DIR}
if [ -d "templates" ]; then
    cp -r templates/*.yaml ${TEMPLATES_DIR}/
    chmod 644 ${TEMPLATES_DIR}/*.yaml
    echo "复制模板文件到 ${TEMPLATES_DIR}/"
else
    echo -e "${red}警告: 未找到 templates 目录${neutral}"
fi

# 5. 设置systemd服务
 echo -e "${green}[5/7] 设置systemd服务...${neutral}"

# 获取实际的Python版本路径
ACTUAL_PYTHON_PATH=$(realpath /usr/bin/python3* | head -n 1)
ACTUAL_PYTHON_LIB_PATH=$(realpath /usr/lib/python3* | head -n 1)/site-packages

# 复制systemd服务文件并根据路径判断是否需要更新
if [ -f "systemd/industrial-scene-optimizer.service" ]; then
    # 复制原始服务文件
    cp "systemd/industrial-scene-optimizer.service" ${SYSTEMD_DIR}/industrial-scene-optimizer.service
    
    # 检查Python库路径是否为/usr/lib/python3.11/site-packages
    if [[ "$ACTUAL_PYTHON_LIB_PATH" != "/usr/lib/python3.11/site-packages" ]]; then
        # 路径不是/usr/lib/python3.11时，更新服务文件中的Python路径
        sed -i "s|/usr/bin/python3.11|${ACTUAL_PYTHON_PATH}|g" ${SYSTEMD_DIR}/industrial-scene-optimizer.service
        echo "更新服务文件中的Python路径到 ${ACTUAL_PYTHON_PATH}"  
        echo "使用实际Python库路径: ${ACTUAL_PYTHON_LIB_PATH}"
    else
        # 路径是/usr/lib/python3.11时，不修改服务文件
        echo "使用默认Python路径 /usr/lib/python3.11，不修改服务文件"
    fi
    
    chmod 644 ${SYSTEMD_DIR}/industrial-scene-optimizer.service
    echo "复制服务文件到 ${SYSTEMD_DIR}/industrial-scene-optimizer.service"
    
    # 重载systemd配置
    systemctl daemon-reload
    
    # 设置服务开机自启
    systemctl enable industrial-scene-optimizer.service
    echo "已设置服务开机自启"
else
    echo -e "${red}警告: 未找到 systemd/industrial-scene-optimizer.service 文件${neutral}"
fi

# 创建数据目录
mkdir -p ${DATA_DIR}
chmod 755 ${DATA_DIR}

# 6. 复制Performance_Data.csv文件
 echo -e "${green}[6/7] 复制Performance_Data.csv文件...${neutral}"

# 创建目标目录
mkdir -p $(dirname "$PERFORMANCE_DATA_PATH")

# 复制Performance_Data.csv文件
if [ -f "$SRC_DIR/Performance_Data.csv" ]; then
    cp "$SRC_DIR/Performance_Data.csv" "$PERFORMANCE_DATA_PATH"
    chmod 644 "$PERFORMANCE_DATA_PATH"
    echo "复制 $SRC_DIR/Performance_Data.csv 到 $PERFORMANCE_DATA_PATH"
else
    echo -e "${yellow}警告: 未找到 $SRC_DIR/Performance_Data.csv 文件${neutral}"
fi

# 7. 复制模型文件
 echo -e "${green}[7/7] 复制模型文件...${neutral}"

# 创建模型目录
mkdir -p "$SCENE_MODEL_DIR"

# 复制模型文件
if [ -d "models" ]; then
    cp -r models/* "$SCENE_MODEL_DIR/"
    chmod 644 "$SCENE_MODEL_DIR"/*.pkl
    echo "复制模型文件到 $SCENE_MODEL_DIR/"
else
    echo -e "${yellow}警告: 未找到 models 目录${neutral}"
fi

# 如果src目录下存在__init__.py文件，也复制它
if [ -f "$SRC_DIR/__init__.py" ]; then
    cp "$SRC_DIR/__init__.py" ${PYTHON_LIB_PATH}/${PYTHON_PACKAGE_NAME}/
    echo "复制 $SRC_DIR/__init__.py 到 ${PYTHON_LIB_PATH}/${PYTHON_PACKAGE_NAME}/"
fi

# 安装完成信息
 echo -e "${green}\n=============================${neutral}"
 echo -e "${green}安装完成!${neutral}"
 echo -e "${green}=============================${neutral}"
 echo -e "${green}1. 配置文件位置:${neutral} ${CONFIG_DIR}/service_config.conf"
 echo -e "${green}2. 模板文件位置:${neutral} ${TEMPLATES_DIR}/"
 echo -e "${green}3. 数据存储位置:${neutral} ${DATA_DIR}"
 echo -e "${green}4. 服务名称:${neutral} industrial-scene-optimizer.service"
 echo -e "${green}5. 模型文件位置:${neutral} $SCENE_MODEL_DIR"
 echo -e "${green}6. 性能数据文件位置:${neutral} $PERFORMANCE_DATA_PATH"
 echo -e "\n${green}启动服务命令:${neutral} systemctl start industrial-scene-optimizer.service"
 echo -e "${green}查看服务状态:${neutral} systemctl status industrial-scene-optimizer.service"
 echo -e "${green}停止服务命令:${neutral} systemctl stop industrial-scene-optimizer.service"
 echo -e "\n${green}或直接使用命令行工具:${neutral} industrial-scene-optimizer --help"
 echo -e "\n${green}恢复系统原始参数:${neutral} restore_original_params" "\n${green}如需卸载，请运行:${neutral} sudo $0 --uninstall"