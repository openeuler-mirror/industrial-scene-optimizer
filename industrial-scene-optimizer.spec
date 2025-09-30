Name: industrial-scene-optimizer
Version: 1.0.0
Release: 1
Summary: Industrial Scene Optimizer for HPC environments
License: Mulan PSL v2
URL: https://gitee.com/openeuler/industrial-scene-optimizer
Source: https://gitee.com/openeuler/industrial-scene-optimizer/repository/archive/v%{version}.tar.gz
BuildRequires: python3-setuptools
Requires: python3 python3-pip python3-setuptools python3-pandas python3-matplotlib python3-scikit-learn python3-psutil atune-collector python3-cycler python3-pyyaml

%description
Industrial Scene Optimizer is a comprehensive tool for HPC environments that performs data collection, scene recognition, and parameter optimization to improve system performance. It collects system metrics, recognizes workload patterns, and applies optimal configuration parameters based on identified scenes.

%prep
%autosetup -n industrial-scene-optimizer-v%{version} -p1

%build
%{py3_build}

%install
%{py3_install}

# 创建启动脚本
mkdir -p %{buildroot}%{_sbindir}
install -m 755 src/industrial-scene-optimizer %{buildroot}%{_sbindir}/
install -m 755 src/restore_original_params %{buildroot}%{_sbindir}/

# 安装配置文件和服务文件
mkdir -p %{buildroot}%{_sysconfdir}/industrial-scene-optimizer
install -m 644 src/service_config.conf %{buildroot}%{_sysconfdir}/industrial-scene-optimizer/

# 安装systemd服务文件
mkdir -p %{buildroot}%{_unitdir}
install -m 644 systemd/industrial-scene-optimizer.service %{buildroot}%{_unitdir}/industrial-scene-optimizer.service

# 安装参数模板文件
mkdir -p %{buildroot}%{_datadir}/industrial-scene-optimizer/templates
install -m 644 templates/*.yaml %{buildroot}%{_datadir}/industrial-scene-optimizer/templates/

# 创建数据目录
mkdir -p %{buildroot}%{_localstatedir}/lib/industrial-scene-optimizer/data

# 创建模型目录
mkdir -p %{buildroot}%{_localstatedir}/lib/industrial-scene-optimizer/models

# 创建日志目录
mkdir -p %{buildroot}%{_localstatedir}/log/industrial-scene-optimizer

# 安装性能数据文件
mkdir -p %{buildroot}%{_datadir}/industrial-scene-optimizer
install -m 644 src/Performance_Data.csv %{buildroot}%{_datadir}/industrial-scene-optimizer/

# 安装预训练模型文件
install -m 644 models/*.pkl %{buildroot}%{_localstatedir}/lib/industrial-scene-optimizer/models/

%check
# 可以在这里添加测试命令

%files
%license LICENSE
%doc README.md README_en.md
%{python3_sitelib}/industrial_scene_optimizer*
%{_sbindir}/industrial-scene-optimizer
%{_sbindir}/restore_original_params
%config(noreplace) %{_sysconfdir}/industrial-scene-optimizer/service_config.conf
%{_unitdir}/industrial-scene-optimizer.service
%{_datadir}/industrial-scene-optimizer/templates/*
%{_datadir}/industrial-scene-optimizer/Performance_Data.csv
%{_localstatedir}/lib/industrial-scene-optimizer/data
%{_localstatedir}/lib/industrial-scene-optimizer/models
%{_localstatedir}/log/industrial-scene-optimizer

%changelog
* %{date '+%a %b %d %Y'} Author <author@example.com> - %{version}-%{release}
- Initial package creation
- Added core functionality for data collection, transformation, scene recognition and parameter optimization
- Included systemd service configuration
- Added parameter templates for different workload scenes
- Fixed configuration file path to match install.sh script
- Added performance data file and pre-trained model files
- Added necessary directories creation for logs, data and models