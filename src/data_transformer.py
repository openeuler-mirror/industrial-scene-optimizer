# -*- coding: utf-8 -*-
"""数据转换器 - 将原始数据转换为识别数据"""

import os
import json
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

# 导入日志工具
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from logger_utils import get_logger

# 使用统一的日志记录器
logger = get_logger()

class DataTransformer:
    """数据转换器 - 将原始数据转换为识别数据"""
    
    def __init__(self, config=None):
        """初始化数据转换器
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        # 支持的编码列表
        self.encodings = ['utf-8', 'gb2312', 'gbk', 'ansi', 'latin1']
    
    def _load_conf_file(self, file_path):
        """加载.conf格式的配置文件
        
        Args:
            file_path: 配置文件路径
            
        Returns:
            配置字典
        """
        config = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 忽略空行和注释行
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # 解析键值对
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # 尝试将值转换为适当的类型
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.isdigit():
                            value = int(value)
                        elif '.' in value and all(part.isdigit() for part in value.split('.') if part):
                            value = float(value)
                        
                        config[key] = value
                
            return config
        except Exception as e:
            logger.error(f"解析配置文件失败: {e}")
            return {}
        
        # 加载配置文件
        self.config = config
        if self.config is None:
            # 首先从环境变量获取配置文件路径
            config_file = os.environ.get('SERVICE_CONFIG_PATH', None)
            
            # 如果环境变量未设置，使用脚本所在目录的配置文件
            if config_file is None or not os.path.exists(config_file):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                config_file = os.path.join(script_dir, "service_config.conf")
            
            # 加载.conf格式配置文件
            if os.path.exists(config_file):
                try:
                    self.config = self._load_conf_file(config_file)
                    logger.info(f"成功加载配置文件: {config_file}")
                except Exception as e:
                    logger.error(f"加载配置文件失败: {e}")
                    self.config = {}
        
        # 设置数据目录
        self.data_dir = self._get_data_directory()
    
    def _get_data_directory(self):
        """从配置文件获取数据目录路径
        
        Returns:
            数据目录路径
        """
        if self.config and "data_dir" in self.config:
            data_dir = self.config["data_dir"]
        elif self.config and "output_dir" in self.config:
            data_dir = self.config["output_dir"]
        else:
            # 默认使用脚本所在目录下的data文件夹
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, "data")
        
        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"数据目录已设置为: {data_dir}")
        
        return data_dir
    
    def _read_file_with_encoding(self, file_path):
        """使用多种编码尝试读取CSV文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            包含数据的DataFrame或None
        """
        for encoding in self.encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"成功使用{encoding}编码读取CSV文件: {file_path}")
                return df
            except UnicodeDecodeError:
                logger.warning(f"使用{encoding}编码读取CSV文件失败: {file_path}")
                continue
            except Exception as e:
                logger.warning(f"使用{encoding}编码读取CSV文件时发生其他错误: {str(e)}, 文件: {file_path}")
                continue
        
        # 最后尝试使用utf-8并替换错误字符
        try:
            df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
            logger.info(f"成功使用utf-8(替换错误字符)编码读取CSV文件: {file_path}")
            return df
        except Exception as e:
            logger.error(f"所有编码尝试都失败: {str(e)}, 文件: {file_path}")
            return None
    
    def load_raw_data(self, file_path):
        """从CSV文件加载原始数据
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            包含原始数据的DataFrame或None
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return None
        
        try:
            # 尝试使用多种编码读取CSV文件
            df = self._read_file_with_encoding(file_path)
            
            if df is None:
                return None
            
            # 尝试转换时间戳列
            if 'TimeStamp' in df.columns:
                try:
                    # 尝试多种时间格式
                    time_formats = ['%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S']
                    timestamp_converted = False
                    
                    for fmt in time_formats:
                        try:
                            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format=fmt)
                            timestamp_converted = True
                            break
                        except:
                            continue
                    
                    if not timestamp_converted:
                        # 尝试自动推断格式
                        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], infer_datetime_format=True)
                        
                    # 设置TimeStamp列为索引
                    df.set_index('TimeStamp', inplace=True)
                    logger.info(f"成功转换并设置TimeStamp列为索引")
                except Exception as e:
                    logger.warning(f"转换TimeStamp列失败: {e}，将跳过设置索引")
                    
            # 检查数据完整性
            if df.empty:
                logger.warning(f"加载的CSV文件为空: {file_path}")
                return None
            
            # 检查并移除可能的特殊字符列名
            df.columns = [str(col).strip().replace('"', '').replace('\'', '') for col in df.columns]
            
            return df
        except Exception as e:
            logger.error(f"加载原始数据失败: {e}")
            return None
    
    def transform_data(self, df):
        """将原始数据转换为识别数据
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            包含识别数据的DataFrame（仅包含平均值）
        """
        if df is None or df.empty:
            logger.error("输入数据为空")
            return None
        
        try:
            # 计算所有数值列的平均值
            numeric_columns = df.select_dtypes(include=['number']).columns
            
            if len(numeric_columns) == 0:
                logger.warning("数据中没有数值列可供处理")
                return None
            
            avg_data = df[numeric_columns].mean().to_frame().T
            
            return avg_data
        except Exception as e:
            logger.error(f"转换数据失败: {e}")
            return None
    
    def transform_file(self, file_path, output_dir=None):
        """转换单个文件的原始数据为识别数据
        
        Args:
            file_path: 原始数据文件路径
            output_dir: 输出目录，如果为None则使用配置文件中的路径或默认路径
            
        Returns:
            输出文件路径或None（如果失败）
        """
        # 加载原始数据
        df = self.load_raw_data(file_path)
        if df is None:
            return None
        
        # 转换数据
        transformed_df = self.transform_data(df)
        if transformed_df is None:
            return None
        
        # 确定输出目录
        if output_dir is None:
            # 优先从配置文件中获取
            if self.config and "transformed_data_dir" in self.config:
                output_dir = self.config["transformed_data_dir"]
            else:
                # 使用默认数据目录
                output_dir = self.data_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        base_name = os.path.basename(file_path)
        file_name, ext = os.path.splitext(base_name)
        output_file_name = f"{file_name}_transformed{ext}"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        # 保存转换后的数据
        try:
            transformed_df.to_csv(output_file_path, index=False)
            logger.info(f"转换后的数据已保存到: {output_file_path}")
            return output_file_path
        except Exception as e:
            logger.error(f"保存转换后的数据失败: {e}")
            return None
    
    def batch_transform_files(self, input_dir=None, output_dir=None):
        """批量转换目录中所有CSV文件的原始数据为识别数据
        
        Args:
            input_dir: 包含原始数据文件的目录，如果为None则使用配置文件中的路径
            output_dir: 输出目录，如果为None则使用配置文件中的路径
            
        Returns:
            成功转换的文件数量
        """
        # 确定输入目录
        if input_dir is None:
            # 优先从配置文件中获取
            if self.config and "raw_data_dir" in self.config:
                input_dir = self.config["raw_data_dir"]
            else:
                # 使用默认数据目录
                input_dir = self.data_dir
        
        if not os.path.exists(input_dir):
            logger.error(f"输入目录不存在: {input_dir}")
            return 0
        
        # 确定输出目录
        if output_dir is None:
            # 优先从配置文件中获取
            if self.config and "transformed_data_dir" in self.config:
                output_dir = self.config["transformed_data_dir"]
            else:
                # 使用默认数据目录
                output_dir = self.data_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 遍历目录中的所有CSV文件
        success_count = 0
        for file_name in os.listdir(input_dir):
            if not file_name.endswith('.csv'):
                continue
            
            # 跳过已经转换过的文件
            if '_transformed' in file_name:
                continue
            
            file_path = os.path.join(input_dir, file_name)
            
            # 转换文件
            result = self.transform_file(file_path, output_dir)
            if result is not None:
                success_count += 1
        
        logger.info(f"批量转换完成，成功转换了 {success_count} 个文件")
        return success_count
    
    def get_recognition_data(self, file_paths=None):
        """从多个文件中获取识别数据，只使用最新的文件并计算平均值
        
        Args:
            file_paths: 文件路径列表或包含原始数据的目录，如果为None则使用配置文件中的数据目录
        
        Returns:
            包含识别数据的DataFrame（仅包含平均值）
        """
        # 处理输入
        if file_paths is None:
            # 优先使用配置文件中的数据目录
            if self.config and "transformed_data_dir" in self.config:
                file_paths = self.config["transformed_data_dir"]
            else:
                file_paths = self.data_dir
        
        if isinstance(file_paths, str):
            # 如果是目录，获取目录中的所有CSV文件
            if os.path.isdir(file_paths):
                file_paths = [os.path.join(file_paths, f) for f in os.listdir(file_paths) 
                              if f.endswith('.csv')]
            else:
                # 如果是单个文件，转换为列表
                file_paths = [file_paths]
        
        # 过滤掉不存在的文件
        valid_files = [f for f in file_paths if os.path.exists(f)]
        
        if not valid_files:
            logger.error("没有有效的数据文件")
            return pd.DataFrame()  # 返回空的DataFrame
        
        # 按修改时间排序，选择最新的文件
        valid_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = valid_files[0]
        
        logger.info(f"使用最新的数据文件: {latest_file}")
        
        try:
            # 读取最新的文件（使用多种编码尝试）
            df = self._read_file_with_encoding(latest_file)
            
            if df is None or df.empty:
                logger.error(f"读取的文件为空或无效: {latest_file}")
                return pd.DataFrame()
            
            # 检查并移除可能的特殊字符列名
            df.columns = [str(col).strip().replace('"', '').replace('\'', '') for col in df.columns]
            
            # 如果文件包含TimeStamp列，视为原始数据
            if 'TimeStamp' in df.columns:
                logger.info(f"检测到原始数据文件，计算所有数据的平均值: {latest_file}")
                
                # 排除TimeStamp列，计算所有数值列的平均值
                numeric_columns = df.select_dtypes(include=['number']).columns
                if len(numeric_columns) == 0:
                    logger.warning("数据中没有数值列可供处理")
                    return pd.DataFrame()
                
                avg_data = df[numeric_columns].mean().to_frame().T
                
                # 添加时间戳信息
                avg_data['source_file'] = os.path.basename(latest_file)
                avg_data['process_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                return avg_data
            else:
                # 对于已转换的数据，也计算所有列的平均值
                logger.info(f"检测到已转换数据文件，计算所有数据的平均值: {latest_file}")
                
                # 计算所有数值列的平均值
                numeric_columns = df.select_dtypes(include=['number']).columns
                if len(numeric_columns) == 0:
                    logger.warning("数据中没有数值列可供处理")
                    return pd.DataFrame()
                
                avg_data = df[numeric_columns].mean().to_frame().T
                
                # 添加元信息
                avg_data['source_file'] = os.path.basename(latest_file)
                avg_data['process_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                return avg_data
        except Exception as e:
            logger.error(f"处理最新文件 {latest_file} 时出错: {e}")
            return pd.DataFrame()
    
    def generate_feature_names(self, df):
        """生成特征名称列表
        
        Args:
            df: 数据DataFrame
            
        Returns:
            特征名称列表
        """
        if df is None or df.empty:
            return []
        
        # 排除时间戳和元信息列
        excluded_columns = ['TimeStamp', 'timestamp', 'index', 'source_file', 'process_time']
        feature_columns = [col for col in df.columns if col not in excluded_columns]
        
        return feature_columns
    
    def preprocess_for_model(self, df):
        """预处理数据以用于模型训练
        
        Args:
            df: 识别数据DataFrame
            
        Returns:
            预处理后的数据（特征矩阵）和特征名称列表
        """
        if df is None or df.empty:
            logger.error("输入数据为空")
            return None, None
        
        try:
            # 生成特征名称
            feature_names = self.generate_feature_names(df)
            
            # 提取特征矩阵
            X = df[feature_names].values
            
            return X, feature_names
        except Exception as e:
            logger.error(f"预处理数据失败: {e}")
            return None, None

if __name__ == "__main__":
    # 示例用法
    transformer = DataTransformer()
    # 测试文件路径（根据实际情况修改）
    test_file = "transformed_data_20250922_180744.csv"  # 示例文件
    if os.path.exists(test_file):
        print(f"测试处理文件: {test_file}")
        result = transformer.get_recognition_data(test_file)
        if not result.empty:
            print("处理结果预览:")
            print(result.head())
        else:
            print("处理失败或结果为空")
    else:
        print(f"测试文件不存在: {test_file}")
        # 使用数据目录进行测试
        print(f"使用数据目录进行测试: {transformer.data_dir}")
        result = transformer.get_recognition_data()
        if not result.empty:
            print("处理结果预览:")
            print(result.head())
        else:
            print("处理失败或结果为空")