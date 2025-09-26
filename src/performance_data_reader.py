import pandas as pd
import numpy as np
import os
import re
import sys

# 首先添加当前目录到Python路径以确保可以找到同级模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入logger_utils，优先使用绝对导入避免相对导入问题
try:
    from logger_utils import get_logger
except ImportError:
    # 如果绝对导入失败，尝试相对导入
    try:
        from .logger_utils import get_logger
    except ImportError:
        raise ImportError("无法导入logger_utils模块，请检查Python路径设置。")

# 使用统一的日志记录器
logger = get_logger('performance_data_reader')

class PerformanceDataReader:
    """专门用于读取和解析Performance_Data格式文件的工具类"""
    
    def __init__(self, scene_mapping=None, inverse_scene_mapping=None):
        """
        初始化PerformanceDataReader
        
        Args:
            scene_mapping: 场景标签映射字典
            inverse_scene_mapping: 逆场景标签映射字典
        """
        # 默认场景映射
        self.default_scene_mapping = {
            0: "compute_intensive",
            1: "data_intensive",
            2: "hybrid_load",
            3: "light_load"
        }
        
        # 默认逆场景映射
        self.default_inverse_scene_mapping = {
            "compute_intensive": 0,
            "data_intensive": 1,
            "hybrid_load": 2,
            "light_load": 3
        }
        
        # 使用提供的映射或默认映射
        self.scene_mapping = scene_mapping if scene_mapping is not None else self.default_scene_mapping
        self.inverse_scene_mapping = inverse_scene_mapping if inverse_scene_mapping is not None else self.default_inverse_scene_mapping
    
    def read_file(self, file_path):
        """
        使用多种编码尝试读取文件内容
        
        Args:
            file_path: 文件路径
        
        Returns:
            str: 文件内容
        """
        # 支持的编码列表
        encodings = ['utf-8', 'gb2312', 'gbk', 'ansi', 'latin1']
        content = None
        
        # 尝试不同编码读取文件内容
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.info(f"成功使用{encoding}编码读取文件内容")
                break
            except UnicodeDecodeError:
                logger.warning(f"使用{encoding}编码读取文件失败")
                continue
        
        if content is None:
            # 最后尝试使用utf-8并替换错误字符
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                logger.info("成功使用utf-8(替换错误字符)编码读取文件")
            except Exception as e:
                logger.error(f"所有编码尝试都失败: {str(e)}")
                raise
        
        return content
    
    def parse_file(self, file_path):
        """
        解析Performance_Data格式文件
        
        Args:
            file_path: 文件路径
        
        Returns:
            DataFrame: 包含特征和标签的数据框
        """
        logger.info(f"开始解析Performance_Data文件: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return None
        
        # 读取文件内容
        content = self.read_file(file_path)
        
        # 分割数据块
        data_blocks = []
        current_block = {}
        lines = content.split('\n')
        
        # 用于匹配Performance_Data行的正则表达式
        performance_data_pattern = re.compile(r'^Performance_Data:"')
        # 用于匹配Scene_Result行的正则表达式
        scene_result_pattern = re.compile(r'^Scene_Result:"')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if performance_data_pattern.match(line):
                # 处理Performance_Data行
                # 提取特征名称
                try:
                    # 移除Performance_Data:"前缀和可能的结尾引号
                    feature_names_part = line[len("Performance_Data:"):].strip().rstrip('"')
                    feature_names = [name.strip() for name in feature_names_part.split(',')]
                    current_block['feature_names'] = feature_names
                    
                    # 下一行应该是数值数据
                    i += 1
                    if i < len(lines):
                        data_line = lines[i].strip()
                        # 移除可能的结尾引号和逗号
                        data_line = data_line.rstrip('"').rstrip(',')
                        # 分割数值
                        values = [val.strip() for val in data_line.split(',')]
                        current_block['values'] = values
                    
                except Exception as e:
                    logger.warning(f"解析Performance_Data行失败: {str(e)}")
                    current_block = {}
                    i += 1
                    continue
                
            elif scene_result_pattern.match(line) and 'feature_names' in current_block:
                # 处理Scene_Result行
                try:
                    # 提取场景结果
                    scene_result = line[len("Scene_Result:"):].strip().strip('"')
                    current_block['scene_result'] = scene_result
                    
                    # 添加完整的数据块
                    data_blocks.append(current_block)
                    current_block = {}
                except Exception as e:
                    logger.warning(f"解析Scene_Result行失败: {str(e)}")
                    current_block = {}
            
            i += 1
        
        logger.info(f"共识别到{len(data_blocks)}个完整数据块")
        
        # 处理数据块，构建DataFrame
        if not data_blocks:
            logger.error("未成功解析任何完整数据块")
            return None
        
        # 准备数据
        features_list = []
        labels_list = []
        
        for block_idx, block in enumerate(data_blocks):
            try:
                # 解析数值数据
                feature_names = block['feature_names']
                values = block['values']
                scene_result = block['scene_result']
                
                # 确保数值数量与特征数量匹配
                if len(values) >= len(feature_names):
                    values = values[:len(feature_names)]
                else:
                    # 如果数值数量不足，用NaN填充
                    values += [''] * (len(feature_names) - len(values))
                
                # 处理'#'后面的进程信息并合并相同的性能参数
                processed_features = {}
                for idx, feature_name in enumerate(feature_names):
                    # 去掉'#'后面的进程信息
                    base_feature = feature_name.split('#')[0] if '#' in feature_name else feature_name
                    
                    # 获取对应的数值
                    val = values[idx]
                    try:
                        numeric_val = float(val) if val != '' else np.nan
                    except ValueError:
                        numeric_val = np.nan
                    
                    # 合并相同的性能参数，数值相加
                    if base_feature in processed_features:
                        if not np.isnan(numeric_val):
                            processed_features[base_feature] += numeric_val
                    else:
                        processed_features[base_feature] = numeric_val
                
                # 转换为列表格式
                processed_feature_names = list(processed_features.keys())
                processed_values = list(processed_features.values())
                
                # 转换场景结果为标签
                if scene_result in self.inverse_scene_mapping:
                    label = self.inverse_scene_mapping[scene_result]
                else:
                    # 如果是新的场景类型，添加到映射中
                    label = len(self.scene_mapping)
                    self.scene_mapping[label] = scene_result
                    self.inverse_scene_mapping[scene_result] = label
                    logger.info(f"发现新的场景类型: {scene_result}，映射为标签{label}")
                
                features_list.append(processed_values)
                labels_list.append(label)
                
                # 只在第一个数据块中设置特征名称
                if block_idx == 0:
                    final_feature_names = processed_feature_names
                
            except Exception as e:
                logger.warning(f"处理数据块{block_idx}失败: {str(e)}")
                continue
        
        if not features_list:
            logger.error("未成功解析任何数据")
            return None
        
        # 创建DataFrame
        df = pd.DataFrame(features_list, columns=final_feature_names)
        
        # 添加标签列
        df['scene_label'] = labels_list
        
        logger.info(f"成功解析Performance_Data文件，共得到{len(df)}条有效数据")
        logger.info(f"处理后保留的特征数量: {len(final_feature_names)}")
        return df
    
    def batch_process_files(self, file_paths):
        """
        批量处理多个Performance_Data格式文件
        
        Args:
            file_paths: 文件路径列表
        
        Returns:
            DataFrame: 合并后的DataFrame
        """
        all_data = []
        
        for file_path in file_paths:
            df = self.parse_file(file_path)
            if df is not None and len(df) > 0:
                all_data.append(df)
        
        if not all_data:
            logger.error("批量处理失败，未成功解析任何文件")
            return None
        
        # 合并所有DataFrame
        merged_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"成功合并{len(file_paths)}个文件的数据，共{len(merged_df)}条记录")
        return merged_df

# 提供便捷的函数接口
def read_performance_data(file_path, scene_mapping=None, inverse_scene_mapping=None):
    """
    读取Performance_Data格式文件的便捷函数
    
    Args:
        file_path: 文件路径
        scene_mapping: 场景标签映射字典
        inverse_scene_mapping: 逆场景标签映射字典
    
    Returns:
        DataFrame: 包含特征和标签的数据框
    """
    reader = PerformanceDataReader(scene_mapping, inverse_scene_mapping)
    return reader.parse_file(file_path)

def batch_read_performance_data(file_paths, scene_mapping=None, inverse_scene_mapping=None):
    """
    批量读取Performance_Data格式文件的便捷函数
    
    Args:
        file_paths: 文件路径列表
        scene_mapping: 场景标签映射字典
        inverse_scene_mapping: 逆场景标签映射字典
    
    Returns:
        DataFrame: 合并后的DataFrame
    """
    reader = PerformanceDataReader(scene_mapping, inverse_scene_mapping)
    return reader.batch_process_files(file_paths)