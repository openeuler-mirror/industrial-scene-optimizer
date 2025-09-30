import os
import joblib
import pandas as pd
import numpy as np
import pickle
import re
import os
import sys

# 确保可以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_transformer import DataTransformer

# 导入日志工具
from logger_utils import get_logger

# 使用统一的日志记录器
logger = get_logger()

class SceneRecognizer:
    """场景识别器 - 加载模型对输入的一组识别数据进行场景判断"""
    
    def __init__(self, model_path="scene_recognizer_model.pkl"):
        """初始化场景识别器
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.transformer = DataTransformer()
        
        # 场景映射字典
        self.scene_mapping = {
            0: "compute_intensive",  # 计算密集型
            1: "data_intensive",     # 数据密集型
            2: "hybrid_load",        # 混合负载型
            3: "light_load"          # 轻量负载型
        }
        
        # 反向场景映射（用于通过场景名称获取ID）
        self.reverse_scene_mapping = {v: k for k, v in self.scene_mapping.items()}
        
        # 尝试加载模型
        self.load_model()
    
    def load_model(self, model_path=None):
        """加载训练好的模型
        
        Args:
            model_path: 模型文件路径，如果为None则使用初始化时的路径
            
        Returns:
            是否成功加载模型
        """
        if model_path is None:
            model_path = self.model_path
        
        # 如果模型路径为None，则不尝试加载
        if model_path is None:
            logger.info("模型路径为None，不尝试加载模型")
            self.model = None
            self.scaler = None
            self.feature_names = None
            return False
        
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            self.model = None
            self.scaler = None
            self.feature_names = None
            return False
        
        try:
            # 加载模型
            logger.debug(f"开始加载模型: {model_path}")
            self.model = joblib.load(model_path)
            logger.info(f"模型已成功加载: {model_path}")
            
            # 检查模型结构
            if not hasattr(self.model, 'predict'):
                logger.error(f"加载的模型不具有predict方法，可能不是有效的机器学习模型")
                self.model = None
                return False
            
            # 尝试加载标准化器
            scaler_path = f"{os.path.splitext(model_path)[0]}_scaler.pkl"
            if os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    logger.info(f"标准化器已成功加载: {scaler_path}")
                except Exception as scaler_e:
                    logger.warning(f"加载标准化器失败: {str(scaler_e)}")
                    self.scaler = None
            else:
                logger.warning(f"标准化器文件不存在: {scaler_path}")
                self.scaler = None
            
            # 尝试加载特征名称
            feature_path = f"{os.path.splitext(model_path)[0]}_features.pkl"
            if os.path.exists(feature_path):
                try:
                    self.feature_names = joblib.load(feature_path)
                    logger.info(f"特征名称已成功加载: {feature_path}")
                except Exception as feature_e:
                    logger.warning(f"加载特征名称失败: {str(feature_e)}")
                    self.feature_names = None
            else:
                logger.warning(f"特征名称文件不存在: {feature_path}")
                self.feature_names = None
            
            return True
        except joblib.externals.loky.process_executor.TerminatedWorkerError:
            logger.error(f"加载模型时工作进程被终止，可能是模型文件损坏或内存不足")
            self.model = None
            self.scaler = None
            return False
        except EOFError:
            logger.error(f"模型文件格式错误或文件已损坏: {model_path}")
            self.model = None
            self.scaler = None
            return False
        except pickle.UnpicklingError:
            logger.error(f"模型文件不是有效的pickle格式: {model_path}")
            self.model = None
            self.scaler = None
            return False
        except ValueError as ve:
            logger.error(f"模型数据格式错误: {str(ve)}")
            self.model = None
            self.scaler = None
            return False
        except ImportError as ie:
            logger.error(f"加载模型所需的依赖包缺失: {str(ie)}")
            self.model = None
            self.scaler = None
            return False
        except Exception as e:
            logger.error(f"加载模型失败: {type(e).__name__} - {str(e)}")
            # 记录异常的堆栈信息，以便更详细地诊断问题
            import traceback
            logger.debug(f"异常堆栈:\n{traceback.format_exc()}")
            self.model = None
            self.scaler = None
            return False
    
    def is_model_loaded(self):
        """检查模型是否已加载
        
        Returns:
            模型是否已加载
        """
        return self.model is not None
    
    def generate_sample_model(self):
        """生成一个简单的示例模型，当没有预训练模型时使用
        
        Returns:
            是否成功生成示例模型
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np
            import joblib
            
            # 生成简单的模拟数据和标签
            np.random.seed(42)
            X = np.random.rand(100, 10)  # 100个样本，10个特征
            y = np.random.randint(0, 4, 100)  # 4个场景类型的随机标签
            
            # 训练一个简单的随机森林模型
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.model.fit(X, y)
            
            # 创建一个简单的标准化器（实际上不做任何标准化）
            class SimpleScaler:
                def transform(self, X):
                    return X
                def fit(self, X, y=None):
                    return self
                    
            self.scaler = SimpleScaler()
            
            print("已生成示例模型")
            return True
        except Exception as e:
            print(f"生成示例模型失败: {e}")
            self.model = None
            self.scaler = None
            return False
    
    def _read_file_with_encoding(self, file_path):
        """使用多种编码尝试读取文件内容
        
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
                logger.info(f"成功使用{encoding}编码读取文件内容: {file_path}")
                break
            except UnicodeDecodeError:
                logger.warning(f"使用{encoding}编码读取文件失败: {file_path}")
                continue
        
        if content is None:
            # 最后尝试使用utf-8并替换错误字符
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                logger.info(f"成功使用utf-8(替换错误字符)编码读取文件: {file_path}")
            except Exception as e:
                logger.error(f"所有编码尝试都失败: {str(e)}, 文件: {file_path}")
                raise
        
        return content
        
    def _clean_feature_name(self, name):
        """清理特征名称，去除#及其后面的内容"""
        if '#' in name:
            return name.split('#')[0].strip()
        return name.strip()
        
    def _merge_duplicate_columns(self, df):
        """合并相同名称的列，并将数据相加
        
        Args:
            df: 输入的DataFrame
            
        Returns:
            合并后的DataFrame
        """
        # 检查是否有重复的列名
        if df.columns.duplicated().any():
            logger.info("检测到重复的列名，开始合并数据")
            
            # 创建一个新的DataFrame来存储合并后的结果
            merged_df = pd.DataFrame()
            
            # 遍历所有唯一的列名
            for col in df.columns.unique():
                # 检查当前列名是否有重复
                if df.columns.tolist().count(col) > 1:
                    logger.info(f"合并列 '{col}' 的所有实例")
                    # 获取所有同名的列
                    duplicate_columns = df.loc[:, df.columns == col]
                    # 将这些列的值相加
                    merged_df[col] = duplicate_columns.sum(axis=1)
                else:
                    # 如果没有重复，直接复制列
                    merged_df[col] = df[col]
            
            logger.info(f"列合并完成，原始列数: {len(df.columns)}, 合并后列数: {len(merged_df.columns)}")
            return merged_df
        else:
            # 如果没有重复的列名，直接返回原始DataFrame
            return df.copy()
    
    def preprocess_data(self, data):
        """预处理输入数据
        
        Args:
            data: 输入的识别数据（可以是DataFrame、数组或文件路径）
        
        Returns:
            预处理后的数据（特征矩阵）
        """
        try:
            # 处理不同类型的输入
            if isinstance(data, str):
                # 如果是文件路径，加载文件
                if os.path.exists(data):
                    # 根据文件扩展名选择合适的读取方法
                    if data.endswith('.txt'):
                        try:
                            # 先尝试使用自定义的编码读取方法
                            content = self._read_file_with_encoding(data)
                            
                            # 处理可能的引号问题
                            if content.startswith('"') and content.endswith('"'):
                                content = content[1:-1]
                            
                            # 保存为临时CSV文件以便使用pandas读取
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
                                temp.write(content.encode('utf-8'))
                                temp_file_path = temp.name
                            
                            df = pd.read_csv(temp_file_path, sep=',', engine='python')
                            os.unlink(temp_file_path)  # 删除临时文件
                        except Exception as e:
                            logger.warning(f"使用自定义方法读取txt文件失败: {str(e)}，尝试备用方法")
                            # 备用方法：直接使用pandas读取，尝试不同的分隔符
                            try:
                                df = pd.read_csv(data, sep=',', engine='python')
                            except:
                                df = pd.read_csv(data, sep=' ', engine='python')
                    else:
                        # 对于CSV文件，尝试多种参数配置
                        try:
                            df = pd.read_csv(data)
                        except:
                            # 如果失败，尝试更灵活的参数
                            df = pd.read_csv(data, sep=',', engine='python', header='infer')
                else:
                    logger.error(f"文件不存在: {data}")
                    print(f"错误：文件不存在: {data}")
                    return None
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            elif isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
            else:
                print(f"错误：不支持的数据类型: {type(data)}")
                return None
            
            # 检查数据是否为空
            if df.empty:
                logger.warning("输入数据为空")
                print("错误：输入数据为空")
                return None
            
            # 清理列名，去除#及其后面的内容
            logger.info("开始清理特征名称")
            df.columns = [self._clean_feature_name(col) for col in df.columns]
            
            # 合并相同名称的列
            df = self._merge_duplicate_columns(df)
            
            # 获取目标特征数量（优先使用模型期望的特征数量）
            target_feature_count = None
            if hasattr(self.model, 'n_features_in_'):
                target_feature_count = self.model.n_features_in_
                logger.info(f"模型期望特征数量: {target_feature_count}")
            elif self.feature_names is not None:
                target_feature_count = len(self.feature_names)
                logger.info(f"从保存的特征名称获取期望特征数量: {target_feature_count}")
            
            # 提取特征矩阵
            # 如果有保存的特征名称，使用这些名称来选择列
            if self.feature_names is not None:
                # 找出数据中存在的特征
                available_features = [col for col in self.feature_names if col in df.columns]
                missing_features = [col for col in self.feature_names if col not in df.columns]
                
                if missing_features:
                    logger.warning(f"数据中缺少以下特征: {missing_features}")
                    print(f"警告：数据中缺少以下特征: {missing_features}")
                
                if available_features:
                    logger.info(f"使用保存的特征名称提取特征，共{len(available_features)}个特征可用")
                    X = df[available_features].values
                    
                    # 如果有目标特征数量，优先按照目标数量处理
                    if target_feature_count is not None:
                        if X.shape[1] != target_feature_count:
                            logger.warning(f"特征数量不匹配 - 期望{target_feature_count}个特征，但当前有{X.shape[1]}个特征")
                            print(f"警告：特征数量不匹配 - 期望{target_feature_count}个特征，但当前有{X.shape[1]}个特征")
                            
                            # 处理特征数量不匹配的情况
                            if X.shape[1] > target_feature_count:
                                logger.info(f"截断特征：保留前{target_feature_count}个特征")
                                print(f"截断特征：保留前{target_feature_count}个特征")
                                X = X[:, :target_feature_count]
                            else:
                                logger.info(f"填充特征：在数据末尾添加{target_feature_count - X.shape[1]}个0值特征")
                                print(f"填充特征：在数据末尾添加{target_feature_count - X.shape[1]}个0值特征")
                                padding = np.zeros((X.shape[0], target_feature_count - X.shape[1]))
                                X = np.hstack((X, padding))
                    elif len(available_features) < len(self.feature_names):
                        # 如果没有目标特征数量，但有保存的特征名称列表，按保存的列表处理
                        missing_count = len(self.feature_names) - len(available_features)
                        logger.info(f"填充{missing_count}个缺失特征的0值")
                        padding = np.zeros((X.shape[0], missing_count))
                        X = np.hstack((X, padding))
                else:
                    logger.warning("没有可用的特征名称匹配，使用所有数值列")
                    # 如果没有匹配的特征名称，使用所有数值列
                    numeric_columns = df.select_dtypes(include=['number']).columns
                    if len(numeric_columns) > 0:
                        X = df[numeric_columns].values
                        
                        # 如果有目标特征数量，按照目标数量处理
                        if target_feature_count is not None and X.shape[1] != target_feature_count:
                            logger.warning(f"特征数量不匹配 - 期望{target_feature_count}个特征，但当前有{X.shape[1]}个特征")
                            print(f"警告：特征数量不匹配 - 期望{target_feature_count}个特征，但当前有{X.shape[1]}个特征")
                            
                            if X.shape[1] > target_feature_count:
                                logger.info(f"截断特征：保留前{target_feature_count}个特征")
                                print(f"截断特征：保留前{target_feature_count}个特征")
                                X = X[:, :target_feature_count]
                            else:
                                logger.info(f"填充特征：在数据末尾添加{target_feature_count - X.shape[1]}个0值特征")
                                print(f"填充特征：在数据末尾添加{target_feature_count - X.shape[1]}个0值特征")
                                padding = np.zeros((X.shape[0], target_feature_count - X.shape[1]))
                                X = np.hstack((X, padding))
                    else:
                        logger.error("数据中没有有效的数值列")
                        return None
            else:
                # 如果没有保存的特征名称，使用transformer的方法
                X, _ = self.transformer.preprocess_for_model(df)
                
                # 如果有目标特征数量，按照目标数量处理
                if target_feature_count is not None and X.shape[1] != target_feature_count:
                    logger.warning(f"特征数量不匹配 - 期望{target_feature_count}个特征，但当前有{X.shape[1]}个特征")
                    print(f"警告：特征数量不匹配 - 期望{target_feature_count}个特征，但当前有{X.shape[1]}个特征")
                    
                    if X.shape[1] > target_feature_count:
                        logger.info(f"截断特征：保留前{target_feature_count}个特征")
                        print(f"截断特征：保留前{target_feature_count}个特征")
                        X = X[:, :target_feature_count]
                    else:
                        logger.info(f"填充特征：在数据末尾添加{target_feature_count - X.shape[1]}个0值特征")
                        print(f"填充特征：在数据末尾添加{target_feature_count - X.shape[1]}个0值特征")
                        padding = np.zeros((X.shape[0], target_feature_count - X.shape[1]))
                        X = np.hstack((X, padding))
            
            # 如果有标准化器，对数据进行标准化
            if self.scaler is not None:
                try:
                    # 标准化数据
                    # 确保标准化器和模型使用相同数量的特征
                    if hasattr(self.scaler, 'n_features_in_') and hasattr(self.model, 'n_features_in_'):
                        # 首先确保数据与模型期望的特征数量一致
                        if X.shape[1] != self.model.n_features_in_:
                            logger.warning(f"数据特征数量与模型期望不匹配 - 期望{self.model.n_features_in_}个特征，但当前有{X.shape[1]}个特征")
                            print(f"警告：数据特征数量与模型期望不匹配 - 期望{self.model.n_features_in_}个特征，但当前有{X.shape[1]}个特征")
                            
                            # 调整数据到模型期望的特征数量
                            if X.shape[1] > self.model.n_features_in_:
                                logger.info(f"截断特征：保留前{self.model.n_features_in_}个特征")
                                print(f"截断特征：保留前{self.model.n_features_in_}个特征")
                                X = X[:, :self.model.n_features_in_]
                            else:
                                logger.info(f"填充特征：在数据末尾添加{self.model.n_features_in_ - X.shape[1]}个0值特征")
                                print(f"填充特征：在数据末尾添加{self.model.n_features_in_ - X.shape[1]}个0值特征")
                                padding = np.zeros((X.shape[0], self.model.n_features_in_ - X.shape[1]))
                                X = np.hstack((X, padding))
                        
                        # 现在确保标准化器与模型期望的特征数量一致
                        if self.scaler.n_features_in_ != self.model.n_features_in_:
                            logger.warning(f"标准化器期望{self.scaler.n_features_in_}个特征，但模型期望{self.model.n_features_in_}个特征")
                            print(f"警告：标准化器期望{self.scaler.n_features_in_}个特征，但模型期望{self.model.n_features_in_}个特征")
                            
                            # 创建一个新的标准化器，使用模型期望的特征数量
                            from sklearn.preprocessing import StandardScaler
                            new_scaler = StandardScaler()
                            # 使用当前数据训练新的标准化器
                            new_scaler.fit(X)
                            logger.info(f"创建了新的标准化器，适应模型期望的{self.model.n_features_in_}个特征")
                            print(f"创建了新的标准化器，适应模型期望的{self.model.n_features_in_}个特征")
                            
                            # 使用新的标准化器
                            X = new_scaler.transform(X)
                        else:
                            # 标准化器和模型期望的特征数量一致，可以直接使用
                            X = self.scaler.transform(X)
                    else:
                        # 如果无法确定标准化器或模型的特征数量要求，直接尝试使用标准化器
                        X = self.scaler.transform(X)
                except Exception as e:
                    # 如果出现特征数量不匹配的错误，记录详细信息
                    if hasattr(self.scaler, 'n_features_in_') and hasattr(self.model, 'n_features_in_'):
                        logger.error(f"标准化数据时出错: {e}. 模型期望{self.model.n_features_in_}个特征，标准化器期望{self.scaler.n_features_in_}个特征")
                        print(f"标准化数据时出错: {e}. 模型期望{self.model.n_features_in_}个特征，标准化器期望{self.scaler.n_features_in_}个特征")
                    else:
                        logger.error(f"标准化数据时出错: {e}")
                        print(f"标准化数据时出错: {e}")
            
            return X
        except Exception as e:
            logger.error(f"预处理数据失败: {e}")
            print(f"预处理数据失败: {e}")
            return None
    
    def recognize_scene(self, data):
        """识别输入数据的场景
        
        Args:
            data: 输入的识别数据
        
        Returns:
            场景识别结果（字符串）或None（如果失败）
        """
        # 检查模型是否已加载
        if not self.is_model_loaded():
            print("错误：模型未加载")
            return None
        
        # 预处理数据
        X = self.preprocess_data(data)
        if X is None:
            return None
        
        try:
            # 重塑数据（如果是一维数组）
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # 最终检查特征数量是否匹配模型期望（作为最后的保障）
            if hasattr(self.model, 'n_features_in_'):
                expected_features = self.model.n_features_in_
                actual_features = X.shape[1]
                
                if actual_features != expected_features:
                    logger.warning(f"特征数量不匹配 - 模型期望{expected_features}个特征，但数据有{actual_features}个特征")
                    print(f"警告：特征数量不匹配 - 模型期望{expected_features}个特征，但数据有{actual_features}个特征")
                    
                    # 处理特征数量不匹配的情况
                    if actual_features > expected_features:
                        logger.info(f"截断特征：保留前{expected_features}个特征")
                        print(f"截断特征：保留前{expected_features}个特征")
                        X = X[:, :expected_features]
                    else:
                        logger.info(f"填充特征：在数据末尾添加{expected_features - actual_features}个0值特征")
                        print(f"填充特征：在数据末尾添加{expected_features - actual_features}个0值特征")
                        padding = np.zeros((X.shape[0], expected_features - actual_features))
                        X = np.hstack((X, padding))
            
            # 使用模型进行预测
            predictions = self.model.predict(X)
            
            # 如果只有一个样本，直接返回场景名称
            if len(predictions) == 1:
                scene_id = predictions[0]
                logger.info(f"场景识别结果: {self.scene_mapping.get(scene_id, 'unknown')}")
                return self.scene_mapping.get(scene_id, "unknown")
            
            # 如果有多个样本，返回场景名称列表
            results = [self.scene_mapping.get(scene_id, "unknown") for scene_id in predictions]
            logger.info(f"批量场景识别完成，共{len(results)}个结果")
            return results
        except Exception as e:
            logger.error(f"场景识别失败: {e}")
            print(f"场景识别失败: {e}")
            return None
    
    def get_scene_probabilities(self, data):
        """获取场景识别的概率分布
        
        Args:
            data: 输入的识别数据
            
        Returns:
            概率分布（DataFrame）或None（如果失败）
        """
        # 检查模型是否已加载
        if not self.is_model_loaded():
            print("错误：模型未加载")
            return None
        
        # 检查模型是否支持概率预测
        if not hasattr(self.model, "predict_proba"):
            print("错误：当前模型不支持概率预测")
            return None
        
        # 预处理数据
        X = self.preprocess_data(data)
        if X is None:
            return None
        
        try:
            # 获取概率分布
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            probabilities = self.model.predict_proba(X)
            
            # 创建包含概率的DataFrame
            scene_names = list(self.scene_mapping.values())
            prob_df = pd.DataFrame(probabilities, columns=scene_names)
            
            return prob_df
        except Exception as e:
            print(f"获取场景概率失败: {e}")
            return None
    
    def batch_recognize(self, file_paths):
        """批量识别多个文件中的场景
        
        Args:
            file_paths: 文件路径列表或包含识别数据的目录
            
        Returns:
            场景识别结果字典（文件路径 -> 场景名称）
        """
        results = {}
        
        # 处理输入
        if isinstance(file_paths, str):
            # 如果是目录，获取目录中的所有转换后文件
            if os.path.isdir(file_paths):
                # 获取目录下所有CSV和TXT文件，不限制文件名中必须包含_transformed
                file_paths = [os.path.join(file_paths, f) for f in os.listdir(file_paths) 
                              if f.endswith('.csv') or f.endswith('.txt')]
                logger.info(f"从目录{file_paths}中找到{len(file_paths)}个数据文件")
            else:
                # 如果是单个文件，转换为列表
                file_paths = [file_paths]
        
        # 对每个文件进行场景识别
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                print(f"警告：文件不存在: {file_path}")
                results[file_path] = "file_not_found"
                continue
            
            logger.info(f"开始识别文件场景: {file_path}")
            # 识别场景
            scene = self.recognize_scene(file_path)
            results[file_path] = scene if scene is not None else "recognition_failed"
            
            # 记录识别结果
            if scene is not None:
                logger.info(f"文件{file_path}的场景识别结果: {scene}")
            else:
                logger.warning(f"文件{file_path}的场景识别失败")
        
        return results
    
    def recognize_real_time(self, collector):
        """实时识别场景
        
        Args:
            collector: 数据采集器实例
            
        Returns:
            场景识别结果或None（如果失败）
        """
        try:
            # 检查采集器是否有collect_raw_data方法
            if not hasattr(collector, "collect_raw_data"):
                print("错误：采集器对象不支持collect_raw_data方法")
                return None
            
            # 采集实时数据
            raw_data = collector.collect_raw_data()
            
            # 转换为DataFrame
            # 假设我们知道字段名称（从collector获取）
            if hasattr(collector, "field_names"):
                field_names = collector.field_names
            else:
                # 如果没有字段名称，使用通用名称
                field_names = [f"feature_{i}" for i in range(len(raw_data))]
            
            # 创建单行DataFrame
            df = pd.DataFrame([raw_data], columns=field_names)
            
            # 识别场景
            return self.recognize_scene(df)
        except Exception as e:
            print(f"实时场景识别失败: {e}")
            return None
    
    def save_recognition_results(self, results, output_file="recognition_results.csv"):
        """保存场景识别结果到文件
        
        Args:
            results: 场景识别结果字典
            output_file: 输出文件路径
            
        Returns:
            是否成功保存
        """
        try:
            # 创建结果DataFrame
            df = pd.DataFrame(list(results.items()), columns=['File_Path', 'Scene'])
            
            # 保存到CSV文件
            df.to_csv(output_file, index=False)
            print(f"场景识别结果已保存到: {output_file}")
            return True
        except Exception as e:
            print(f"保存场景识别结果失败: {e}")
            return False

if __name__ == "__main__":
    # 示例用法
    recognizer = SceneRecognizer()
    
    # 识别单个文件的场景
    # scene = recognizer.recognize_scene("data/sample_transformed.csv")
    # print(f"识别的场景: {scene}")
    
    # 批量识别多个文件的场景
    # results = recognizer.batch_recognize("data")
    # for file_path, scene in results.items():
    #     print(f"文件: {file_path}, 场景: {scene}")
    
    # 获取场景概率
    # prob_df = recognizer.get_scene_probabilities("data/sample_transformed.csv")
    # if prob_df is not None:
    #     print("场景概率分布:")
    #     print(prob_df)