import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 配置matplotlib中文字体 - 更健壮的解决方案
import matplotlib
import matplotlib.font_manager as fm
import os
import sys
import argparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from datetime import datetime

# 首先添加当前目录到Python路径以确保可以找到同级模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入必要的模块，优先使用绝对导入避免相对导入问题
try:
    from performance_data_reader import read_performance_data, PerformanceDataReader
    from logger_utils import get_logger
except ImportError:
    # 如果绝对导入失败，尝试相对导入
    try:
        from .performance_data_reader import read_performance_data, PerformanceDataReader
        from .logger_utils import get_logger
    except ImportError:
        raise ImportError("无法导入必要的模块，请检查Python路径设置。")

# 默认文件路径设置
DEFAULT_PERFORMANCE_DATA_PATH = '/usr/share/industrial-scene-optimizer/Performance_Data.csv'
DEFAULT_MODEL_DIR = '/usr/share/industrial-scene-optimizer/models'

# 使用统一的日志记录器
logger = get_logger('ModelTrainer')

# 配置matplotlib字体设置，抑制字体警告
# 使用更通用的设置，适应不同操作系统
try:
    # 禁用matplotlib的字体查找警告
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    
    # 设置基础字体配置，使用系统默认字体
    plt.rcParams['font.family'] = ['sans-serif']
    
    # 使用matplotlib的默认字体列表，避免自定义搜索导致的问题
    # 仅添加安全的系统字体
    safe_system_fonts = ['Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['font.sans-serif'] = safe_system_fonts + plt.rcParams['font.sans-serif']
    
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 检查是否有可用的中文字体（仅记录信息，不强制使用）
    has_chinese_font = False
    try:
        # 定义常见中文字体的关键词
        chinese_font_keywords = ['sim', 'hei', 'song', 'kai', 'yahei', 'microsoft', 'heiti', 'wenquan']
        available_fonts = {f.name.lower() for f in fm.fontManager.ttflist}
        
        for font_name in available_fonts:
            for keyword in chinese_font_keywords:
                if keyword in font_name:
                    logger.info(f"系统中检测到中文字体: {font_name}")
                    has_chinese_font = True
                    break
            if has_chinese_font:
                break
    except Exception as e:
        logger.debug(f"中文字体检测过程中出错: {str(e)}")
        
    # 只记录信息，不再设置特定字体，避免冲突
    if has_chinese_font:
        logger.info("系统中有可用的中文字体，图表可能支持中文显示")
except Exception as e:
    logger.warning(f"配置matplotlib字体时出错: {str(e)}")
    # 为visualize_results方法提供英文标签替代方案
def safe_text(text):
    """安全处理文本，避免中文字体显示问题"""
    # 扩展的中英文映射字典，包含更多中文标签
    chinese_to_english = {
        '预测标签': 'Predicted Label',
        '真实标签': 'True Label',
        '混淆矩阵': 'Confusion Matrix',
        '交叉验证分数': 'Cross-Validation Scores',
        '场景识别': 'Scene Recognition',
        '准确率': 'Accuracy',
        '折数': 'Fold',
        '平均分数': 'Average Score',
        '拓': 'Expand',
        '数': 'Number',
        '据': 'Data',
        '精': 'Precise',
        '确': 'Accurate',
        '率': 'Rate',
        '平': 'Flat',
        '均': 'Average',
        '分': 'Score'
    }
    
    # 处理包含多个中文字符的情况
    result = text
    for chinese, english in chinese_to_english.items():
        if chinese in result:
            result = result.replace(chinese, english)
    
    return result

# 将safe_text函数添加到plt模块，方便全局使用
plt.safe_text = safe_text

class ModelTrainer:
    """模型训练器 - 使用识别数据和场景结果训练模型"""
    
    def __init__(self, model_path=None, model_type="random_forest"):
        """初始化模型训练器
        
        Args:
            model_path: 模型保存路径，如果为None则使用默认路径
            model_type: 模型类型，支持'random_forest'和'gradient_boosting'
        """
        # 如果未指定模型路径，使用默认路径
        if model_path is None:
            # 确保默认模型目录存在
            os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
            model_path = os.path.join(DEFAULT_MODEL_DIR, "scene_recognizer_model.pkl")
        
        self.model_path = model_path
        self.model_type = model_type
        self.model = self._initialize_model()
        self.scaler_type = 'standard'  # 支持'standard', 'minmax', 'robust'
        self.scaler = self._initialize_scaler()
        self.scene_mapping = {
            0: "compute_intensive",  # 计算密集型
            1: "data_intensive",     # 数据密集型
            2: "hybrid_load",        # 混合负载型
            3: "light_load"          # 轻量负载型
        }
        self.inverse_scene_mapping = {v: k for k, v in self.scene_mapping.items()}
        self.feature_names = None  # 将存储特征名称
    
    def _initialize_model(self):
        """初始化模型，支持随机森林和梯度提升"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            logger.warning(f"不支持的模型类型: {self.model_type}，使用随机森林作为默认模型")
            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    def _initialize_scaler(self):
        """初始化数据标准化器"""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            logger.warning(f"不支持的标准化器类型: {self.scaler_type}，使用StandardScaler作为默认")
            return StandardScaler()
    
    def set_scaler_type(self, scaler_type):
        """设置数据标准化器类型
        
        Args:
            scaler_type: 标准化器类型，支持'standard', 'minmax', 'robust'
        """
        self.scaler_type = scaler_type
        self.scaler = self._initialize_scaler()
    
    def load_model(self, model_path=None):
        """加载已训练的模型
        
        Args:
            model_path: 模型文件路径，如果为None则使用初始化时的路径
            
        Returns:
            是否成功加载模型
        """
        if model_path is None:
            model_path = self.model_path
        
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            return False
        
        try:
            self.model = joblib.load(model_path)
            logger.info(f"模型已成功加载: {model_path}")
            
            # 尝试加载标准化器
            scaler_path = f"{os.path.splitext(model_path)[0]}_scaler.pkl"
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"标准化器已成功加载: {scaler_path}")
            
            # 尝试加载特征名称
            feature_path = f"{os.path.splitext(model_path)[0]}_features.pkl"
            if os.path.exists(feature_path):
                self.feature_names = joblib.load(feature_path)
                logger.info(f"特征名称已成功加载: {feature_path}")
            
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return False
    
    def save_model(self, model_path=None):
        """保存训练好的模型
        
        Args:
            model_path: 模型保存路径，如果为None则使用初始化时的路径
            
        Returns:
            是否成功保存模型
        """
        if model_path is None:
            model_path = self.model_path
        
        # 确保目录存在
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        try:
            joblib.dump(self.model, model_path)
            logger.info(f"模型已成功保存到: {model_path}")
            
            # 保存标准化器
            scaler_path = f"{os.path.splitext(model_path)[0]}_scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"标准化器已成功保存到: {scaler_path}")
            
            # 保存特征名称
            if self.feature_names is not None:
                feature_path = f"{os.path.splitext(model_path)[0]}_features.pkl"
                joblib.dump(self.feature_names, feature_path)
                logger.info(f"特征名称已成功保存到: {feature_path}")
            
            return True
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
            return False
    
    def _preprocess_data(self, X):
        """数据预处理：处理缺失值、异常值等
        
        Args:
            X: 输入特征矩阵
            
        Returns:
            处理后的特征矩阵
        """
        # 转换为DataFrame便于处理
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = X.copy()
            self.feature_names = df.columns.tolist()
        
        # 检查并移除全是缺失值的列
        non_empty_columns = []
        for col in df.columns:
            non_null_count = df[col].count()
            if non_null_count > 0:
                non_empty_columns.append(col)
            else:
                logger.warning(f"列{col}全是缺失值，将其移除")
        
        # 如果所有列都是空的，使用0填充
        if not non_empty_columns:
            logger.error("所有特征列都是缺失值，使用0矩阵代替")
            return np.zeros_like(X)
        
        # 只保留非空列
        df = df[non_empty_columns]
        
        # 使用中位数填充缺失值
        try:
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(df)
        except ValueError as e:
            # 如果中位数填充失败，使用0填充
            logger.warning(f"中位数填充失败: {str(e)}，使用0填充")
            X_imputed = np.nan_to_num(df.values, nan=0.0)
        
        # 处理无限值
        X_imputed = np.nan_to_num(X_imputed, posinf=0.0, neginf=0.0)
        
        return X_imputed
    
    def prepare_data(self, X, y, test_size=0.2, perform_feature_selection=False):
        """准备训练和测试数据
        
        Args:
            X: 特征矩阵
            y: 标签
            test_size: 测试集比例
            perform_feature_selection: 是否执行特征选择
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # 预处理数据
        X_preprocessed = self._preprocess_data(X)
        
        # 检查数据格式
        if len(X_preprocessed.shape) == 1:
            X_preprocessed = X_preprocessed.reshape(-1, 1)
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_preprocessed, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # 特征选择
        if perform_feature_selection and X_train.shape[1] > 10:
            selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            logger.info(f"特征选择完成，保留特征数量: {X_train.shape[1]}")
            
            # 更新特征名称，保留被选择的特征
            if self.feature_names is not None and len(self.feature_names) >= X_train.shape[1]:
                # 获取选择器的支持掩码
                support_mask = selector.get_support()
                # 过滤特征名称，只保留被选择的特征
                if len(support_mask) == len(self.feature_names):
                    selected_features = [self.feature_names[i] for i, selected in enumerate(support_mask) if selected]
                    # 确保选择的特征数量与实际使用的特征数量一致
                    if len(selected_features) == X_train.shape[1]:
                        self.feature_names = selected_features
                        logger.info(f"更新特征名称列表，保留{len(self.feature_names)}个特征名称")
        
        # 标准化特征 - 在特征选择后进行标准化，确保标准化器和模型使用相同数量的特征
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X, y, n_estimators=100, test_size=0.2, hyper_param_tuning=False, perform_feature_selection=False):
        """训练模型
        
        Args:
            X: 特征矩阵
            y: 标签
            n_estimators: 决策树数量
            test_size: 测试集比例
            hyper_param_tuning: 是否进行超参数调优
            perform_feature_selection: 是否执行特征选择
            
        Returns:
            训练结果字典
        """
        # 检查数据数量
        if len(X) < 100:
            logger.warning(f"训练数据不足100组，当前只有{len(X)}组数据")
        
        # 准备数据
        X_train, X_test, y_train, y_test = self.prepare_data(X, y, test_size, perform_feature_selection)
        
        try:
            if hyper_param_tuning:
                # 根据模型类型定义超参数搜索空间
                if self.model_type == 'random_forest':
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 20, 30, 40],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['auto', 'sqrt']
                    }
                    base_estimator = RandomForestClassifier(random_state=42, n_jobs=-1)
                else:  # gradient_boosting
                    param_grid = {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    }
                    base_estimator = GradientBoostingClassifier(random_state=42)
                
                # 使用网格搜索进行超参数调优
                grid_search = GridSearchCV(
                    estimator=base_estimator,
                    param_grid=param_grid,
                    cv=5,
                    n_jobs=-1,
                    verbose=1,
                    scoring='accuracy'
                )
                
                logger.info("开始超参数调优...")
                grid_search.fit(X_train, y_train)
                
                # 使用最佳参数配置模型
                logger.info(f"最佳参数: {grid_search.best_params_}")
                self.model = grid_search.best_estimator_
            else:
                # 更新模型参数
                if self.model_type == 'random_forest':
                    self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
                else:
                    self.model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
                
                # 训练模型
                logger.info(f"开始训练模型（{n_estimators}棵决策树）...")
                self.model.fit(X_train, y_train)
            
            # 交叉验证
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
            logger.info(f"交叉验证分数: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # 在测试集上评估模型
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
            
            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            report = classification_report(y_test, y_pred, target_names=list(self.scene_mapping.values()))
            cm = confusion_matrix(y_test, y_pred)
            
            logger.info(f"模型训练完成，准确率: {accuracy:.4f}")
            logger.info(f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
            logger.info("分类报告:\n" + report)
            logger.info("混淆矩阵:\n" + str(cm))
            
            # 特征重要性分析（如果模型支持）
            if hasattr(self.model, 'feature_importances_'):
                self._analyze_feature_importance(X_train.shape[1])
            
            # 保存模型
            self.save_model()
            
            # 返回训练结果
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_scores': cv_scores,
                'classification_report': report,
                'confusion_matrix': cm,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'model_path': self.model_path,
                'y_pred_proba': y_pred_proba
            }
        except Exception as e:
            logger.error(f"训练模型失败: {str(e)}")
            return None
    
    def _analyze_feature_importance(self, n_features):
        """分析特征重要性并记录日志"""
        if self.feature_names:
            if len(self.feature_names) == n_features:
                try:
                    importances = self.model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    # 记录最重要的10个特征
                    logger.info("特征重要性排名（前10个）:")
                    for f in range(min(10, n_features)):
                        if indices[f] < len(self.feature_names):
                            logger.info(f"{f + 1}. {self.feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
                except Exception as e:
                    logger.error(f"提取特征重要性时出错: {str(e)}")
            else:
                logger.warning(f"特征名称数量不匹配: 期望{n_features}个，实际{len(self.feature_names)}个")
        else:
            logger.warning("无法进行特征重要性分析，特征名称未设置")
    
    def visualize_results(self, results, save_dir='visualizations'):
        """可视化训练结果
        
        Args:
            results: 训练结果字典
            save_dir: 可视化结果保存目录
        """
        # 确保目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 混淆矩阵可视化
        plt.figure(figsize=(10, 8))
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=list(self.scene_mapping.values()),
                    yticklabels=list(self.scene_mapping.values()))
        
        # 强制使用safe_text函数处理所有文本
        plt.xlabel(safe_text('预测标签'))
        plt.ylabel(safe_text('真实标签'))
        plt.title(safe_text('混淆矩阵'))
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
        logger.info(f"混淆矩阵可视化已保存到: {os.path.join(save_dir, 'confusion_matrix.png')}")
        
        # 交叉验证分数可视化
        if 'cv_scores' in results:
            plt.figure(figsize=(8, 6))
            plt.bar(range(len(results['cv_scores'])), results['cv_scores'], color='skyblue')
            
            # 强制使用safe_text函数处理所有文本
            mean_label = f'平均分数: {results["cv_scores"].mean():.4f}'
            plt.axhline(y=results['cv_scores'].mean(), color='r', linestyle='--', label=safe_text(mean_label))
            plt.xlabel(safe_text('折数'))
            plt.ylabel(safe_text('准确率'))
            plt.title(safe_text('交叉验证分数'))
        
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'cv_scores.png'))
            plt.close()
            logger.info(f"交叉验证分数可视化已保存到: {os.path.join(save_dir, 'cv_scores.png')}")
    
    def _read_csv_with_encoding(self, file_path):
        """尝试使用不同的编码读取CSV文件"""
        encodings = ['utf-8', 'gb2312', 'gbk', 'ansi', 'latin1']
        
        for encoding in encodings:
            try:
                logger.info(f"尝试使用{encoding}编码读取文件: {file_path}")
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"成功使用{encoding}编码读取文件")
                return df
            except UnicodeDecodeError:
                logger.warning(f"使用{encoding}编码读取文件失败")
                continue
        
        # 如果所有编码都失败，尝试使用errors='replace'参数
        try:
            logger.info(f"尝试使用utf-8编码(替换错误字符)读取文件: {file_path}")
            df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
            logger.info("成功使用utf-8(替换错误字符)编码读取文件")
            return df
        except Exception as e:
            logger.error(f"所有编码尝试都失败: {str(e)}")
            raise
    
    def _process_performance_data_file(self, file_path):
        """处理Performance_Data格式的文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            处理后的DataFrame或None（如果失败）
        """
        try:
            logger.info(f"处理Performance_Data格式文件: {file_path}")
            
            # 尝试使用新的三行一组解析方法
            result = self._parse_csv_by_triple_lines(file_path)
            if result is not None:
                return result
            
            # 如果三行一组解析失败，尝试传统格式解析
            logger.warning("三行一组格式解析失败，尝试使用传统格式解析")
            
            # 使用PerformanceDataReader处理文件
            reader = PerformanceDataReader(self.scene_mapping, self.inverse_scene_mapping)
            df = reader.parse_file(file_path)
            
            # 更新场景映射（如果有新的场景类型被发现）
            if df is not None:
                self.scene_mapping = reader.scene_mapping
                self.inverse_scene_mapping = reader.inverse_scene_mapping
            
            return df
        except Exception as e:
            logger.error(f"处理文件失败: {str(e)}")
            # 降级处理：尝试使用简单的行解析方法
            return self._simple_parse_performance_data_file(file_path)
    
    def _clean_feature_name(self, name):
        """清理特征名称，去除#及其后面的内容"""
        if '#' in name:
            return name.split('#')[0].strip()
        return name.strip()
    
    def _parse_csv_by_triple_lines(self, file_path):
        """专门解析Performance_Data.csv的三行一组格式
        
        格式说明：
        - 第一行：性能参数（逗号分隔的特征名称）
        - 第二行：性能数据（逗号分隔的数值）
        - 第三行：场景结果（直接是场景名称）
        
        Args:
            file_path: Performance_Data.csv文件路径
            
        Returns:
            处理后的DataFrame或None（如果失败）
        """
        try:
            logger.info(f"尝试使用三行一组格式解析文件: {file_path}")
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                
            # 检查文件是否为空
            if not lines:
                logger.error("文件内容为空")
                return None
            
            # 解析文件内容（三行一组）
            feature_names = None
            features_list = []
            labels_list = []
            
            # 处理第一组来获取特征名称
            if len(lines) >= 3:
                # 第一组的第一行是特征名称
                feature_line = lines[0]
                raw_feature_names = [name.strip() for name in feature_line.split(',') if name.strip()]
                
                # 清理特征名称，去除#及其后面的内容
                feature_names = [self._clean_feature_name(name) for name in raw_feature_names]
                
                if not feature_names:
                    logger.warning("未提取到有效的特征名称")
                    return None
                
                logger.info(f"从第一行提取到{len(feature_names)}个特征名称")
                
                # 按三行一组处理数据
                for i in range(0, len(lines), 3):
                    # 确保有足够的行
                    if i + 2 < len(lines):
                        # 第二行是数值数据
                        data_line = lines[i+1]
                        current_values = [val.strip() for val in data_line.split(',')]
                        
                        # 第三行是场景结果
                        scene_line = lines[i+2]
                        
                        # 转换数值
                        try:
                            # 确保数值数量与特征名称数量匹配
                            if len(current_values) >= len(feature_names):
                                current_values = current_values[:len(feature_names)]
                            else:
                                # 如果数值数量不足，用NaN填充
                                current_values += [''] * (len(feature_names) - len(current_values))
                            
                            # 转换为浮点数
                            numeric_values = []
                            for val in current_values:
                                if val == '':
                                    numeric_values.append(np.nan)
                                else:
                                    try:
                                        numeric_values.append(float(val))
                                    except ValueError:
                                        numeric_values.append(np.nan)
                            
                            features_list.append(numeric_values)
                            
                            # 转换场景结果为标签
                            if scene_line in self.inverse_scene_mapping:
                                labels_list.append(self.inverse_scene_mapping[scene_line])
                            else:
                                # 如果是新的场景类型，添加到映射中
                                new_label = len(self.scene_mapping)
                                self.scene_mapping[new_label] = scene_line
                                self.inverse_scene_mapping[scene_line] = new_label
                                labels_list.append(new_label)
                                logger.info(f"发现新的场景类型: {scene_line}，映射为标签{new_label}")
                            
                        except Exception as e:
                            logger.warning(f"解析第{i//3+1}组数据失败: {str(e)}")
                            continue
            
            if not features_list:
                logger.warning("未成功解析到三行一组格式的数据")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(features_list, columns=feature_names)
            
            # 合并相同名称的列并相加
            df_merged = self._merge_duplicate_columns(df)
            
            # 添加标签列
            df_merged['scene_label'] = labels_list
            
            logger.info(f"成功使用三行一组格式解析Performance_Data.csv，共得到{len(df_merged)}条有效数据")
            return df_merged
            
        except Exception as e:
            logger.error(f"三行一组格式解析失败: {str(e)}")
            return None
    
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
    
    def _simple_parse_performance_data_file(self, file_path):
        """使用更简单的方式解析Performance_Data格式文件（降级方案）
        
        Args:
            file_path: 文件路径
            
        Returns:
            处理后的DataFrame或None（如果失败）
        """
        try:
            logger.info(f"使用降级方案处理Performance_Data格式文件: {file_path}")
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                
            # 检查文件是否为空
            if not lines:
                logger.error("文件内容为空")
                return None
            
            # 简单解析文件内容
            feature_names = []
            features_list = []
            labels_list = []
            current_values = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # 尝试识别性能参数行（不依赖特定前缀）
                if current_values is None and line and ',' in line:
                    # 假设这是特征名称行
                    raw_feature_names = [name.strip() for name in line.split(',')]
                    
                    # 清理特征名称，去除#及其后面的内容
                    feature_names = [self._clean_feature_name(name) for name in raw_feature_names]
                    
                    # 下一行应该是数值数据
                    if i + 1 < len(lines):
                        data_line = lines[i+1].strip().rstrip('"')
                        current_values = [val.strip() for val in data_line.split(',')]
                
                # 处理Scene_Result行或直接的场景结果行
                elif (line.startswith('Scene_Result:"') or line in self.inverse_scene_mapping or line) and current_values is not None:
                    # 提取场景结果
                    if line.startswith('Scene_Result:"'):
                        scene_result = line[len('Scene_Result:"'):].strip('"')
                    else:
                        # 直接使用行内容作为场景结果
                        scene_result = line
                    
                    # 转换数值
                    try:
                        # 确保数值数量与特征名称数量匹配
                        if len(current_values) >= len(feature_names):
                            current_values = current_values[:len(feature_names)]
                        else:
                            # 如果数值数量不足，用NaN填充
                            current_values += [''] * (len(feature_names) - len(current_values))
                        
                        # 转换为浮点数
                        numeric_values = []
                        for val in current_values:
                            if val == '':
                                numeric_values.append(np.nan)
                            else:
                                try:
                                    numeric_values.append(float(val))
                                except ValueError:
                                    numeric_values.append(np.nan)
                        
                        features_list.append(numeric_values)
                        
                        # 转换场景结果为标签
                        if scene_result in self.inverse_scene_mapping:
                            labels_list.append(self.inverse_scene_mapping[scene_result])
                        else:
                            # 如果是新的场景类型，添加到映射中
                            new_label = len(self.scene_mapping)
                            self.scene_mapping[new_label] = scene_result
                            self.inverse_scene_mapping[scene_result] = new_label
                            labels_list.append(new_label)
                            logger.info(f"发现新的场景类型: {scene_result}，映射为标签{new_label}")
                        
                    except Exception as e:
                        logger.warning(f"解析数值数据失败: {str(e)}")
                    
                    # 重置当前值
                    current_values = None
            
            if not features_list:
                logger.error("未成功解析任何数据")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(features_list, columns=feature_names)
            
            # 合并相同名称的列并相加
            df_merged = self._merge_duplicate_columns(df)
            
            # 添加标签列
            df_merged['scene_label'] = labels_list
            
            logger.info(f"成功使用降级方案处理Performance_Data文件，共得到{len(df_merged)}条有效数据")
            return df_merged
            
        except Exception as e:
            logger.error(f"降级方案处理文件失败: {str(e)}")
            return None
    
    def train_from_files(self, data_file, labels_file=None, n_estimators=100, test_size=0.2, 
                        hyper_param_tuning=False, perform_feature_selection=False):
        """从文件加载数据并训练模型
        
        Args:
            data_file: 数据文件路径，可以是包含特征和标签的单一文件
            labels_file: 标签文件路径，如果为None则从data_file中提取标签列
            n_estimators: 决策树数量
            test_size: 测试集比例
            hyper_param_tuning: 是否进行超参数调优
            perform_feature_selection: 是否执行特征选择
            
        Returns:
            训练结果字典或None（如果失败）
        """
        try:
            # 加载数据 - 使用编码检测函数
            df = self._read_csv_with_encoding(data_file)
            
            # 清理数据中的无效字符
            df = df.replace(['\ufffd', '\u0000'], '', regex=True)
            
            # 检查是否包含时间戳和源文件路径等非特征列
            non_feature_columns = ['process_time', 'source_file', 'timestamp', 'datetime']
            feature_columns = [col for col in df.columns if col not in non_feature_columns and col.lower() not in non_feature_columns]
            
            # 如果提供了单独的标签文件
            if labels_file and os.path.exists(labels_file):
                y_df = self._read_csv_with_encoding(labels_file)
                y = y_df.values.ravel()
                X = df[feature_columns].values
            else:
                # 假设标签在data_file中，需要用户指定或自动检测
                # 这里简化处理，实际应用中需要更智能的检测或用户指定
                logger.warning("未提供标签文件，假设数据中不包含标签列，使用模拟标签进行训练")
                # 为了演示，我们使用模拟标签
                X = df[feature_columns].values
                y = self._generate_mock_labels(len(X))
            
            # 保存特征名称
            self.feature_names = feature_columns
            
            # 检查数据一致性
            if len(X) != len(y):
                logger.error(f"数据和标签数量不匹配: {len(X)} vs {len(y)}")
                return None
            
            # 确保标签是整数
            y = y.astype(int)
            
            # 训练模型
            return self.train(X, y, n_estimators, test_size, hyper_param_tuning, perform_feature_selection)
        except Exception as e:
            logger.error(f"从文件训练模型失败: {str(e)}")
            return None
    
    def _generate_mock_labels(self, n_samples):
        """为无标签数据生成模拟标签"""
        # 为每种场景生成不同比例的数据
        n_compute = int(n_samples * 0.3)  # 30%计算密集型
        n_data = int(n_samples * 0.25)    # 25%数据密集型
        n_hybrid = int(n_samples * 0.25)  # 25%混合负载型
        n_light = n_samples - n_compute - n_data - n_hybrid  # 剩余为轻量负载型
        
        # 生成对应的标签
        y = np.concatenate([
            np.zeros(n_compute),       # 计算密集型
            np.ones(n_data),           # 数据密集型
            np.full(n_hybrid, 2),      # 混合负载型
            np.full(n_light, 3)        # 轻量负载型
        ])
        
        # 打乱数据顺序
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        y = y[indices]
        
        # 确保标签是整数
        return y.astype(int)
    
    def train_from_dataframe(self, df, label_column=None, n_estimators=100, test_size=0.2, 
                           hyper_param_tuning=False, perform_feature_selection=False):
        """从DataFrame加载数据并训练模型
        
        Args:
            df: 包含特征和标签的DataFrame
            label_column: 标签列名，如果为None则生成模拟标签
            n_estimators: 决策树数量
            test_size: 测试集比例
            hyper_param_tuning: 是否进行超参数调优
            perform_feature_selection: 是否执行特征选择
            
        Returns:
            训练结果字典或None（如果失败）
        """
        try:
            # 清理数据中的无效字符
            df = df.replace(['\ufffd', '\u0000'], '', regex=True)
            
            # 检查是否包含时间戳和源文件路径等非特征列
            non_feature_columns = ['process_time', 'source_file', 'timestamp', 'datetime']
            if label_column:
                non_feature_columns.append(label_column)
            
            feature_columns = [col for col in df.columns if col not in non_feature_columns and col.lower() not in non_feature_columns]
            
            # 提取特征
            X = df[feature_columns].values
            
            # 保存特征名称
            self.feature_names = feature_columns
            
            # 提取或生成标签
            if label_column and label_column in df.columns:
                y = df[label_column].values.ravel()
                # 确保标签是整数
                y = y.astype(int)
            else:
                logger.warning("未找到有效标签列，使用模拟标签进行训练")
                y = self._generate_mock_labels(len(X))
            
            # 训练模型
            return self.train(X, y, n_estimators, test_size, hyper_param_tuning, perform_feature_selection)
        except Exception as e:
            logger.error(f"从DataFrame训练模型失败: {str(e)}")
            return None
    
    def generate_simulation_data(self, num_samples=1000, n_features=50):
        """生成模拟训练数据，模拟HPC场景的真实数据分布
        
        Args:
            num_samples: 样本数量
            n_features: 特征数量
            
        Returns:
            X: 特征矩阵
            y: 标签
        """
        logger.info(f"生成{num_samples}组模拟训练数据，特征数量: {n_features}...")
        
        # 为每种场景生成不同特征的数据
        n_compute = int(num_samples * 0.3)  # 30%计算密集型
        n_data = int(num_samples * 0.25)    # 25%数据密集型
        n_hybrid = int(num_samples * 0.25)  # 25%混合负载型
        n_light = num_samples - n_compute - n_data - n_hybrid  # 剩余为轻量负载型
        
        # 根据HPC场景特点生成更真实的数据分布
        # 计算密集型：CPU相关特征值较高，I/O较低
        compute_cpu_features = np.random.normal(loc=0.7, scale=0.15, size=(n_compute, int(n_features*0.4)))
        compute_io_features = np.random.normal(loc=0.3, scale=0.1, size=(n_compute, int(n_features*0.3)))
        compute_other_features = np.random.normal(loc=0.5, scale=0.2, size=(n_compute, n_features - int(n_features*0.4) - int(n_features*0.3)))
        compute_data = np.hstack([compute_cpu_features, compute_io_features, compute_other_features])
        
        # 数据密集型：I/O相关特征值较高，CPU中等
        data_cpu_features = np.random.normal(loc=0.5, scale=0.15, size=(n_data, int(n_features*0.4)))
        data_io_features = np.random.normal(loc=0.8, scale=0.1, size=(n_data, int(n_features*0.3)))
        data_other_features = np.random.normal(loc=0.5, scale=0.2, size=(n_data, n_features - int(n_features*0.4) - int(n_features*0.3)))
        data_data = np.hstack([data_cpu_features, data_io_features, data_other_features])
        
        # 混合负载型：CPU和I/O都较高
        hybrid_cpu_features = np.random.normal(loc=0.65, scale=0.15, size=(n_hybrid, int(n_features*0.4)))
        hybrid_io_features = np.random.normal(loc=0.65, scale=0.15, size=(n_hybrid, int(n_features*0.3)))
        hybrid_other_features = np.random.normal(loc=0.6, scale=0.2, size=(n_hybrid, n_features - int(n_features*0.4) - int(n_features*0.3)))
        hybrid_data = np.hstack([hybrid_cpu_features, hybrid_io_features, hybrid_other_features])
        
        # 轻量负载型：所有特征都较低
        light_cpu_features = np.random.normal(loc=0.2, scale=0.1, size=(n_light, int(n_features*0.4)))
        light_io_features = np.random.normal(loc=0.2, scale=0.1, size=(n_light, int(n_features*0.3)))
        light_other_features = np.random.normal(loc=0.3, scale=0.15, size=(n_light, n_features - int(n_features*0.4) - int(n_features*0.3)))
        light_data = np.hstack([light_cpu_features, light_io_features, light_other_features])
        
        # 合并数据
        X = np.vstack([compute_data, data_data, hybrid_data, light_data])
        
        # 归一化到0-1范围
        X = np.clip(X, 0, 1)
        
        # 生成对应的标签
        y = np.concatenate([
            np.zeros(n_compute),       # 计算密集型
            np.ones(n_data),           # 数据密集型
            np.full(n_hybrid, 2),      # 混合负载型
            np.full(n_light, 3)        # 轻量负载型
        ])
        
        # 打乱数据顺序
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # 确保标签是整数
        y = y.astype(int)
        
        return X, y
        
    def train_all_hpc_scenes(self, data_source=None, num_samples=1000, hyper_param_tuning=True):
        """训练所有HPC场景的识别模型
        
        Args:
            data_source: 数据源，可以是文件路径、DataFrame或None（使用模拟数据）
            num_samples: 当使用模拟数据时，生成的样本数量
            hyper_param_tuning: 是否进行超参数调优
            
        Returns:
            训练好的模型
        """
        logger.info("开始训练所有HPC场景的识别模型")
        
        try:
            # 获取训练数据
            if data_source is None:
                # 使用模拟数据
                logger.info(f"使用模拟数据进行训练，生成{num_samples}个样本")
                self.train_with_simulation_data(num_samples, hyper_param_tuning=hyper_param_tuning)
            elif isinstance(data_source, pd.DataFrame):
                # 使用提供的DataFrame
                logger.info(f"使用提供的DataFrame进行训练，共{len(data_source)}个样本")
                self.train_from_dataframe(data_source, hyper_param_tuning=hyper_param_tuning)
            elif isinstance(data_source, str):
                # 使用文件数据
                logger.info(f"从文件{data_source}加载训练数据")
                if data_source.endswith('.txt'):
                    # 处理Performance_Data.txt文件
                    self.train_with_performance_data(data_source, hyper_param_tuning=hyper_param_tuning)
                elif data_source.endswith('.csv'):
                    # 处理CSV文件
                    self.train_from_files([data_source], hyper_param_tuning=hyper_param_tuning)
                else:
                    logger.error(f"不支持的文件格式: {data_source}")
                    return None
            else:
                logger.error(f"不支持的数据源类型: {type(data_source)}")
                return None
            
            # 确保模型已训练成功
            if self.model is not None:
                logger.info("所有HPC场景的识别模型训练完成")
                return self.model
            else:
                logger.error("模型训练失败，模型对象为空")
                return None
        except Exception as e:
            logger.error(f"训练HPC场景识别模型时发生错误: {str(e)}")
            # 记录异常的堆栈信息
            import traceback
            logger.debug(f"异常堆栈:\n{traceback.format_exc()}")
            return None
    
    def train_with_simulation_data(self, num_samples=1000, n_estimators=100, hyper_param_tuning=False):
        """使用模拟数据训练模型
        
        Args:
            num_samples: 模拟样本数量
            n_estimators: 决策树数量
            hyper_param_tuning: 是否进行超参数调优
            
        Returns:
            训练结果字典或None（如果失败）
        """
        # 生成模拟数据，根据CSV文件特点设置特征数量
        X, y = self.generate_simulation_data(num_samples, n_features=50)  # 假设约50个特征
        
        # 训练模型
        return self.train(X, y, n_estimators, hyper_param_tuning=hyper_param_tuning)
    
    def train_with_all_transformed_data(self, data_file='all_transformed_data.csv', **kwargs):
        """使用all_transformed_data.csv文件训练模型的便捷方法
        
        Args:
            data_file: 数据文件路径
            **kwargs: 传递给train_from_files的其他参数
            
        Returns:
            训练结果字典或None（如果失败）
        """
        try:
            logger.info(f"尝试处理all_transformed_data.csv文件: {data_file}")
            
            # 首先尝试使用_process_performance_data_file处理，因为all_transformed_data.csv
            # 实际上是Performance_Data格式，而不是标准CSV格式
            df = self._process_performance_data_file(data_file)
            
            if df is not None and len(df) > 0:
                logger.info(f"成功使用Performance_Data格式处理文件，获得{len(df)}条数据")
                
                # 提取特征和标签
                non_feature_columns = ['scene_label', 'process_time', 'source_file', 'timestamp', 'datetime']
                feature_columns = [col for col in df.columns if col not in non_feature_columns and col.lower() not in non_feature_columns]
                
                X = df[feature_columns].values
                y = df['scene_label'].values.ravel()
                
                # 保存特征名称
                self.feature_names = feature_columns
                
                # 训练模型
                return self.train(X, y, **kwargs)
            else:
                logger.warning("使用Performance_Data格式处理失败，尝试使用标准CSV格式处理")
                # 如果处理失败，回退到原来的方法
                return self.train_from_files(data_file, **kwargs)
        except Exception as e:
            logger.error(f"处理all_transformed_data.csv文件失败: {str(e)}")
            # 如果出错，回退到原来的方法
            return self.train_from_files(data_file, **kwargs)

    def train_with_performance_data(self, data_file='Performance_Data.txt', **kwargs):
        """使用Performance_Data.txt文件训练模型的便捷方法
        
        Args:
            data_file: Performance_Data格式数据文件路径
            **kwargs: 传递给train方法的其他参数
            
        Returns:
            训练结果字典或None（如果失败）
        """
        try:
            logger.info(f"尝试处理Performance_Data文件: {data_file}")
            
            # 检查文件是否存在
            if not os.path.exists(data_file):
                logger.error(f"文件不存在: {data_file}")
                return None
                
            # 使用_process_performance_data_file处理Performance_Data格式文件
            df = self._process_performance_data_file(data_file)
            
            if df is not None and len(df) > 0:
                logger.info(f"成功处理Performance_Data文件，获得{len(df)}条数据")
                
                # 数据预处理
                # 1. 识别并移除非特征列
                non_feature_columns = ['scene_label', 'process_time', 'source_file', 'timestamp', 'datetime']
                feature_columns = [col for col in df.columns if col not in non_feature_columns and col.lower() not in non_feature_columns]
                
                # 2. 识别并移除全是缺失值的特征列
                valid_feature_columns = []
                for col in feature_columns:
                    non_null_count = df[col].count()
                    if non_null_count > 0:
                        valid_feature_columns.append(col)
                    else:
                        logger.warning(f"特征{col}全是缺失值，将其移除")
                
                if not valid_feature_columns:
                    logger.error("所有特征列都是缺失值，无法训练模型")
                    return None
                
                # 3. 处理缺失值
                # 使用中位数填充数值型特征的缺失值
                df_valid = df[valid_feature_columns + ['scene_label']].copy()
                numeric_columns = df_valid.select_dtypes(include=[np.number]).columns
                
                for col in numeric_columns:
                    if col != 'scene_label':  # 排除标签列
                        try:
                            median_value = df_valid[col].median()
                            df_valid[col] = df_valid[col].fillna(median_value)
                        except (ValueError, TypeError) as e:
                            # 如果中位数计算失败，使用0填充
                            logger.warning(f"计算特征{col}的中位数失败: {str(e)}，使用0填充")
                            df_valid[col] = df_valid[col].fillna(0)
                
                # 4. 提取特征和标签
                X = df_valid[valid_feature_columns].values
                y = df_valid['scene_label'].values.ravel()
                
                # 保存特征名称
                self.feature_names = valid_feature_columns
                
                # 训练模型
                return self.train(X, y, **kwargs)
            else:
                logger.error("未成功解析Performance_Data文件")
                return None
        except Exception as e:
            logger.error(f"处理Performance_Data文件失败: {str(e)}")
            return None

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='工业场景识别模型训练器')
    parser.add_argument('--data-file', type=str, help='指定Performance_Data.csv文件路径，默认使用/usr/share/industrial-scene-optimizer/Performance_Data.csv')
    parser.add_argument('--model-dir', type=str, help='指定模型保存目录，默认使用/usr/share/industrial-scene-optimizer/models')
    parser.add_argument('--n-estimators', type=int, default=200, help='决策树数量，默认为200')
    parser.add_argument('--test-size', type=float, default=0.2, help='测试集比例，默认为0.2')
    parser.add_argument('--hyper-param-tuning', action='store_true', help='是否进行超参数调优')
    parser.add_argument('--feature-selection', action='store_true', help='是否执行特征选择')
    args = parser.parse_args()
    
    # 确定数据文件路径
    data_file = args.data_file if args.data_file else DEFAULT_PERFORMANCE_DATA_PATH
    
    # 确定模型保存目录
    model_dir = args.model_dir if args.model_dir else DEFAULT_MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "scene_recognizer_model.pkl")
    
    # 初始化模型训练器
    trainer = ModelTrainer(model_path=model_path, model_type="random_forest")
    
    # 设置标准化器类型
    trainer.set_scaler_type('standard')
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        logger.error(f"数据文件不存在: {data_file}")
        logger.info(f"正在使用模拟数据进行训练...")
        # 使用模拟数据训练模型
        results = trainer.train_with_simulation_data(
            num_samples=1000,
            n_estimators=args.n_estimators,
            hyper_param_tuning=args.hyper_param_tuning
        )
    else:
        # 使用指定的Performance_Data文件训练模型
        logger.info(f"使用数据文件: {data_file} 进行训练")
        results = trainer.train_with_performance_data(
            data_file=data_file,
            n_estimators=args.n_estimators,
            test_size=args.test_size,
            hyper_param_tuning=args.hyper_param_tuning,
            perform_feature_selection=args.feature_selection
        )
    
    # 可视化结果
    if results:
        trainer.visualize_results(results)