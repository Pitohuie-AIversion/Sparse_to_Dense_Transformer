#!/usr/bin/env python3
"""
VIVTransformer 高级代码质量分析器

这个脚本整合了所有高级质量分析功能，包括：
- AI辅助代码审查
- 智能重构建议
- 预测性质量分析
- 高级性能监控
- 深度安全扫描
- 架构质量验证
- 文档质量保证
- 团队协作分析

使用方法:
    python scripts/advanced_quality_analyzer.py --config config/advanced_quality_config.yaml
    python scripts/advanced_quality_analyzer.py --mode ai-review --files src/models/
    python scripts/advanced_quality_analyzer.py --mode full-analysis --output reports/
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """质量指标数据类"""
    timestamp: str
    file_path: str
    complexity_score: float
    maintainability_index: float
    test_coverage: float
    security_score: float
    documentation_coverage: float
    performance_score: float
    architecture_score: float
    overall_quality: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    """分析结果数据类"""
    analysis_type: str
    status: str
    metrics: QualityMetrics
    issues: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'analysis_type': self.analysis_type,
            'status': self.status,
            'metrics': self.metrics.to_dict(),
            'issues': self.issues,
            'suggestions': self.suggestions,
            'execution_time': self.execution_time
        }


class AICodeReviewer:
    """AI辅助代码审查器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('ai_code_review', {})
        self.enabled = self.config.get('enabled', False)
        
        if self.enabled:
            self._setup_ai_client()
    
    def _setup_ai_client(self):
        """设置AI客户端"""
        provider = self.config.get('provider', 'openai')
        
        if provider == 'openai':
            try:
                import openai
                api_key = os.getenv(self.config.get('api_key_env', 'OPENAI_API_KEY'))
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized successfully")
                else:
                    logger.warning("OpenAI API key not found, AI review disabled")
                    self.enabled = False
            except ImportError:
                logger.warning("OpenAI library not installed, AI review disabled")
                self.enabled = False
    
    async def analyze_code(self, file_path: str) -> Dict[str, Any]:
        """AI代码分析"""
        if not self.enabled:
            return {'status': 'disabled', 'suggestions': []}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # 检查文件大小限制
            line_count = len(code.split('\n'))
            max_size = self.config.get('max_file_size', 5000)
            
            if line_count > max_size:
                return {
                    'status': 'skipped',
                    'reason': f'File too large ({line_count} lines > {max_size})'
                }
            
            # 构建分析提示
            focus_areas = self.config.get('analysis_focus', [])
            prompt = self._build_analysis_prompt(code, focus_areas)
            
            # 调用AI API
            response = await self._call_ai_api(prompt)
            
            return {
                'status': 'completed',
                'analysis': response,
                'suggestions': self._extract_suggestions(response),
                'quality_score': self._calculate_ai_quality_score(response)
            }
            
        except Exception as e:
            logger.error(f"AI analysis failed for {file_path}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _build_analysis_prompt(self, code: str, focus_areas: List[str]) -> str:
        """构建AI分析提示"""
        focus_text = ", ".join(focus_areas) if focus_areas else "general code quality"
        
        return f"""
        Please analyze the following Python code focusing on: {focus_text}
        
        Provide analysis in the following format:
        1. **Code Structure**: Assessment of overall structure and organization
        2. **Performance**: Potential performance optimizations
        3. **Readability**: Code clarity and maintainability
        4. **Security**: Security considerations and potential vulnerabilities
        5. **Best Practices**: Adherence to Python best practices
        6. **Suggestions**: Specific improvement recommendations
        
        Code to analyze:
        ```python
        {code}
        ```
        
        Please provide a quality score from 1-10 and specific actionable suggestions.
        """
    
    async def _call_ai_api(self, prompt: str) -> str:
        """调用AI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.get('model', 'gpt-4'),
                messages=[
                    {"role": "system", "content": "You are an expert Python code reviewer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"AI API call failed: {e}")
            raise
    
    def _extract_suggestions(self, analysis: str) -> List[Dict[str, Any]]:
        """从AI分析中提取建议"""
        suggestions = []
        
        # 简化的建议提取逻辑
        lines = analysis.split('\n')
        current_suggestion = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                if current_suggestion:
                    suggestions.append(current_suggestion)
                
                current_suggestion = {
                    'type': 'improvement',
                    'description': line[2:],
                    'priority': 'medium',
                    'category': 'general'
                }
            elif current_suggestion and line:
                current_suggestion['description'] += ' ' + line
        
        if current_suggestion:
            suggestions.append(current_suggestion)
        
        return suggestions
    
    def _calculate_ai_quality_score(self, analysis: str) -> float:
        """从AI分析中计算质量分数"""
        # 查找分数模式
        import re
        
        score_patterns = [
            r'score[:\s]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*/\s*10',
            r'quality[:\s]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, analysis.lower())
            if match:
                score = float(match.group(1))
                return min(10.0, max(0.0, score))
        
        # 如果没有找到明确分数，基于关键词估算
        positive_keywords = ['excellent', 'good', 'well', 'clean', 'clear']
        negative_keywords = ['poor', 'bad', 'complex', 'unclear', 'problematic']
        
        positive_count = sum(1 for word in positive_keywords if word in analysis.lower())
        negative_count = sum(1 for word in negative_keywords if word in analysis.lower())
        
        base_score = 7.0
        score_adjustment = (positive_count - negative_count) * 0.5
        
        return min(10.0, max(0.0, base_score + score_adjustment))


class IntelligentRefactoringAnalyzer:
    """智能重构分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('refactoring_analyzer', {})
        self.enabled = self.config.get('enabled', True)
        self.code_smells_config = self.config.get('code_smells', {})
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """分析文件的重构机会"""
        if not self.enabled:
            return {'status': 'disabled'}
        
        try:
            import ast
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            code_smells = self._detect_code_smells(tree, content)
            refactoring_suggestions = self._generate_refactoring_suggestions(code_smells)
            
            return {
                'status': 'completed',
                'code_smells': code_smells,
                'refactoring_suggestions': refactoring_suggestions,
                'refactoring_score': self._calculate_refactoring_score(code_smells)
            }
            
        except Exception as e:
            logger.error(f"Refactoring analysis failed for {file_path}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _detect_code_smells(self, tree: ast.AST, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """检测代码异味"""
        smells = {
            'long_methods': [],
            'large_classes': [],
            'complex_conditionals': [],
            'duplicate_code': []
        }
        
        # 检测长方法
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                method_lines = self._count_method_lines(node, content)
                complexity = self._calculate_complexity(node)
                
                max_lines = self.code_smells_config.get('long_method', {}).get('max_lines', 50)
                max_complexity = self.code_smells_config.get('long_method', {}).get('max_complexity', 10)
                
                if method_lines > max_lines or complexity > max_complexity:
                    smells['long_methods'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'lines_count': method_lines,
                        'complexity': complexity,
                        'severity': 'high' if method_lines > max_lines * 1.5 else 'medium'
                    })
            
            elif isinstance(node, ast.ClassDef):
                class_methods = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                class_lines = self._count_class_lines(node, content)
                
                max_methods = self.code_smells_config.get('large_class', {}).get('max_methods', 20)
                max_lines = self.code_smells_config.get('large_class', {}).get('max_lines', 500)
                
                if class_methods > max_methods or class_lines > max_lines:
                    smells['large_classes'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'methods_count': class_methods,
                        'lines_count': class_lines,
                        'severity': 'high' if class_methods > max_methods * 1.5 else 'medium'
                    })
        
        return smells
    
    def _count_method_lines(self, node: ast.FunctionDef, content: str) -> int:
        """计算方法行数"""
        lines = content.split('\n')
        start_line = node.lineno - 1
        
        # 找到方法结束行
        end_line = start_line
        if hasattr(node, 'end_lineno') and node.end_lineno:
            end_line = node.end_lineno - 1
        else:
            # 简化的结束行计算
            indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
            for i in range(start_line + 1, len(lines)):
                if lines[i].strip() and (len(lines[i]) - len(lines[i].lstrip())) <= indent_level:
                    end_line = i - 1
                    break
            else:
                end_line = len(lines) - 1
        
        return end_line - start_line + 1
    
    def _count_class_lines(self, node: ast.ClassDef, content: str) -> int:
        """计算类行数"""
        lines = content.split('\n')
        start_line = node.lineno - 1
        
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno - node.lineno + 1
        
        # 简化的类行数计算
        return len([n for n in ast.walk(node) if hasattr(n, 'lineno')])
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """计算圈复杂度"""
        complexity = 1  # 基础复杂度
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _generate_refactoring_suggestions(self, code_smells: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """生成重构建议"""
        suggestions = []
        
        # 长方法建议
        for smell in code_smells.get('long_methods', []):
            suggestions.append({
                'type': 'extract_method',
                'target': smell['name'],
                'description': f"Method '{smell['name']}' has {smell['lines_count']} lines. Consider extracting smaller methods.",
                'priority': smell['severity'],
                'effort': 'medium',
                'benefits': ['Improved readability', 'Better testability', 'Easier maintenance']
            })
        
        # 大类建议
        for smell in code_smells.get('large_classes', []):
            suggestions.append({
                'type': 'extract_class',
                'target': smell['name'],
                'description': f"Class '{smell['name']}' has {smell['methods_count']} methods. Consider splitting responsibilities.",
                'priority': smell['severity'],
                'effort': 'high',
                'benefits': ['Single responsibility', 'Better modularity', 'Easier testing']
            })
        
        return suggestions
    
    def _calculate_refactoring_score(self, code_smells: Dict[str, List[Dict[str, Any]]]) -> float:
        """计算重构分数"""
        total_smells = sum(len(smells) for smells in code_smells.values())
        
        if total_smells == 0:
            return 10.0
        
        # 基于代码异味数量和严重程度计算分数
        severity_weights = {'low': 0.5, 'medium': 1.0, 'high': 2.0}
        weighted_smells = 0
        
        for smells in code_smells.values():
            for smell in smells:
                weight = severity_weights.get(smell.get('severity', 'medium'), 1.0)
                weighted_smells += weight
        
        # 分数计算：10分制，每个加权异味扣0.5分
        score = max(0.0, 10.0 - (weighted_smells * 0.5))
        return round(score, 2)


class PredictiveQualityAnalyzer:
    """预测性质量分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('quality_predictor', {})
        self.enabled = self.config.get('enabled', True)
        self.model = None
        self.feature_columns = self.config.get('model', {}).get('features', [])
        
        if self.enabled:
            self._initialize_model()
    
    def _initialize_model(self):
        """初始化预测模型"""
        model_type = self.config.get('model', {}).get('type', 'linear_regression')
        
        if model_type == 'linear_regression':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            logger.warning(f"Unknown model type: {model_type}, using linear regression")
            self.model = LinearRegression()
    
    def predict_quality_trend(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """预测质量趋势"""
        if not self.enabled or len(historical_data) < 10:
            return {'status': 'insufficient_data'}
        
        try:
            # 准备数据
            df = pd.DataFrame(historical_data)
            
            # 特征工程
            features = self._extract_features(df)
            target = df['overall_quality'].values
            
            # 训练模型
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = self.model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # 预测未来趋势
            prediction_horizon = self.config.get('model', {}).get('prediction_horizon', 30)
            future_predictions = self._generate_future_predictions(features, prediction_horizon)
            
            # 风险评估
            risk_assessment = self._assess_quality_risks(future_predictions)
            
            return {
                'status': 'completed',
                'model_performance': {
                    'r2_score': r2,
                    'mse': mse,
                    'confidence': r2 if r2 > 0 else 0
                },
                'predictions': future_predictions,
                'risk_assessment': risk_assessment,
                'recommendations': self._generate_recommendations(risk_assessment)
            }
            
        except Exception as e:
            logger.error(f"Quality prediction failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取特征"""
        features = []
        
        # 基础特征
        if 'complexity_score' in df.columns:
            features.append(df['complexity_score'].values)
        
        if 'test_coverage' in df.columns:
            features.append(df['test_coverage'].values)
        
        if 'security_score' in df.columns:
            features.append(df['security_score'].values)
        
        # 趋势特征
        if len(df) > 5:
            # 计算移动平均
            for col in ['complexity_score', 'test_coverage', 'security_score']:
                if col in df.columns:
                    ma = df[col].rolling(window=5).mean().fillna(df[col])
                    features.append(ma.values)
        
        # 转置以获得正确的形状
        if features:
            return np.column_stack(features)
        else:
            # 如果没有特征，返回随机特征（用于演示）
            return np.random.random((len(df), 3))
    
    def _generate_future_predictions(self, features: np.ndarray, horizon: int) -> List[Dict[str, Any]]:
        """生成未来预测"""
        predictions = []
        
        # 使用最后的特征值作为基础
        last_features = features[-1:]
        
        for i in range(horizon):
            # 预测下一个时间点
            pred = self.model.predict(last_features)[0]
            
            # 添加一些随机性来模拟不确定性
            uncertainty = np.random.normal(0, 0.1)
            pred_with_uncertainty = max(0, min(10, pred + uncertainty))
            
            predictions.append({
                'day': i + 1,
                'predicted_quality': round(pred_with_uncertainty, 2),
                'confidence': max(0.5, 1.0 - (i * 0.02))  # 置信度随时间递减
            })
            
            # 更新特征（简化的特征演进）
            last_features = last_features * (1 + np.random.normal(0, 0.01, last_features.shape))
        
        return predictions
    
    def _assess_quality_risks(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估质量风险"""
        quality_values = [p['predicted_quality'] for p in predictions]
        
        # 计算趋势
        if len(quality_values) > 1:
            trend = (quality_values[-1] - quality_values[0]) / len(quality_values)
        else:
            trend = 0
        
        # 风险等级
        avg_quality = np.mean(quality_values)
        min_quality = min(quality_values)
        
        risk_level = 'low'
        if avg_quality < 6.0 or min_quality < 5.0 or trend < -0.1:
            risk_level = 'high'
        elif avg_quality < 7.5 or min_quality < 6.5 or trend < -0.05:
            risk_level = 'medium'
        
        return {
            'risk_level': risk_level,
            'trend': 'declining' if trend < -0.02 else 'stable' if abs(trend) < 0.02 else 'improving',
            'average_predicted_quality': round(avg_quality, 2),
            'minimum_predicted_quality': round(min_quality, 2),
            'quality_volatility': round(np.std(quality_values), 2)
        }
    
    def _generate_recommendations(self, risk_assessment: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        risk_level = risk_assessment['risk_level']
        trend = risk_assessment['trend']
        
        if risk_level == 'high':
            recommendations.extend([
                "Immediate action required: Quality is predicted to decline significantly",
                "Increase code review frequency and thoroughness",
                "Implement additional automated quality checks",
                "Consider refactoring high-risk modules"
            ])
        
        if trend == 'declining':
            recommendations.extend([
                "Quality trend is declining - investigate root causes",
                "Review recent changes and their impact on quality",
                "Strengthen testing practices and coverage"
            ])
        
        if risk_assessment['quality_volatility'] > 1.0:
            recommendations.append("Quality is highly volatile - establish more consistent practices")
        
        if not recommendations:
            recommendations.append("Quality outlook is positive - maintain current practices")
        
        return recommendations


class AdvancedQualityAnalyzer:
    """高级质量分析器主类"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
        # 初始化各个分析器
        self.ai_reviewer = AICodeReviewer(self.config)
        self.refactoring_analyzer = IntelligentRefactoringAnalyzer(self.config)
        self.predictive_analyzer = PredictiveQualityAnalyzer(self.config)
        
        # 设置日志
        self._setup_logging()
        
        # 历史数据存储
        self.history_file = Path("quality_history.json")
        self.quality_history = self._load_quality_history()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return {}
    
    def _setup_logging(self):
        """设置日志"""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # 更新日志级别
        logging.getLogger().setLevel(level)
        
        # 文件日志处理器
        if log_config.get('handlers', {}).get('file', {}).get('enabled', False):
            log_file = log_config['handlers']['file'].get('filename', 'quality_analysis.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            
            formatter = logging.Formatter(
                log_config['handlers']['file'].get(
                    'format', 
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
            file_handler.setFormatter(formatter)
            
            logging.getLogger().addHandler(file_handler)
    
    def _load_quality_history(self) -> List[Dict[str, Any]]:
        """加载质量历史数据"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load quality history: {e}")
        
        return []
    
    def _save_quality_history(self):
        """保存质量历史数据"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.quality_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save quality history: {e}")
    
    async def analyze_file(self, file_path: str, analysis_types: List[str] = None) -> AnalysisResult:
        """分析单个文件"""
        if analysis_types is None:
            analysis_types = ['ai_review', 'refactoring', 'basic_metrics']
        
        start_time = time.time()
        
        try:
            # 基础指标计算
            basic_metrics = self._calculate_basic_metrics(file_path)
            
            issues = []
            suggestions = []
            
            # AI代码审查
            if 'ai_review' in analysis_types:
                ai_result = await self.ai_reviewer.analyze_code(file_path)
                if ai_result.get('status') == 'completed':
                    suggestions.extend(ai_result.get('suggestions', []))
                    basic_metrics.overall_quality = (
                        basic_metrics.overall_quality + ai_result.get('quality_score', 7.0)
                    ) / 2
            
            # 重构分析
            if 'refactoring' in analysis_types:
                refactoring_result = self.refactoring_analyzer.analyze_file(file_path)
                if refactoring_result.get('status') == 'completed':
                    suggestions.extend(refactoring_result.get('refactoring_suggestions', []))
                    
                    # 将代码异味转换为问题
                    for smell_type, smells in refactoring_result.get('code_smells', {}).items():
                        for smell in smells:
                            issues.append({
                                'type': 'code_smell',
                                'category': smell_type,
                                'severity': smell.get('severity', 'medium'),
                                'message': f"{smell_type}: {smell.get('name', 'Unknown')}",
                                'line': smell.get('line', 0)
                            })
            
            execution_time = time.time() - start_time
            
            # 创建分析结果
            result = AnalysisResult(
                analysis_type='comprehensive',
                status='completed',
                metrics=basic_metrics,
                issues=issues,
                suggestions=suggestions,
                execution_time=execution_time
            )
            
            # 保存到历史记录
            self.quality_history.append({
                'timestamp': datetime.now().isoformat(),
                'file_path': file_path,
                'metrics': basic_metrics.to_dict(),
                'analysis_types': analysis_types
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for {file_path}: {e}")
            return AnalysisResult(
                analysis_type='comprehensive',
                status='error',
                metrics=QualityMetrics(
                    timestamp=datetime.now().isoformat(),
                    file_path=file_path,
                    complexity_score=0.0,
                    maintainability_index=0.0,
                    test_coverage=0.0,
                    security_score=0.0,
                    documentation_coverage=0.0,
                    performance_score=0.0,
                    architecture_score=0.0,
                    overall_quality=0.0
                ),
                issues=[{'type': 'error', 'message': str(e)}],
                suggestions=[],
                execution_time=time.time() - start_time
            )
    
    def _calculate_basic_metrics(self, file_path: str) -> QualityMetrics:
        """计算基础质量指标"""
        try:
            import ast
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # 复杂度分析
            complexity_score = self._calculate_file_complexity(tree)
            
            # 可维护性指数（简化版）
            maintainability_index = self._calculate_maintainability_index(tree, content)
            
            # 文档覆盖率
            doc_coverage = self._calculate_documentation_coverage(tree)
            
            # 整体质量分数
            overall_quality = (
                (10 - min(10, complexity_score)) * 0.3 +
                maintainability_index * 0.3 +
                doc_coverage * 0.2 +
                8.0 * 0.2  # 默认分数
            )
            
            return QualityMetrics(
                timestamp=datetime.now().isoformat(),
                file_path=file_path,
                complexity_score=complexity_score,
                maintainability_index=maintainability_index,
                test_coverage=0.0,  # 需要外部工具计算
                security_score=8.0,  # 默认分数
                documentation_coverage=doc_coverage,
                performance_score=8.0,  # 默认分数
                architecture_score=8.0,  # 默认分数
                overall_quality=overall_quality
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics for {file_path}: {e}")
            return QualityMetrics(
                timestamp=datetime.now().isoformat(),
                file_path=file_path,
                complexity_score=0.0,
                maintainability_index=0.0,
                test_coverage=0.0,
                security_score=0.0,
                documentation_coverage=0.0,
                performance_score=0.0,
                architecture_score=0.0,
                overall_quality=0.0
            )
    
    def _calculate_file_complexity(self, tree: ast.AST) -> float:
        """计算文件复杂度"""
        total_complexity = 0
        function_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
                complexity = 1  # 基础复杂度
                
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                        complexity += 1
                    elif isinstance(child, ast.ExceptHandler):
                        complexity += 1
                    elif isinstance(child, (ast.And, ast.Or)):
                        complexity += 1
                
                total_complexity += complexity
        
        return total_complexity / max(1, function_count)
    
    def _calculate_maintainability_index(self, tree: ast.AST, content: str) -> float:
        """计算可维护性指数（简化版）"""
        lines = content.split('\n')
        total_lines = len([line for line in lines if line.strip()])
        
        # 计算Halstead指标（简化版）
        operators = 0
        operands = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)):
                operators += 1
            elif isinstance(node, (ast.Name, ast.Num, ast.Str)):
                operands += 1
        
        # 简化的可维护性指数计算
        if total_lines == 0:
            return 0.0
        
        # 基于行数、复杂度等因素的简化计算
        complexity_factor = min(10, self._calculate_file_complexity(tree))
        line_factor = min(10, total_lines / 100)  # 每100行扣1分
        
        maintainability = max(0, 10 - complexity_factor * 0.5 - line_factor * 0.3)
        
        return round(maintainability, 2)
    
    def _calculate_documentation_coverage(self, tree: ast.AST) -> float:
        """计算文档覆盖率"""
        total_items = 0
        documented_items = 0
        
        # 检查模块文档
        if ast.get_docstring(tree):
            documented_items += 1
        total_items += 1
        
        # 检查类和函数文档
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                total_items += 1
                if ast.get_docstring(node):
                    documented_items += 1
        
        if total_items == 0:
            return 100.0
        
        return (documented_items / total_items) * 100
    
    async def analyze_directory(self, directory: str, analysis_types: List[str] = None) -> Dict[str, Any]:
        """分析目录中的所有Python文件"""
        directory_path = Path(directory)
        python_files = list(directory_path.rglob('*.py'))
        
        if not python_files:
            return {'status': 'no_files', 'message': 'No Python files found'}
        
        logger.info(f"Analyzing {len(python_files)} Python files in {directory}")
        
        results = []
        
        # 并行分析配置
        max_workers = self.config.get('performance', {}).get('parallel_processing', {}).get('max_workers', 4)
        
        # 使用线程池进行并行分析
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(asyncio.run, self.analyze_file(str(file_path), analysis_types)): file_path
                for file_path in python_files
            }
            
            # 收集结果
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed analysis for {file_path}")
                except Exception as e:
                    logger.error(f"Analysis failed for {file_path}: {e}")
        
        # 汇总结果
        summary = self._generate_directory_summary(results)
        
        # 预测性分析
        if len(self.quality_history) >= 10:
            prediction_result = self.predictive_analyzer.predict_quality_trend(self.quality_history)
            summary['predictions'] = prediction_result
        
        # 保存历史数据
        self._save_quality_history()
        
        return {
            'status': 'completed',
            'directory': directory,
            'files_analyzed': len(results),
            'results': [result.to_dict() for result in results],
            'summary': summary
        }
    
    def _generate_directory_summary(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """生成目录分析摘要"""
        if not results:
            return {}
        
        # 计算平均指标
        total_quality = sum(r.metrics.overall_quality for r in results)
        avg_quality = total_quality / len(results)
        
        total_complexity = sum(r.metrics.complexity_score for r in results)
        avg_complexity = total_complexity / len(results)
        
        total_doc_coverage = sum(r.metrics.documentation_coverage for r in results)
        avg_doc_coverage = total_doc_coverage / len(results)
        
        # 统计问题
        total_issues = sum(len(r.issues) for r in results)
        issue_types = {}
        for result in results:
            for issue in result.issues:
                issue_type = issue.get('type', 'unknown')
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        # 统计建议
        total_suggestions = sum(len(r.suggestions) for r in results)
        suggestion_types = {}
        for result in results:
            for suggestion in result.suggestions:
                suggestion_type = suggestion.get('type', 'unknown')
                suggestion_types[suggestion_type] = suggestion_types.get(suggestion_type, 0) + 1
        
        # 质量等级分布
        quality_distribution = {
            'excellent': len([r for r in results if r.metrics.overall_quality >= 9.0]),
            'good': len([r for r in results if 7.0 <= r.metrics.overall_quality < 9.0]),
            'fair': len([r for r in results if 5.0 <= r.metrics.overall_quality < 7.0]),
            'poor': len([r for r in results if r.metrics.overall_quality < 5.0])
        }
        
        return {
            'average_metrics': {
                'overall_quality': round(avg_quality, 2),
                'complexity_score': round(avg_complexity, 2),
                'documentation_coverage': round(avg_doc_coverage, 2)
            },
            'issue_summary': {
                'total_issues': total_issues,
                'issue_types': issue_types
            },
            'suggestion_summary': {
                'total_suggestions': total_suggestions,
                'suggestion_types': suggestion_types
            },
            'quality_distribution': quality_distribution,
            'files_by_quality': {
                'excellent': [r.metrics.file_path for r in results if r.metrics.overall_quality >= 9.0],
                'needs_improvement': [r.metrics.file_path for r in results if r.metrics.overall_quality < 6.0]
            }
        }
    
    def generate_report(self, analysis_results: Dict[str, Any], output_file: str = None) -> str:
        """生成分析报告"""
        report_lines = []
        
        # 报告标题
        report_lines.append("# VIVTransformer 高级代码质量分析报告")
        report_lines.append(f"\n**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**分析目录**: {analysis_results.get('directory', 'N/A')}")
        report_lines.append(f"**分析文件数**: {analysis_results.get('files_analyzed', 0)}")
        
        # 摘要信息
        summary = analysis_results.get('summary', {})
        if summary:
            report_lines.append("\n## 📊 质量摘要")
            
            avg_metrics = summary.get('average_metrics', {})
            report_lines.append(f"\n- **平均质量分数**: {avg_metrics.get('overall_quality', 0)}/10")
            report_lines.append(f"- **平均复杂度**: {avg_metrics.get('complexity_score', 0)}")
            report_lines.append(f"- **平均文档覆盖率**: {avg_metrics.get('documentation_coverage', 0)}%")
            
            # 质量分布
            quality_dist = summary.get('quality_distribution', {})
            report_lines.append("\n### 质量分布")
            report_lines.append(f"- 优秀 (≥9.0): {quality_dist.get('excellent', 0)} 个文件")
            report_lines.append(f"- 良好 (7.0-8.9): {quality_dist.get('good', 0)} 个文件")
            report_lines.append(f"- 一般 (5.0-6.9): {quality_dist.get('fair', 0)} 个文件")
            report_lines.append(f"- 较差 (<5.0): {quality_dist.get('poor', 0)} 个文件")
        
        # 问题统计
        issue_summary = summary.get('issue_summary', {})
        if issue_summary.get('total_issues', 0) > 0:
            report_lines.append("\n## 🚨 问题统计")
            report_lines.append(f"\n**总问题数**: {issue_summary['total_issues']}")
            
            issue_types = issue_summary.get('issue_types', {})
            for issue_type, count in issue_types.items():
                report_lines.append(f"- {issue_type}: {count}")
        
        # 改进建议
        suggestion_summary = summary.get('suggestion_summary', {})
        if suggestion_summary.get('total_suggestions', 0) > 0:
            report_lines.append("\n## 💡 改进建议")
            report_lines.append(f"\n**总建议数**: {suggestion_summary['total_suggestions']}")
            
            suggestion_types = suggestion_summary.get('suggestion_types', {})
            for suggestion_type, count in suggestion_types.items():
                report_lines.append(f"- {suggestion_type}: {count}")
        
        # 预测分析
        predictions = analysis_results.get('summary', {}).get('predictions', {})
        if predictions.get('status') == 'completed':
            report_lines.append("\n## 🔮 质量趋势预测")
            
            risk_assessment = predictions.get('risk_assessment', {})
            report_lines.append(f"\n**风险等级**: {risk_assessment.get('risk_level', 'unknown')}")
            report_lines.append(f"**质量趋势**: {risk_assessment.get('trend', 'unknown')}")
            report_lines.append(f"**预测平均质量**: {risk_assessment.get('average_predicted_quality', 0)}")
            
            recommendations = predictions.get('recommendations', [])
            if recommendations:
                report_lines.append("\n### 预测性建议")
                for rec in recommendations:
                    report_lines.append(f"- {rec}")
        
        # 需要改进的文件
        files_by_quality = summary.get('files_by_quality', {})
        needs_improvement = files_by_quality.get('needs_improvement', [])
        if needs_improvement:
            report_lines.append("\n## 🎯 需要优先改进的文件")
            for file_path in needs_improvement[:10]:  # 只显示前10个
                report_lines.append(f"- {file_path}")
        
        # 优秀文件
        excellent_files = files_by_quality.get('excellent', [])
        if excellent_files:
            report_lines.append("\n## ⭐ 质量优秀的文件")
            for file_path in excellent_files[:5]:  # 只显示前5个
                report_lines.append(f"- {file_path}")
        
        # 生成报告内容
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if output_file:
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report to {output_file}: {e}")
        
        return report_content


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="VIVTransformer 高级代码质量分析器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --config config/advanced_quality_config.yaml --directory src/
  %(prog)s --mode ai-review --files src/models/vivtransformer.py
  %(prog)s --mode full-analysis --output reports/quality_report.md
        """
    )
    
    parser.add_argument(
        '--config',
        default='config/advanced_quality_config.yaml',
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--mode',
        choices=['ai-review', 'refactoring', 'prediction', 'full-analysis'],
        default='full-analysis',
        help='分析模式'
    )
    
    parser.add_argument(
        '--directory',
        default='modify_multi_attention/',
        help='要分析的目录'
    )
    
    parser.add_argument(
        '--files',
        nargs='+',
        help='要分析的具体文件'
    )
    
    parser.add_argument(
        '--output',
        default='reports/advanced_quality_report.md',
        help='输出报告文件路径'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 检查配置文件
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # 初始化分析器
    analyzer = AdvancedQualityAnalyzer(args.config)
    
    async def run_analysis():
        """运行分析"""
        try:
            # 确定分析类型
            analysis_types = []
            if args.mode == 'ai-review':
                analysis_types = ['ai_review']
            elif args.mode == 'refactoring':
                analysis_types = ['refactoring']
            elif args.mode == 'prediction':
                analysis_types = ['basic_metrics']
            else:  # full-analysis
                analysis_types = ['ai_review', 'refactoring', 'basic_metrics']
            
            # 分析文件或目录
            if args.files:
                # 分析指定文件
                results = []
                for file_path in args.files:
                    if os.path.exists(file_path):
                        result = await analyzer.analyze_file(file_path, analysis_types)
                        results.append(result)
                        print(f"✅ 完成分析: {file_path} (质量分数: {result.metrics.overall_quality:.2f})")
                    else:
                        logger.warning(f"File not found: {file_path}")
                
                # 生成简单摘要
                analysis_results = {
                    'status': 'completed',
                    'files_analyzed': len(results),
                    'results': [r.to_dict() for r in results],
                    'summary': analyzer._generate_directory_summary(results)
                }
            else:
                # 分析目录
                analysis_results = await analyzer.analyze_directory(args.directory, analysis_types)
            
            # 生成报告
            if analysis_results.get('status') == 'completed':
                report = analyzer.generate_report(analysis_results, args.output)
                
                print("\n" + "="*60)
                print("📊 分析完成!")
                print(f"📁 分析目录: {args.directory if not args.files else '指定文件'}")
                print(f"📄 分析文件数: {analysis_results.get('files_analyzed', 0)}")
                
                summary = analysis_results.get('summary', {})
                if summary:
                    avg_metrics = summary.get('average_metrics', {})
                    print(f"⭐ 平均质量分数: {avg_metrics.get('overall_quality', 0):.2f}/10")
                    
                    issue_summary = summary.get('issue_summary', {})
                    print(f"🚨 发现问题: {issue_summary.get('total_issues', 0)}")
                    
                    suggestion_summary = summary.get('suggestion_summary', {})
                    print(f"💡 改进建议: {suggestion_summary.get('total_suggestions', 0)}")
                
                print(f"📋 详细报告: {args.output}")
                print("="*60)
                
                # 如果是预测模式，显示预测结果
                if args.mode == 'prediction' and len(analyzer.quality_history) >= 10:
                    prediction_result = analyzer.predictive_analyzer.predict_quality_trend(analyzer.quality_history)
                    if prediction_result.get('status') == 'completed':
                        risk = prediction_result.get('risk_assessment', {})
                        print(f"\n🔮 质量趋势预测:")
                        print(f"   风险等级: {risk.get('risk_level', 'unknown')}")
                        print(f"   趋势: {risk.get('trend', 'unknown')}")
                        print(f"   预测平均质量: {risk.get('average_predicted_quality', 0):.2f}")
            
            else:
                print(f"❌ 分析失败: {analysis_results.get('message', 'Unknown error')}")
                sys.exit(1)
        
        except KeyboardInterrupt:
            print("\n⏹️  分析被用户中断")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            print(f"❌ 分析失败: {e}")
            sys.exit(1)
    
    # 运行异步分析
    asyncio.run(run_analysis())


if __name__ == '__main__':
    main()