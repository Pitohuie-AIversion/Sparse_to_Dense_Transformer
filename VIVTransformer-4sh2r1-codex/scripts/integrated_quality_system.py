#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIVTransformer 集成质量系统

功能:
- 统一的质量分析入口
- 多维度质量评估
- 智能化质量建议
- 质量趋势分析
- 自动化质量报告
- 质量门禁集成

作者: VIVTransformer Team
日期: 2024
"""

import os
import sys
import json
import yaml
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QualityDimension:
    """质量维度评估结果"""
    name: str
    score: float
    status: str
    issues_count: int
    critical_issues: int
    recommendations: List[str]
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntegratedQualityReport:
    """集成质量报告"""
    timestamp: str
    project_path: str
    overall_score: float
    overall_status: str
    dimensions: List[QualityDimension]
    summary: Dict[str, Any]
    trends: Dict[str, Any]
    action_plan: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QualitySystemConfig:
    """质量系统配置管理"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or 'advanced_quality_config.yaml'
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}")
        
        # 返回默认配置
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'quality_dimensions': {
                'code_quality': {
                    'enabled': True,
                    'weight': 0.25,
                    'tools': ['pylint', 'flake8', 'mypy']
                },
                'architecture_quality': {
                    'enabled': True,
                    'weight': 0.20,
                    'tools': ['architecture_validator']
                },
                'performance': {
                    'enabled': True,
                    'weight': 0.15,
                    'tools': ['performance_monitor']
                },
                'security': {
                    'enabled': True,
                    'weight': 0.15,
                    'tools': ['bandit', 'safety']
                },
                'team_collaboration': {
                    'enabled': True,
                    'weight': 0.15,
                    'tools': ['collaboration_analyzer']
                },
                'documentation': {
                    'enabled': True,
                    'weight': 0.10,
                    'tools': ['doc_analyzer']
                }
            },
            'thresholds': {
                'excellent': 90,
                'good': 75,
                'acceptable': 60,
                'poor': 40
            },
            'reporting': {
                'formats': ['markdown', 'json', 'html'],
                'include_trends': True,
                'include_recommendations': True
            }
        }
    
    def get_dimension_config(self, dimension: str) -> Dict[str, Any]:
        """获取指定维度的配置"""
        return self.config.get('quality_dimensions', {}).get(dimension, {})
    
    def is_dimension_enabled(self, dimension: str) -> bool:
        """检查维度是否启用"""
        return self.get_dimension_config(dimension).get('enabled', False)
    
    def get_dimension_weight(self, dimension: str) -> float:
        """获取维度权重"""
        return self.get_dimension_config(dimension).get('weight', 0.0)


class QualityAnalysisEngine:
    """质量分析引擎"""
    
    def __init__(self, project_path: str, config: QualitySystemConfig):
        self.project_path = project_path
        self.config = config
        self.scripts_dir = Path(__file__).parent
    
    def analyze_code_quality(self) -> QualityDimension:
        """分析代码质量"""
        logger.info("分析代码质量...")
        
        try:
            script_path = Path(self.project_path) / 'scripts' / 'code_quality.py'
            result = subprocess.run([
                sys.executable, str(script_path),
                '--format', 'json'
            ], capture_output=True, text=True, timeout=300, cwd=self.project_path)
            
            logger.debug(f"代码质量检查返回码: {result.returncode}")
            logger.debug(f"代码质量检查stdout长度: {len(result.stdout) if result.stdout else 0}")
            logger.debug(f"代码质量检查stderr: {result.stderr[:200] if result.stderr else 'None'}")
            
            if result.returncode in [0, 1]:  # 允许退出码1，因为有质量问题时会返回1
                if result.stdout:
                    try:
                        data = json.loads(result.stdout)
                        logger.debug(f"解析的数据结构: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                        
                        # 计算质量分数
                        score = self._calculate_code_quality_score(data)
                        status = self._get_status_from_score(score)
                        logger.debug(f"计算的质量分数: {score}")
                    except json.JSONDecodeError as e:
                        logger.error(f"无法解析代码质量检查结果: {e}")
                        logger.error(f"原始输出: {result.stdout[:500]}")
                        return self._create_error_dimension("Code Quality")
                else:
                    logger.error(f"代码质量检查返回空输出: 返回码 {result.returncode}")
                    logger.error(f"错误输出: {result.stderr}")
                    return self._create_error_dimension("Code Quality")
                
                summary = data.get('summary', {})
                severity_counts = summary.get('severity_counts', {})
                
                # 生成建议
                recommendations = []
                if severity_counts.get('critical', 0) > 0:
                    recommendations.append("立即修复严重问题")
                if severity_counts.get('error', 0) > 0:
                    recommendations.append("修复错误级别问题")
                if severity_counts.get('warning', 0) > 10:
                    recommendations.append("减少警告数量")
                if not summary.get('overall_success', True):
                    recommendations.append("检查工具配置")
                recommendations.append("运行代码格式化工具")
                
                return QualityDimension(
                        name="Code Quality",
                        score=score,
                        status=status,
                        issues_count=summary.get('total_issues', 0),
                        critical_issues=severity_counts.get('critical', 0),
                        recommendations=recommendations[:5],
                        details=data
                    )
            else:
                logger.error(f"代码质量检查失败: 返回码 {result.returncode}")
                logger.error(f"错误输出: {result.stderr}")
                if result.stderr and "Traceback" in result.stderr:
                    logger.error("代码质量脚本执行时出现异常")
                return self._create_error_dimension("Code Quality")
            
        except Exception as e:
            logger.error(f"代码质量分析失败: {e}")
        
        return self._create_error_dimension("Code Quality")
    
    def analyze_architecture_quality(self) -> QualityDimension:
        """分析架构质量"""
        logger.info("分析架构质量...")
        
        try:
            result = subprocess.run([
                sys.executable,
                str(self.scripts_dir / 'architecture_quality_validator.py'),
                '--check-architecture', self.project_path,
                '--format', 'json',
                '--output', 'temp_architecture.json'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists('temp_architecture.json'):
                with open('temp_architecture.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                health = data.get('architecture_health', {})
                score = health.get('overall_score', 0)
                status = self._get_status_from_score(score)
                
                violations = data.get('violations', [])
                critical_violations = [v for v in violations if v.get('severity') == 'high']
                
                os.remove('temp_architecture.json')
                
                return QualityDimension(
                    name="Architecture Quality",
                    score=score,
                    status=status,
                    issues_count=len(violations),
                    critical_issues=len(critical_violations),
                    recommendations=health.get('recommendations', [])[:5],
                    details=data
                )
            
        except Exception as e:
            logger.error(f"架构质量分析失败: {e}")
        
        return self._create_error_dimension("Architecture Quality")
    
    def analyze_performance(self) -> QualityDimension:
        """分析性能"""
        logger.info("分析性能...")
        
        try:
            result = subprocess.run([
                sys.executable,
                str(self.scripts_dir / 'performance_monitor.py'),
                '--analyze', self.project_path,
                '--format', 'json',
                '--output', 'temp_performance.json'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists('temp_performance.json'):
                with open('temp_performance.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 基于性能指标计算分数
                score = self._calculate_performance_score(data)
                status = self._get_status_from_score(score)
                
                bottlenecks = data.get('performance_bottlenecks', [])
                critical_bottlenecks = [b for b in bottlenecks if b.get('severity') in ['critical', 'high']]
                
                os.remove('temp_performance.json')
                
                return QualityDimension(
                    name="Performance",
                    score=score,
                    status=status,
                    issues_count=len(bottlenecks),
                    critical_issues=len(critical_bottlenecks),
                    recommendations=data.get('recommendations', [])[:5],
                    details=data
                )
            
        except Exception as e:
            logger.error(f"性能分析失败: {e}")
        
        return self._create_error_dimension("Performance")
    
    def analyze_security(self) -> QualityDimension:
        """分析安全性"""
        logger.info("分析安全性...")
        
        try:
            # 运行安全检查
            security_issues = []
            
            # Bandit安全扫描
            try:
                result = subprocess.run([
                    'bandit', '-r', self.project_path, '-f', 'json'
                ], capture_output=True, text=True, timeout=180)
                
                if result.stdout:
                    bandit_data = json.loads(result.stdout)
                    security_issues.extend(bandit_data.get('results', []))
            except Exception as e:
                logger.warning(f"Bandit扫描失败: {e}")
            
            # 计算安全分数
            critical_issues = [i for i in security_issues if i.get('issue_severity') == 'HIGH']
            total_issues = len(security_issues)
            
            if total_issues == 0:
                score = 100.0
            else:
                score = max(0, 100 - (len(critical_issues) * 20 + (total_issues - len(critical_issues)) * 5))
            
            status = self._get_status_from_score(score)
            
            recommendations = []
            if critical_issues:
                recommendations.append("修复高危安全漏洞")
            if total_issues > 10:
                recommendations.append("建立安全代码审查流程")
            recommendations.append("定期进行安全扫描")
            
            return QualityDimension(
                name="Security",
                score=score,
                status=status,
                issues_count=total_issues,
                critical_issues=len(critical_issues),
                recommendations=recommendations,
                details={'security_issues': security_issues}
            )
            
        except Exception as e:
            logger.error(f"安全分析失败: {e}")
        
        return self._create_error_dimension("Security")
    
    def analyze_team_collaboration(self) -> QualityDimension:
        """分析团队协作"""
        logger.info("分析团队协作...")
        
        try:
            result = subprocess.run([
                sys.executable,
                str(self.scripts_dir / 'team_collaboration_analyzer.py'),
                '--analyze', self.project_path,
                '--format', 'json',
                '--output', 'temp_collaboration.json'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists('temp_collaboration.json'):
                with open('temp_collaboration.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                metrics = data.get('collaboration_metrics', {})
                score = self._calculate_collaboration_score(metrics)
                status = self._get_status_from_score(score)
                
                knowledge_gaps = data.get('knowledge_gaps', [])
                critical_gaps = [g for g in knowledge_gaps if g.get('risk_level') == 'critical']
                
                os.remove('temp_collaboration.json')
                
                return QualityDimension(
                    name="Team Collaboration",
                    score=score,
                    status=status,
                    issues_count=len(knowledge_gaps),
                    critical_issues=len(critical_gaps),
                    recommendations=data.get('recommendations', [])[:5],
                    details=data
                )
            
        except Exception as e:
            logger.error(f"团队协作分析失败: {e}")
        
        return self._create_error_dimension("Team Collaboration")
    
    def analyze_documentation(self) -> QualityDimension:
        """分析文档质量"""
        logger.info("分析文档质量...")
        
        try:
            # 简化的文档分析
            python_files = list(Path(self.project_path).rglob('*.py'))
            doc_files = list(Path(self.project_path).rglob('*.md')) + \
                       list(Path(self.project_path).rglob('*.rst'))
            
            total_functions = 0
            documented_functions = 0
            
            for file_path in python_files[:20]:  # 限制分析文件数量
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    import ast
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                            if ast.get_docstring(node):
                                documented_functions += 1
                
                except Exception:
                    continue
            
            coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 0
            score = min(100, coverage + (len(doc_files) * 5))  # 文档文件加分
            status = self._get_status_from_score(score)
            
            issues = max(0, total_functions - documented_functions)
            
            recommendations = []
            if coverage < 70:
                recommendations.append("提高函数文档覆盖率")
            if len(doc_files) < 3:
                recommendations.append("增加项目文档")
            recommendations.append("建立文档维护流程")
            
            return QualityDimension(
                name="Documentation",
                score=score,
                status=status,
                issues_count=issues,
                critical_issues=0,
                recommendations=recommendations,
                details={
                    'function_coverage': coverage,
                    'total_functions': total_functions,
                    'documented_functions': documented_functions,
                    'doc_files_count': len(doc_files)
                }
            )
            
        except Exception as e:
            logger.error(f"文档分析失败: {e}")
        
        return self._create_error_dimension("Documentation")
    
    def _calculate_code_quality_score(self, data: Dict[str, Any]) -> float:
        """计算代码质量分数"""
        summary = data.get('summary', {})
        total_issues = summary.get('total_issues', 0)
        severity_counts = summary.get('severity_counts', {})
        
        critical_issues = severity_counts.get('critical', 0)
        error_issues = severity_counts.get('error', 0)
        warning_issues = severity_counts.get('warning', 0)
        
        if total_issues == 0:
            return 100.0
        
        # 基于问题数量和严重程度计算分数
        score = 100 - (critical_issues * 20 + error_issues * 10 + warning_issues * 5)
        return max(0.0, min(100.0, score))
    
    def _calculate_performance_score(self, data: Dict[str, Any]) -> float:
        """计算性能分数"""
        bottlenecks = data.get('performance_bottlenecks', [])
        critical_bottlenecks = [b for b in bottlenecks if b.get('severity') in ['critical', 'high']]
        
        if not bottlenecks:
            return 100.0
        
        score = 100 - (len(critical_bottlenecks) * 20 + (len(bottlenecks) - len(critical_bottlenecks)) * 5)
        return max(0.0, min(100.0, score))
    
    def _calculate_collaboration_score(self, metrics: Dict[str, Any]) -> float:
        """计算协作分数"""
        collaboration_index = metrics.get('collaboration_index', 0)
        knowledge_distribution = metrics.get('knowledge_distribution', 0)
        bus_factor = metrics.get('bus_factor', 0)
        
        # 综合计算协作分数
        score = (collaboration_index + knowledge_distribution) / 2
        
        # Bus factor调整
        if bus_factor > 5:
            score -= 10
        elif bus_factor < 2:
            score -= 20
        
        return max(0.0, min(100.0, score))
    
    def _get_status_from_score(self, score: float) -> str:
        """根据分数获取状态"""
        thresholds = self.config.config.get('thresholds', {})
        
        if score >= thresholds.get('excellent', 90):
            return 'excellent'
        elif score >= thresholds.get('good', 75):
            return 'good'
        elif score >= thresholds.get('acceptable', 60):
            return 'acceptable'
        elif score >= thresholds.get('poor', 40):
            return 'poor'
        else:
            return 'critical'
    
    def _create_error_dimension(self, name: str) -> QualityDimension:
        """创建错误维度"""
        return QualityDimension(
            name=name,
            score=0.0,
            status='error',
            issues_count=0,
            critical_issues=0,
            recommendations=[f"{name} 分析失败，请检查工具配置"],
            details={'error': True}
        )


class IntegratedQualitySystem:
    """集成质量系统"""
    
    def __init__(self, project_path: str, config_path: str = None):
        self.project_path = project_path
        self.config = QualitySystemConfig(config_path)
        self.analysis_engine = QualityAnalysisEngine(project_path, self.config)
    
    def run_comprehensive_analysis(self, parallel: bool = True) -> IntegratedQualityReport:
        """运行综合质量分析"""
        logger.info("开始综合质量分析...")
        start_time = time.time()
        
        dimensions = []
        
        # 定义分析任务
        analysis_tasks = [
            ('code_quality', self.analysis_engine.analyze_code_quality),
            ('architecture_quality', self.analysis_engine.analyze_architecture_quality),
            ('performance', self.analysis_engine.analyze_performance),
            ('security', self.analysis_engine.analyze_security),
            ('team_collaboration', self.analysis_engine.analyze_team_collaboration),
            ('documentation', self.analysis_engine.analyze_documentation)
        ]
        
        if parallel:
            # 并行执行分析
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_task = {}
                
                for dimension_name, analysis_func in analysis_tasks:
                    if self.config.is_dimension_enabled(dimension_name):
                        future = executor.submit(analysis_func)
                        future_to_task[future] = dimension_name
                
                for future in as_completed(future_to_task):
                    dimension_name = future_to_task[future]
                    try:
                        dimension = future.result()
                        dimensions.append(dimension)
                        logger.info(f"{dimension.name} 分析完成: {dimension.score:.2f}")
                    except Exception as e:
                        logger.error(f"{dimension_name} 分析失败: {e}")
                        dimensions.append(self.analysis_engine._create_error_dimension(dimension_name))
        else:
            # 串行执行分析
            for dimension_name, analysis_func in analysis_tasks:
                if self.config.is_dimension_enabled(dimension_name):
                    try:
                        dimension = analysis_func()
                        dimensions.append(dimension)
                        logger.info(f"{dimension.name} 分析完成: {dimension.score:.2f}")
                    except Exception as e:
                        logger.error(f"{dimension_name} 分析失败: {e}")
                        dimensions.append(self.analysis_engine._create_error_dimension(dimension_name))
        
        # 计算总体分数
        overall_score = self._calculate_overall_score(dimensions)
        overall_status = self.analysis_engine._get_status_from_score(overall_score)
        
        # 生成摘要
        summary = self._generate_summary(dimensions)
        
        # 生成行动计划
        action_plan = self._generate_action_plan(dimensions)
        
        analysis_time = time.time() - start_time
        logger.info(f"综合质量分析完成，耗时 {analysis_time:.2f} 秒")
        
        return IntegratedQualityReport(
            timestamp=datetime.now().isoformat(),
            project_path=self.project_path,
            overall_score=overall_score,
            overall_status=overall_status,
            dimensions=dimensions,
            summary=summary,
            trends={},  # 趋势分析需要历史数据
            action_plan=action_plan
        )
    
    def _calculate_overall_score(self, dimensions: List[QualityDimension]) -> float:
        """计算总体质量分数"""
        if not dimensions:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dimension in dimensions:
            # 从维度名称推断配置键
            dimension_key = dimension.name.lower().replace(' ', '_')
            weight = self.config.get_dimension_weight(dimension_key)
            
            if weight > 0:
                weighted_sum += dimension.score * weight
                total_weight += weight
        
        if total_weight == 0:
            return sum(d.score for d in dimensions) / len(dimensions)
        
        return weighted_sum / total_weight
    
    def _generate_summary(self, dimensions: List[QualityDimension]) -> Dict[str, Any]:
        """生成分析摘要"""
        total_issues = sum(d.issues_count for d in dimensions)
        total_critical = sum(d.critical_issues for d in dimensions)
        
        status_counts = {}
        for dimension in dimensions:
            status = dimension.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_dimensions': len(dimensions),
            'total_issues': total_issues,
            'critical_issues': total_critical,
            'status_distribution': status_counts,
            'best_dimension': max(dimensions, key=lambda d: d.score).name if dimensions else None,
            'worst_dimension': min(dimensions, key=lambda d: d.score).name if dimensions else None
        }
    
    def _generate_action_plan(self, dimensions: List[QualityDimension]) -> List[str]:
        """生成行动计划"""
        action_plan = []
        
        # 按分数排序，优先处理分数最低的维度
        sorted_dimensions = sorted(dimensions, key=lambda d: d.score)
        
        for dimension in sorted_dimensions:
            if dimension.score < 60:  # 需要改进的维度
                action_plan.append(f"优先改进 {dimension.name} (当前分数: {dimension.score:.1f})")
                
                # 添加具体建议
                for rec in dimension.recommendations[:2]:
                    action_plan.append(f"  - {rec}")
        
        # 添加通用建议
        if not action_plan:
            action_plan.append("质量状况良好，继续保持")
        
        action_plan.append("建立持续质量监控机制")
        action_plan.append("定期进行质量评估和改进")
        
        return action_plan
    
    def generate_integrated_report(self, report: IntegratedQualityReport) -> str:
        """生成集成质量报告"""
        lines = []
        
        # 报告标题
        lines.append("# VIVTransformer 集成质量分析报告")
        lines.append(f"\n**生成时间**: {report.timestamp}")
        lines.append(f"**项目路径**: {report.project_path}")
        
        # 总体评估
        lines.append("\n## 🎯 总体质量评估")
        lines.append(f"\n**总体分数**: {report.overall_score:.2f}/100")
        
        status_emoji = {
            'excellent': '🟢',
            'good': '🟡',
            'acceptable': '🟠',
            'poor': '🔴',
            'critical': '⚫'
        }
        emoji = status_emoji.get(report.overall_status, '⚪')
        lines.append(f"**质量状态**: {emoji} {report.overall_status.title()}")
        
        # 维度分析
        lines.append("\n## 📊 质量维度分析")
        
        for dimension in sorted(report.dimensions, key=lambda d: d.score, reverse=True):
            emoji = status_emoji.get(dimension.status, '⚪')
            lines.append(f"\n### {emoji} {dimension.name}")
            lines.append(f"- **分数**: {dimension.score:.2f}/100")
            lines.append(f"- **状态**: {dimension.status.title()}")
            lines.append(f"- **问题数量**: {dimension.issues_count}")
            
            if dimension.critical_issues > 0:
                lines.append(f"- **严重问题**: {dimension.critical_issues}")
            
            if dimension.recommendations:
                lines.append("- **建议**:")
                for rec in dimension.recommendations[:3]:
                    lines.append(f"  - {rec}")
        
        # 问题摘要
        summary = report.summary
        lines.append("\n## ⚠️ 问题摘要")
        lines.append(f"\n**总问题数**: {summary['total_issues']}")
        lines.append(f"**严重问题**: {summary['critical_issues']}")
        
        if summary['status_distribution']:
            lines.append("\n**状态分布**:")
            for status, count in summary['status_distribution'].items():
                emoji = status_emoji.get(status, '⚪')
                lines.append(f"- {emoji} {status.title()}: {count}")
        
        # 行动计划
        lines.append("\n## 🎯 行动计划")
        
        for i, action in enumerate(report.action_plan, 1):
            if action.startswith('  -'):
                lines.append(action)
            else:
                lines.append(f"\n{i}. {action}")
        
        # 质量趋势（如果有历史数据）
        if report.trends:
            lines.append("\n## 📈 质量趋势")
            lines.append("\n*趋势分析需要历史数据支持*")
        
        # 下一步建议
        lines.append("\n## 💡 下一步建议")
        
        if summary['critical_issues'] > 0:
            lines.append("\n1. **立即行动**: 修复所有严重问题")
        
        if report.overall_score < 70:
            lines.append("\n2. **质量改进**: 重点提升低分维度")
        
        lines.append("\n3. **持续监控**: 建立自动化质量检查")
        lines.append("\n4. **团队培训**: 提升团队质量意识")
        lines.append("\n5. **流程优化**: 完善开发和审查流程")
        
        return "\n".join(lines)
    
    def save_report(self, report: IntegratedQualityReport, output_path: str, format_type: str = 'markdown'):
        """保存报告"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format_type == 'json':
            content = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        elif format_type == 'markdown':
            content = self.generate_integrated_report(report)
        else:
            raise ValueError(f"不支持的格式: {format_type}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"报告已保存: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="VIVTransformer 集成质量系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --analyze .
  %(prog)s --config custom_config.yaml
  %(prog)s --parallel --output reports/quality_report.md
        """
    )
    
    parser.add_argument(
        '--analyze',
        metavar='PATH',
        default='.',
        help='分析指定项目路径'
    )
    
    parser.add_argument(
        '--config',
        metavar='CONFIG',
        help='质量系统配置文件路径'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='并行执行分析任务'
    )
    
    parser.add_argument(
        '--output',
        default='reports/integrated_quality_report.md',
        help='输出报告文件路径'
    )
    
    parser.add_argument(
        '--format',
        choices=['markdown', 'json'],
        default='markdown',
        help='输出格式'
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
    
    try:
        if not os.path.exists(args.analyze):
            print(f"[ERROR] 路径不存在: {args.analyze}")
            sys.exit(1)
        
        print(f"[INFO] 启动集成质量分析: {args.analyze}")
        
        # 初始化质量系统
        quality_system = IntegratedQualitySystem(args.analyze, args.config)
        
        # 运行综合分析
        report = quality_system.run_comprehensive_analysis(parallel=args.parallel)
        
        # 显示结果摘要
        print("\n" + "="*60)
        print("[RESULT] 集成质量分析结果")
        print(f"[SCORE] 总体分数: {report.overall_score:.2f}/100")
        print(f"[STATUS] 质量状态: {report.overall_status.title()}")
        print(f"[DIMS] 分析维度: {len(report.dimensions)}")
        print(f"[ISSUES] 总问题数: {report.summary['total_issues']}")
        print(f"[CRITICAL] 严重问题: {report.summary['critical_issues']}")
        
        # 显示各维度分数
        print("\n[DIMENSIONS] 维度分数:")
        for dimension in sorted(report.dimensions, key=lambda d: d.score, reverse=True):
            status_map = {
                'excellent': '[EXCELLENT]', 'good': '[GOOD]', 'acceptable': '[OK]', 
                'poor': '[POOR]', 'critical': '[CRITICAL]', 'error': '[ERROR]'
            }
            status = status_map.get(dimension.status, '[UNKNOWN]')
            print(f"   {status} {dimension.name}: {dimension.score:.1f}")
        
        # 保存报告
        quality_system.save_report(report, args.output, args.format)
        print(f"\n[REPORT] 详细报告已保存: {args.output}")
        
        # 显示关键行动项
        if report.action_plan:
            print("\n[ACTION] 关键行动项:")
            for action in report.action_plan[:5]:
                if not action.startswith('  -'):
                    print(f"   - {action}")
        
        print("="*60)
        
        # 根据质量状态设置退出码
        if report.overall_status in ['critical', 'poor']:
            sys.exit(1)
        elif report.overall_status == 'acceptable':
            sys.exit(2)
        else:
            sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n[INFO] 操作被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"集成质量分析失败: {e}")
        print(f"[ERROR] 集成质量分析失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()