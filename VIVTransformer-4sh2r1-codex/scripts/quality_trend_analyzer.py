#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIVTransformer 质量趋势分析器

功能:
- 质量历史数据收集
- 趋势分析和预测
- 质量回归检测
- 改进效果评估
- 质量报告生成
- 可视化图表生成

作者: VIVTransformer Team
日期: 2024
"""

import os
import sys
import json
import sqlite3
import logging
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QualitySnapshot:
    """质量快照"""
    timestamp: str
    commit_hash: str
    branch: str
    overall_score: float
    dimensions: Dict[str, float]
    metrics: Dict[str, Any]
    issues_count: int
    critical_issues: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrendAnalysis:
    """趋势分析结果"""
    metric_name: str
    trend_direction: str  # 'improving', 'declining', 'stable'
    trend_strength: float  # 0-1, 1表示强趋势
    current_value: float
    average_value: float
    change_rate: float  # 变化率
    prediction: Optional[float]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityTrendReport:
    """质量趋势报告"""
    timestamp: str
    project_path: str
    analysis_period: str
    snapshots_count: int
    overall_trend: TrendAnalysis
    dimension_trends: List[TrendAnalysis]
    quality_events: List[Dict[str, Any]]
    recommendations: List[str]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QualityDatabase:
    """质量数据库管理"""
    
    def __init__(self, db_path: str = 'quality_history.db'):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建质量快照表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                commit_hash TEXT,
                branch TEXT,
                overall_score REAL,
                dimensions TEXT,  -- JSON格式
                metrics TEXT,     -- JSON格式
                issues_count INTEGER,
                critical_issues INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建质量事件表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT,  -- 'improvement', 'regression', 'milestone'
                description TEXT,
                impact_score REAL,
                related_commit TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON quality_snapshots(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_branch ON quality_snapshots(branch)')
        
        conn.commit()
        conn.close()
    
    def save_snapshot(self, snapshot: QualitySnapshot):
        """保存质量快照"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quality_snapshots 
            (timestamp, commit_hash, branch, overall_score, dimensions, metrics, issues_count, critical_issues)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            snapshot.timestamp,
            snapshot.commit_hash,
            snapshot.branch,
            snapshot.overall_score,
            json.dumps(snapshot.dimensions),
            json.dumps(snapshot.metrics),
            snapshot.issues_count,
            snapshot.critical_issues
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"质量快照已保存: {snapshot.timestamp}")
    
    def get_snapshots(self, 
                     days: int = 30, 
                     branch: str = None,
                     limit: int = None) -> List[QualitySnapshot]:
        """获取质量快照"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构建查询
        query = '''
            SELECT timestamp, commit_hash, branch, overall_score, dimensions, metrics, issues_count, critical_issues
            FROM quality_snapshots
            WHERE timestamp >= ?
        '''
        params = [datetime.now().isoformat()[:10]]  # 最近N天
        
        if branch:
            query += ' AND branch = ?'
            params.append(branch)
        
        query += ' ORDER BY timestamp DESC'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        snapshots = []
        for row in rows:
            snapshot = QualitySnapshot(
                timestamp=row[0],
                commit_hash=row[1] or '',
                branch=row[2] or '',
                overall_score=row[3] or 0.0,
                dimensions=json.loads(row[4]) if row[4] else {},
                metrics=json.loads(row[5]) if row[5] else {},
                issues_count=row[6] or 0,
                critical_issues=row[7] or 0
            )
            snapshots.append(snapshot)
        
        conn.close()
        return snapshots
    
    def save_event(self, event_type: str, description: str, impact_score: float = 0.0, commit_hash: str = None):
        """保存质量事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quality_events (timestamp, event_type, description, impact_score, related_commit)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            event_type,
            description,
            impact_score,
            commit_hash
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"质量事件已记录: {event_type} - {description}")
    
    def get_events(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取质量事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT timestamp, event_type, description, impact_score, related_commit
            FROM quality_events
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        ''', (since_date,))
        
        events = []
        for row in cursor.fetchall():
            events.append({
                'timestamp': row[0],
                'event_type': row[1],
                'description': row[2],
                'impact_score': row[3],
                'related_commit': row[4]
            })
        
        conn.close()
        return events


class TrendAnalyzer:
    """趋势分析器"""
    
    def __init__(self):
        pass
    
    def analyze_metric_trend(self, values: List[float], timestamps: List[str], metric_name: str) -> TrendAnalysis:
        """分析指标趋势"""
        if len(values) < 2:
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction='stable',
                trend_strength=0.0,
                current_value=values[0] if values else 0.0,
                average_value=values[0] if values else 0.0,
                change_rate=0.0,
                prediction=None,
                confidence=0.0
            )
        
        # 计算基本统计
        current_value = values[0]  # 最新值
        average_value = statistics.mean(values)
        
        # 计算趋势方向和强度
        trend_direction, trend_strength = self._calculate_trend(values)
        
        # 计算变化率
        if len(values) >= 2:
            change_rate = (values[0] - values[-1]) / values[-1] * 100 if values[-1] != 0 else 0.0
        else:
            change_rate = 0.0
        
        # 简单预测（线性回归）
        prediction, confidence = self._predict_next_value(values)
        
        return TrendAnalysis(
            metric_name=metric_name,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            current_value=current_value,
            average_value=average_value,
            change_rate=change_rate,
            prediction=prediction,
            confidence=confidence
        )
    
    def _calculate_trend(self, values: List[float]) -> Tuple[str, float]:
        """计算趋势方向和强度"""
        if len(values) < 3:
            return 'stable', 0.0
        
        # 计算移动平均
        window_size = min(5, len(values) // 2)
        recent_avg = statistics.mean(values[:window_size])
        older_avg = statistics.mean(values[-window_size:])
        
        # 计算趋势
        diff = recent_avg - older_avg
        max_value = max(values)
        min_value = min(values)
        value_range = max_value - min_value
        
        if value_range == 0:
            return 'stable', 0.0
        
        # 标准化差异
        normalized_diff = abs(diff) / value_range
        
        # 确定方向
        if diff > value_range * 0.05:  # 5%阈值
            direction = 'improving'
        elif diff < -value_range * 0.05:
            direction = 'declining'
        else:
            direction = 'stable'
        
        # 计算强度（0-1）
        strength = min(1.0, normalized_diff * 2)
        
        return direction, strength
    
    def _predict_next_value(self, values: List[float]) -> Tuple[Optional[float], float]:
        """预测下一个值"""
        if len(values) < 3:
            return None, 0.0
        
        try:
            # 简单线性回归
            n = len(values)
            x = list(range(n))
            y = values
            
            # 计算斜率和截距
            x_mean = statistics.mean(x)
            y_mean = statistics.mean(y)
            
            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                return None, 0.0
            
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            
            # 预测下一个值
            next_x = n
            prediction = slope * next_x + intercept
            
            # 计算置信度（基于R²）
            y_pred = [slope * x[i] + intercept for i in range(n)]
            ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
            ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            confidence = max(0.0, min(1.0, r_squared))
            
            return prediction, confidence
            
        except Exception as e:
            logger.warning(f"预测计算失败: {e}")
            return None, 0.0
    
    def detect_quality_events(self, snapshots: List[QualitySnapshot]) -> List[Dict[str, Any]]:
        """检测质量事件"""
        events = []
        
        if len(snapshots) < 2:
            return events
        
        # 按时间排序（最新的在前）
        sorted_snapshots = sorted(snapshots, key=lambda s: s.timestamp, reverse=True)
        
        for i in range(len(sorted_snapshots) - 1):
            current = sorted_snapshots[i]
            previous = sorted_snapshots[i + 1]
            
            # 检测显著改进
            score_change = current.overall_score - previous.overall_score
            if score_change >= 10.0:  # 分数提升10分以上
                events.append({
                    'timestamp': current.timestamp,
                    'type': 'improvement',
                    'description': f'质量显著改进: {score_change:.1f}分',
                    'impact': score_change,
                    'commit': current.commit_hash
                })
            
            # 检测质量回归
            elif score_change <= -5.0:  # 分数下降5分以上
                events.append({
                    'timestamp': current.timestamp,
                    'type': 'regression',
                    'description': f'质量回归: {abs(score_change):.1f}分',
                    'impact': score_change,
                    'commit': current.commit_hash
                })
            
            # 检测严重问题增加
            critical_change = current.critical_issues - previous.critical_issues
            if critical_change > 0:
                events.append({
                    'timestamp': current.timestamp,
                    'type': 'critical_issues',
                    'description': f'新增{critical_change}个严重问题',
                    'impact': -critical_change * 5,  # 负面影响
                    'commit': current.commit_hash
                })
            
            # 检测里程碑
            if current.overall_score >= 90 and previous.overall_score < 90:
                events.append({
                    'timestamp': current.timestamp,
                    'type': 'milestone',
                    'description': '达到优秀质量标准（90分+）',
                    'impact': 10,
                    'commit': current.commit_hash
                })
        
        return events


class QualityTrendAnalyzer:
    """质量趋势分析器主类"""
    
    def __init__(self, project_path: str, db_path: str = None):
        self.project_path = project_path
        self.db = QualityDatabase(db_path or 'quality_history.db')
        self.analyzer = TrendAnalyzer()
    
    def collect_current_snapshot(self) -> QualitySnapshot:
        """收集当前质量快照"""
        logger.info("收集当前质量快照...")
        
        try:
            # 运行质量分析
            scripts_dir = Path(__file__).parent
            import subprocess
            
            result = subprocess.run([
                sys.executable,
                str(scripts_dir / 'integrated_quality_system.py'),
                '--analyze', self.project_path,
                '--format', 'json',
                '--output', 'temp_current_snapshot.json'
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and os.path.exists('temp_current_snapshot.json'):
                with open('temp_current_snapshot.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取维度分数
                dimensions = {}
                for dimension in data.get('dimensions', []):
                    dimension_name = dimension['name'].lower().replace(' ', '_')
                    dimensions[dimension_name] = dimension['score']
                
                # 获取Git信息
                commit_hash, branch = self._get_git_info()
                
                snapshot = QualitySnapshot(
                    timestamp=datetime.now().isoformat(),
                    commit_hash=commit_hash,
                    branch=branch,
                    overall_score=data.get('overall_score', 0.0),
                    dimensions=dimensions,
                    metrics=data.get('summary', {}),
                    issues_count=data.get('summary', {}).get('total_issues', 0),
                    critical_issues=data.get('summary', {}).get('critical_issues', 0)
                )
                
                # 清理临时文件
                os.remove('temp_current_snapshot.json')
                
                return snapshot
            
        except Exception as e:
            logger.error(f"收集质量快照失败: {e}")
        
        # 返回空快照
        return QualitySnapshot(
            timestamp=datetime.now().isoformat(),
            commit_hash='',
            branch='',
            overall_score=0.0,
            dimensions={},
            metrics={},
            issues_count=0,
            critical_issues=0
        )
    
    def _get_git_info(self) -> Tuple[str, str]:
        """获取Git信息"""
        try:
            import subprocess
            
            # 获取当前提交哈希
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.project_path
            )
            commit_hash = result.stdout.strip() if result.returncode == 0 else ''
            
            # 获取当前分支
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.project_path
            )
            branch = result.stdout.strip() if result.returncode == 0 else ''
            
            return commit_hash, branch
            
        except Exception as e:
            logger.warning(f"获取Git信息失败: {e}")
            return '', ''
    
    def analyze_trends(self, days: int = 30, branch: str = None) -> QualityTrendReport:
        """分析质量趋势"""
        logger.info(f"分析最近{days}天的质量趋势...")
        
        # 获取历史快照
        snapshots = self.db.get_snapshots(days=days, branch=branch)
        
        if len(snapshots) < 2:
            logger.warning("历史数据不足，无法进行趋势分析")
            return self._create_empty_report()
        
        # 分析总体趋势
        overall_scores = [s.overall_score for s in snapshots]
        timestamps = [s.timestamp for s in snapshots]
        overall_trend = self.analyzer.analyze_metric_trend(overall_scores, timestamps, 'overall_score')
        
        # 分析各维度趋势
        dimension_trends = []
        all_dimensions = set()
        for snapshot in snapshots:
            all_dimensions.update(snapshot.dimensions.keys())
        
        for dimension in all_dimensions:
            values = []
            for snapshot in snapshots:
                values.append(snapshot.dimensions.get(dimension, 0.0))
            
            if any(v > 0 for v in values):  # 只分析有数据的维度
                trend = self.analyzer.analyze_metric_trend(values, timestamps, dimension)
                dimension_trends.append(trend)
        
        # 检测质量事件
        quality_events = self.analyzer.detect_quality_events(snapshots)
        
        # 获取历史事件
        historical_events = self.db.get_events(days=days)
        quality_events.extend(historical_events)
        
        # 生成建议
        recommendations = self._generate_recommendations(overall_trend, dimension_trends, quality_events)
        
        # 生成摘要
        summary = self._generate_summary(snapshots, overall_trend, dimension_trends)
        
        return QualityTrendReport(
            timestamp=datetime.now().isoformat(),
            project_path=self.project_path,
            analysis_period=f"{days} days",
            snapshots_count=len(snapshots),
            overall_trend=overall_trend,
            dimension_trends=dimension_trends,
            quality_events=quality_events,
            recommendations=recommendations,
            summary=summary
        )
    
    def _create_empty_report(self) -> QualityTrendReport:
        """创建空报告"""
        return QualityTrendReport(
            timestamp=datetime.now().isoformat(),
            project_path=self.project_path,
            analysis_period="insufficient data",
            snapshots_count=0,
            overall_trend=TrendAnalysis(
                metric_name='overall_score',
                trend_direction='stable',
                trend_strength=0.0,
                current_value=0.0,
                average_value=0.0,
                change_rate=0.0,
                prediction=None,
                confidence=0.0
            ),
            dimension_trends=[],
            quality_events=[],
            recommendations=["需要更多历史数据进行趋势分析"],
            summary={'insufficient_data': True}
        )
    
    def _generate_recommendations(self, 
                                overall_trend: TrendAnalysis,
                                dimension_trends: List[TrendAnalysis],
                                events: List[Dict[str, Any]]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于总体趋势的建议
        if overall_trend.trend_direction == 'declining':
            recommendations.append(f"⚠️ 总体质量呈下降趋势（{overall_trend.change_rate:.1f}%），需要立即关注")
            recommendations.append("建议进行全面的代码审查和重构")
        elif overall_trend.trend_direction == 'improving':
            recommendations.append(f"✅ 总体质量持续改进（+{overall_trend.change_rate:.1f}%），保持良好势头")
        
        # 基于维度趋势的建议
        declining_dimensions = [t for t in dimension_trends if t.trend_direction == 'declining']
        if declining_dimensions:
            recommendations.append("🔍 以下维度需要重点关注:")
            for trend in declining_dimensions[:3]:  # 只显示前3个
                recommendations.append(f"  - {trend.metric_name}: {trend.change_rate:.1f}%")
        
        # 基于事件的建议
        recent_regressions = [e for e in events if e.get('type') == 'regression']
        if recent_regressions:
            recommendations.append("🚨 检测到质量回归，建议回滚或修复相关提交")
        
        critical_issues = [e for e in events if e.get('type') == 'critical_issues']
        if critical_issues:
            recommendations.append("🔴 严重问题增加，优先修复高危漏洞")
        
        # 预测性建议
        if overall_trend.prediction and overall_trend.confidence > 0.7:
            if overall_trend.prediction < overall_trend.current_value * 0.9:
                recommendations.append("📉 预测质量可能继续下降，建议采取预防措施")
        
        if not recommendations:
            recommendations.append("📊 质量趋势稳定，继续保持当前开发实践")
        
        return recommendations
    
    def _generate_summary(self, 
                         snapshots: List[QualitySnapshot],
                         overall_trend: TrendAnalysis,
                         dimension_trends: List[TrendAnalysis]) -> Dict[str, Any]:
        """生成摘要"""
        if not snapshots:
            return {'no_data': True}
        
        # 计算统计信息
        scores = [s.overall_score for s in snapshots]
        
        return {
            'analysis_period': len(snapshots),
            'current_score': overall_trend.current_value,
            'average_score': overall_trend.average_value,
            'best_score': max(scores),
            'worst_score': min(scores),
            'score_variance': statistics.variance(scores) if len(scores) > 1 else 0,
            'trend_direction': overall_trend.trend_direction,
            'trend_strength': overall_trend.trend_strength,
            'improving_dimensions': len([t for t in dimension_trends if t.trend_direction == 'improving']),
            'declining_dimensions': len([t for t in dimension_trends if t.trend_direction == 'declining']),
            'stable_dimensions': len([t for t in dimension_trends if t.trend_direction == 'stable'])
        }
    
    def generate_trend_report(self, report: QualityTrendReport) -> str:
        """生成趋势报告"""
        lines = []
        
        # 报告标题
        lines.append("# 📈 质量趋势分析报告")
        lines.append(f"\n**分析时间**: {report.timestamp}")
        lines.append(f"**项目路径**: {report.project_path}")
        lines.append(f"**分析周期**: {report.analysis_period}")
        lines.append(f"**数据点数**: {report.snapshots_count}")
        
        # 总体趋势
        lines.append("\n## 🎯 总体质量趋势")
        
        trend = report.overall_trend
        trend_emoji = {
            'improving': '📈',
            'declining': '📉',
            'stable': '➡️'
        }
        emoji = trend_emoji.get(trend.trend_direction, '➡️')
        
        lines.append(f"\n**当前分数**: {trend.current_value:.2f}/100")
        lines.append(f"**平均分数**: {trend.average_value:.2f}/100")
        lines.append(f"**趋势方向**: {emoji} {trend.trend_direction.title()}")
        lines.append(f"**趋势强度**: {trend.trend_strength:.2f}/1.0")
        lines.append(f"**变化率**: {trend.change_rate:+.1f}%")
        
        if trend.prediction and trend.confidence > 0.5:
            lines.append(f"**预测值**: {trend.prediction:.2f} (置信度: {trend.confidence:.1%})")
        
        # 维度趋势
        if report.dimension_trends:
            lines.append("\n## 📊 维度趋势分析")
            
            # 按趋势方向分组
            improving = [t for t in report.dimension_trends if t.trend_direction == 'improving']
            declining = [t for t in report.dimension_trends if t.trend_direction == 'declining']
            stable = [t for t in report.dimension_trends if t.trend_direction == 'stable']
            
            if improving:
                lines.append("\n### 📈 改进中的维度")
                for trend in sorted(improving, key=lambda t: t.change_rate, reverse=True):
                    lines.append(f"- **{trend.metric_name}**: {trend.current_value:.1f} (+{trend.change_rate:.1f}%)")
            
            if declining:
                lines.append("\n### 📉 下降中的维度")
                for trend in sorted(declining, key=lambda t: t.change_rate):
                    lines.append(f"- **{trend.metric_name}**: {trend.current_value:.1f} ({trend.change_rate:.1f}%)")
            
            if stable:
                lines.append("\n### ➡️ 稳定的维度")
                for trend in stable[:5]:  # 只显示前5个
                    lines.append(f"- **{trend.metric_name}**: {trend.current_value:.1f}")
        
        # 质量事件
        if report.quality_events:
            lines.append("\n## 🎭 质量事件")
            
            # 按类型分组
            events_by_type = {}
            for event in report.quality_events:
                event_type = event.get('type', 'unknown')
                if event_type not in events_by_type:
                    events_by_type[event_type] = []
                events_by_type[event_type].append(event)
            
            for event_type, events in events_by_type.items():
                type_emoji = {
                    'improvement': '✅',
                    'regression': '❌',
                    'milestone': '🏆',
                    'critical_issues': '🚨'
                }
                emoji = type_emoji.get(event_type, '📝')
                
                lines.append(f"\n### {emoji} {event_type.title()}")
                for event in events[:3]:  # 只显示前3个
                    timestamp = event.get('timestamp', '')[:10]  # 只显示日期
                    description = event.get('description', '')
                    lines.append(f"- **{timestamp}**: {description}")
        
        # 统计摘要
        if 'no_data' not in report.summary:
            summary = report.summary
            lines.append("\n## 📋 统计摘要")
            lines.append(f"\n- **最高分数**: {summary.get('best_score', 0):.1f}")
            lines.append(f"- **最低分数**: {summary.get('worst_score', 0):.1f}")
            lines.append(f"- **分数方差**: {summary.get('score_variance', 0):.2f}")
            lines.append(f"- **改进维度**: {summary.get('improving_dimensions', 0)}")
            lines.append(f"- **下降维度**: {summary.get('declining_dimensions', 0)}")
            lines.append(f"- **稳定维度**: {summary.get('stable_dimensions', 0)}")
        
        # 建议和行动项
        if report.recommendations:
            lines.append("\n## 💡 建议和行动项")
            for i, rec in enumerate(report.recommendations, 1):
                if rec.startswith('  -'):
                    lines.append(rec)
                else:
                    lines.append(f"\n{i}. {rec}")
        
        # 下一步
        lines.append("\n## 🚀 下一步")
        
        if report.overall_trend.trend_direction == 'declining':
            lines.append("\n1. **立即行动**: 识别和修复质量下降的根本原因")
            lines.append("2. **代码审查**: 加强代码审查流程")
            lines.append("3. **工具升级**: 考虑升级或增加质量检查工具")
        elif report.overall_trend.trend_direction == 'improving':
            lines.append("\n1. **保持势头**: 继续当前的良好实践")
            lines.append("2. **经验分享**: 总结和分享成功经验")
            lines.append("3. **标准提升**: 考虑提高质量标准")
        else:
            lines.append("\n1. **持续监控**: 保持质量监控和定期分析")
            lines.append("2. **预防措施**: 建立质量预警机制")
            lines.append("3. **流程优化**: 优化开发和测试流程")
        
        return "\n".join(lines)
    
    def save_report(self, report: QualityTrendReport, output_path: str, format_type: str = 'markdown'):
        """保存报告"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format_type == 'json':
            content = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        elif format_type == 'markdown':
            content = self.generate_trend_report(report)
        else:
            raise ValueError(f"不支持的格式: {format_type}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"趋势报告已保存: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="VIVTransformer 质量趋势分析器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --collect .
  %(prog)s --analyze --days 30
  %(prog)s --report --output reports/trend_report.md
        """
    )
    
    parser.add_argument(
        '--collect',
        metavar='PATH',
        help='收集当前质量快照'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='分析质量趋势'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='生成趋势报告'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='分析天数'
    )
    
    parser.add_argument(
        '--branch',
        help='指定分支'
    )
    
    parser.add_argument(
        '--output',
        default='reports/quality_trend_report.md',
        help='输出报告文件路径'
    )
    
    parser.add_argument(
        '--format',
        choices=['markdown', 'json'],
        default='markdown',
        help='输出格式'
    )
    
    parser.add_argument(
        '--db-path',
        help='数据库文件路径'
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
        project_path = args.collect or '.'
        
        if not os.path.exists(project_path):
            print(f"❌ 路径不存在: {project_path}")
            sys.exit(1)
        
        # 初始化趋势分析器
        trend_analyzer = QualityTrendAnalyzer(project_path, args.db_path)
        
        if args.collect:
            print(f"📸 收集质量快照: {project_path}")
            snapshot = trend_analyzer.collect_current_snapshot()
            trend_analyzer.db.save_snapshot(snapshot)
            print(f"✅ 快照已保存: {snapshot.overall_score:.2f}分")
        
        if args.analyze or args.report:
            print(f"📈 分析质量趋势: 最近{args.days}天")
            report = trend_analyzer.analyze_trends(days=args.days, branch=args.branch)
            
            # 显示结果摘要
            print("\n" + "="*60)
            print("📈 质量趋势分析结果")
            print(f"📊 数据点数: {report.snapshots_count}")
            
            if report.snapshots_count > 0:
                trend = report.overall_trend
                trend_emoji = {
                    'improving': '📈',
                    'declining': '📉',
                    'stable': '➡️'
                }
                emoji = trend_emoji.get(trend.trend_direction, '➡️')
                
                print(f"🎯 当前分数: {trend.current_value:.2f}/100")
                print(f"📈 趋势方向: {emoji} {trend.trend_direction.title()}")
                print(f"📊 变化率: {trend.change_rate:+.1f}%")
                print(f"🎭 质量事件: {len(report.quality_events)}")
            else:
                print("⚠️  数据不足，无法进行趋势分析")
            
            if args.report:
                # 保存报告
                trend_analyzer.save_report(report, args.output, args.format)
                print(f"\n📋 趋势报告已保存: {args.output}")
            
            # 显示关键建议
            if report.recommendations:
                print("\n💡 关键建议:")
                for rec in report.recommendations[:3]:
                    if not rec.startswith('  -'):
                        print(f"   • {rec}")
            
            print("="*60)
        
        if not any([args.collect, args.analyze, args.report]):
            print("请指定操作: --collect, --analyze, 或 --report")
            parser.print_help()
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⏹️  操作被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"趋势分析失败: {e}")
        print(f"❌ 趋势分析失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()