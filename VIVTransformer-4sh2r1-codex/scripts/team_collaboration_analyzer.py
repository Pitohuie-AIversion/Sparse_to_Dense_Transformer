#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIVTransformer 团队协作质量分析工具

功能:
- 代码所有权和贡献分析
- 知识共享和文档覆盖率评估
- 团队协作模式分析
- 代码审查质量评估
- 团队技能分布分析
- 协作效率指标计算

作者: VIVTransformer Team
日期: 2024
"""

import os
import sys
import git
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import ast

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ContributorStats:
    """贡献者统计信息"""
    name: str
    email: str
    commits_count: int
    lines_added: int
    lines_deleted: int
    files_modified: int
    first_commit: str
    last_commit: str
    expertise_areas: List[str]
    collaboration_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CodeOwnership:
    """代码所有权信息"""
    file_path: str
    primary_owner: str
    secondary_owners: List[str]
    ownership_distribution: Dict[str, float]
    last_modified: str
    modification_frequency: int
    risk_level: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeGap:
    """知识缺口信息"""
    area: str
    description: str
    affected_files: List[str]
    risk_level: str
    recommended_actions: List[str]
    experts: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CollaborationMetrics:
    """协作指标"""
    team_size: int
    active_contributors: int
    collaboration_index: float
    knowledge_distribution: float
    bus_factor: int
    review_coverage: float
    documentation_coverage: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GitAnalyzer:
    """Git仓库分析器"""
    
    def __init__(self, repo_path: str):
        try:
            self.repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError:
            raise ValueError(f"无效的Git仓库: {repo_path}")
        
        self.repo_path = repo_path
        
    def get_contributors(self, since_days: int = 365) -> List[ContributorStats]:
        """获取贡献者统计信息"""
        since_date = datetime.now() - timedelta(days=since_days)
        
        contributors = {}
        
        # 遍历所有提交
        for commit in self.repo.iter_commits(since=since_date):
            author_name = commit.author.name
            author_email = commit.author.email
            
            if author_name not in contributors:
                contributors[author_name] = {
                    'name': author_name,
                    'email': author_email,
                    'commits': [],
                    'lines_added': 0,
                    'lines_deleted': 0,
                    'files_modified': set(),
                    'first_commit': commit.committed_datetime,
                    'last_commit': commit.committed_datetime
                }
            
            contributor = contributors[author_name]
            contributor['commits'].append(commit)
            contributor['last_commit'] = max(contributor['last_commit'], commit.committed_datetime)
            contributor['first_commit'] = min(contributor['first_commit'], commit.committed_datetime)
            
            # 统计代码变更
            try:
                stats = commit.stats
                contributor['lines_added'] += stats.total['insertions']
                contributor['lines_deleted'] += stats.total['deletions']
                contributor['files_modified'].update(stats.files.keys())
            except Exception as e:
                logger.warning(f"无法获取提交统计信息: {e}")
        
        # 转换为ContributorStats对象
        contributor_stats = []
        for name, data in contributors.items():
            expertise_areas = self._analyze_expertise_areas(data['files_modified'])
            collaboration_score = self._calculate_collaboration_score(data, contributors)
            
            stats = ContributorStats(
                name=data['name'],
                email=data['email'],
                commits_count=len(data['commits']),
                lines_added=data['lines_added'],
                lines_deleted=data['lines_deleted'],
                files_modified=len(data['files_modified']),
                first_commit=data['first_commit'].isoformat(),
                last_commit=data['last_commit'].isoformat(),
                expertise_areas=expertise_areas,
                collaboration_score=collaboration_score
            )
            contributor_stats.append(stats)
        
        return sorted(contributor_stats, key=lambda x: x.commits_count, reverse=True)
    
    def _analyze_expertise_areas(self, files: Set[str]) -> List[str]:
        """分析专业领域"""
        areas = []
        
        # 根据文件路径和扩展名判断专业领域
        file_patterns = {
            'Frontend': ['.js', '.jsx', '.ts', '.tsx', '.vue', '.html', '.css', '.scss'],
            'Backend': ['.py', '.java', '.go', '.php', '.rb', '.cs'],
            'Database': ['.sql', '.migration', 'models/', 'schema/'],
            'DevOps': ['Dockerfile', '.yml', '.yaml', 'deploy/', 'ci/', '.github/'],
            'Testing': ['test_', '_test.', 'tests/', 'spec/'],
            'Documentation': ['.md', '.rst', 'docs/', 'README'],
            'Configuration': ['.json', '.xml', '.ini', '.conf', 'config/']
        }
        
        for area, patterns in file_patterns.items():
            if any(any(pattern in file for pattern in patterns) for file in files):
                areas.append(area)
        
        return areas or ['General']
    
    def _calculate_collaboration_score(self, contributor_data: Dict, all_contributors: Dict) -> float:
        """计算协作分数"""
        # 基于提交频率、文件覆盖范围、与其他贡献者的交互等计算
        commits_count = len(contributor_data['commits'])
        files_count = len(contributor_data['files_modified'])
        
        # 标准化分数
        max_commits = max(len(data['commits']) for data in all_contributors.values())
        max_files = max(len(data['files_modified']) for data in all_contributors.values())
        
        commit_score = commits_count / max_commits if max_commits > 0 else 0
        file_score = files_count / max_files if max_files > 0 else 0
        
        return round((commit_score + file_score) / 2 * 100, 2)
    
    def analyze_code_ownership(self) -> List[CodeOwnership]:
        """分析代码所有权"""
        ownership_data = []
        
        # 获取所有Python文件
        python_files = list(Path(self.repo_path).rglob('*.py'))
        
        for file_path in python_files:
            try:
                # 获取文件的提交历史
                commits = list(self.repo.iter_commits(paths=str(file_path), max_count=50))
                
                if not commits:
                    continue
                
                # 统计每个作者的贡献
                author_contributions = defaultdict(int)
                for commit in commits:
                    author_contributions[commit.author.name] += 1
                
                # 计算所有权分布
                total_commits = sum(author_contributions.values())
                ownership_distribution = {
                    author: round(count / total_commits * 100, 2)
                    for author, count in author_contributions.items()
                }
                
                # 确定主要和次要所有者
                sorted_owners = sorted(ownership_distribution.items(), key=lambda x: x[1], reverse=True)
                primary_owner = sorted_owners[0][0] if sorted_owners else "Unknown"
                secondary_owners = [owner for owner, _ in sorted_owners[1:4]]  # 前3个次要所有者
                
                # 计算风险等级
                risk_level = self._calculate_ownership_risk(ownership_distribution)
                
                ownership = CodeOwnership(
                    file_path=str(file_path.relative_to(self.repo_path)),
                    primary_owner=primary_owner,
                    secondary_owners=secondary_owners,
                    ownership_distribution=ownership_distribution,
                    last_modified=commits[0].committed_datetime.isoformat(),
                    modification_frequency=len(commits),
                    risk_level=risk_level
                )
                
                ownership_data.append(ownership)
                
            except Exception as e:
                logger.warning(f"分析文件 {file_path} 所有权时出错: {e}")
        
        return ownership_data
    
    def _calculate_ownership_risk(self, ownership_distribution: Dict[str, float]) -> str:
        """计算所有权风险等级"""
        if not ownership_distribution:
            return "high"
        
        max_ownership = max(ownership_distribution.values())
        owner_count = len(ownership_distribution)
        
        if max_ownership > 80 and owner_count == 1:
            return "critical"  # 单点故障
        elif max_ownership > 60 and owner_count <= 2:
            return "high"  # 高风险
        elif max_ownership > 40 or owner_count <= 3:
            return "medium"  # 中等风险
        else:
            return "low"  # 低风险


class DocumentationAnalyzer:
    """文档分析器"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
    
    def analyze_documentation_coverage(self) -> Dict[str, Any]:
        """分析文档覆盖率"""
        python_files = list(Path(self.project_path).rglob('*.py'))
        doc_files = list(Path(self.project_path).rglob('*.md')) + \
                   list(Path(self.project_path).rglob('*.rst'))
        
        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                        if ast.get_docstring(node):
                            documented_functions += 1
                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
                        if ast.get_docstring(node):
                            documented_classes += 1
            
            except Exception as e:
                logger.warning(f"分析文件 {file_path} 文档时出错: {e}")
        
        function_coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 0
        class_coverage = (documented_classes / total_classes * 100) if total_classes > 0 else 0
        overall_coverage = ((documented_functions + documented_classes) / 
                          (total_functions + total_classes) * 100) if (total_functions + total_classes) > 0 else 0
        
        return {
            'total_python_files': len(python_files),
            'total_doc_files': len(doc_files),
            'function_coverage': round(function_coverage, 2),
            'class_coverage': round(class_coverage, 2),
            'overall_coverage': round(overall_coverage, 2),
            'total_functions': total_functions,
            'documented_functions': documented_functions,
            'total_classes': total_classes,
            'documented_classes': documented_classes,
            'documentation_files': [str(f.relative_to(self.project_path)) for f in doc_files]
        }
    
    def identify_knowledge_gaps(self, contributors: List[ContributorStats], 
                              ownership_data: List[CodeOwnership]) -> List[KnowledgeGap]:
        """识别知识缺口"""
        gaps = []
        
        # 分析单点故障风险
        critical_files = [o for o in ownership_data if o.risk_level == 'critical']
        if critical_files:
            gap = KnowledgeGap(
                area="Critical File Ownership",
                description=f"{len(critical_files)} 个关键文件只有单一维护者",
                affected_files=[f.file_path for f in critical_files],
                risk_level="critical",
                recommended_actions=[
                    "安排代码审查和知识分享会议",
                    "创建详细的技术文档",
                    "培训其他团队成员"
                ],
                experts=[f.primary_owner for f in critical_files]
            )
            gaps.append(gap)
        
        # 分析专业领域分布
        expertise_distribution = defaultdict(list)
        for contributor in contributors:
            for area in contributor.expertise_areas:
                expertise_distribution[area].append(contributor.name)
        
        for area, experts in expertise_distribution.items():
            if len(experts) <= 1:
                gap = KnowledgeGap(
                    area=f"{area} Expertise",
                    description=f"{area} 领域只有 {len(experts)} 个专家",
                    affected_files=[],
                    risk_level="high" if len(experts) == 1 else "medium",
                    recommended_actions=[
                        f"培训更多团队成员掌握 {area} 技能",
                        "创建技术指南和最佳实践文档",
                        "安排跨领域协作项目"
                    ],
                    experts=experts
                )
                gaps.append(gap)
        
        return gaps


class TeamCollaborationAnalyzer:
    """团队协作分析器"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.git_analyzer = None
        self.doc_analyzer = DocumentationAnalyzer(project_path)
        
        # 尝试初始化Git分析器
        try:
            self.git_analyzer = GitAnalyzer(project_path)
        except ValueError as e:
            logger.warning(f"Git分析器初始化失败: {e}")
    
    def analyze_team_collaboration(self) -> Dict[str, Any]:
        """分析团队协作情况"""
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'project_path': self.project_path,
            'contributors': [],
            'code_ownership': [],
            'knowledge_gaps': [],
            'documentation_analysis': {},
            'collaboration_metrics': {},
            'recommendations': []
        }
        
        # Git仓库分析
        if self.git_analyzer:
            logger.info("分析贡献者信息...")
            contributors = self.git_analyzer.get_contributors()
            results['contributors'] = [c.to_dict() for c in contributors]
            
            logger.info("分析代码所有权...")
            ownership_data = self.git_analyzer.analyze_code_ownership()
            results['code_ownership'] = [o.to_dict() for o in ownership_data]
            
            # 识别知识缺口
            logger.info("识别知识缺口...")
            knowledge_gaps = self.doc_analyzer.identify_knowledge_gaps(contributors, ownership_data)
            results['knowledge_gaps'] = [g.to_dict() for g in knowledge_gaps]
            
            # 计算协作指标
            logger.info("计算协作指标...")
            collaboration_metrics = self._calculate_collaboration_metrics(contributors, ownership_data)
            results['collaboration_metrics'] = collaboration_metrics.to_dict()
        
        # 文档分析
        logger.info("分析文档覆盖率...")
        doc_analysis = self.doc_analyzer.analyze_documentation_coverage()
        results['documentation_analysis'] = doc_analysis
        
        # 生成建议
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _calculate_collaboration_metrics(self, contributors: List[ContributorStats], 
                                       ownership_data: List[CodeOwnership]) -> CollaborationMetrics:
        """计算协作指标"""
        if not contributors:
            return CollaborationMetrics(
                team_size=0, active_contributors=0, collaboration_index=0.0,
                knowledge_distribution=0.0, bus_factor=0, review_coverage=0.0,
                documentation_coverage=0.0
            )
        
        # 团队规模和活跃贡献者
        team_size = len(contributors)
        recent_date = datetime.now() - timedelta(days=30)
        active_contributors = sum(1 for c in contributors 
                                if datetime.fromisoformat(c.last_commit) > recent_date)
        
        # 协作指数（基于贡献分布的均匀程度）
        total_commits = sum(c.commits_count for c in contributors)
        if total_commits > 0:
            commit_distribution = [c.commits_count / total_commits for c in contributors]
            # 使用基尼系数的反向作为协作指数
            collaboration_index = 1.0 - self._calculate_gini_coefficient(commit_distribution)
        else:
            collaboration_index = 0.0
        
        # 知识分布（专业领域的覆盖程度）
        all_areas = set()
        for c in contributors:
            all_areas.update(c.expertise_areas)
        
        area_coverage = defaultdict(int)
        for c in contributors:
            for area in c.expertise_areas:
                area_coverage[area] += 1
        
        # 计算知识分布均匀程度
        if area_coverage:
            area_counts = list(area_coverage.values())
            knowledge_distribution = 1.0 - self._calculate_gini_coefficient(
                [count / sum(area_counts) for count in area_counts]
            )
        else:
            knowledge_distribution = 0.0
        
        # Bus Factor（关键人员流失风险）
        critical_ownership = [o for o in ownership_data if o.risk_level in ['critical', 'high']]
        unique_critical_owners = set(o.primary_owner for o in critical_ownership)
        bus_factor = len(unique_critical_owners)
        
        # 审查覆盖率（简化计算）
        review_coverage = min(80.0, active_contributors / team_size * 100) if team_size > 0 else 0.0
        
        return CollaborationMetrics(
            team_size=team_size,
            active_contributors=active_contributors,
            collaboration_index=round(collaboration_index * 100, 2),
            knowledge_distribution=round(knowledge_distribution * 100, 2),
            bus_factor=bus_factor,
            review_coverage=round(review_coverage, 2),
            documentation_coverage=0.0  # 将在文档分析中设置
        )
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """计算基尼系数"""
        if not values or len(values) == 1:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = sum((i + 1) * val for i, val in enumerate(sorted_values))
        
        return (2 * cumsum) / (n * sum(sorted_values)) - (n + 1) / n
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于协作指标的建议
        metrics = results.get('collaboration_metrics', {})
        
        if metrics.get('collaboration_index', 0) < 50:
            recommendations.append("提高团队协作均衡性，避免工作过度集中在少数人身上")
        
        if metrics.get('bus_factor', 0) > 3:
            recommendations.append("降低关键人员依赖风险，加强知识分享和文档建设")
        
        if metrics.get('knowledge_distribution', 0) < 60:
            recommendations.append("改善知识分布，培养跨领域技能")
        
        # 基于文档覆盖率的建议
        doc_analysis = results.get('documentation_analysis', {})
        if doc_analysis.get('overall_coverage', 0) < 70:
            recommendations.append("提高代码文档覆盖率，特别是函数和类的文档字符串")
        
        # 基于知识缺口的建议
        knowledge_gaps = results.get('knowledge_gaps', [])
        critical_gaps = [g for g in knowledge_gaps if g.get('risk_level') == 'critical']
        if critical_gaps:
            recommendations.append("优先解决关键知识缺口，安排知识转移和培训")
        
        # 基于代码所有权的建议
        ownership_data = results.get('code_ownership', [])
        critical_files = [o for o in ownership_data if o.get('risk_level') == 'critical']
        if len(critical_files) > 5:
            recommendations.append("减少单点故障文件，鼓励代码审查和协作开发")
        
        # 通用建议
        if not recommendations:
            recommendations.extend([
                "继续保持良好的团队协作模式",
                "定期进行团队协作质量评估",
                "建立持续的知识分享机制"
            ])
        
        return recommendations
    
    def generate_collaboration_report(self, results: Dict[str, Any]) -> str:
        """生成团队协作报告"""
        report_lines = []
        
        # 报告标题
        report_lines.append("# 团队协作质量分析报告")
        report_lines.append(f"\n**生成时间**: {results['analysis_timestamp']}")
        report_lines.append(f"**项目路径**: {results['project_path']}")
        
        # 团队概览
        contributors = results.get('contributors', [])
        metrics = results.get('collaboration_metrics', {})
        
        report_lines.append("\n## 👥 团队概览")
        report_lines.append(f"\n**团队规模**: {metrics.get('team_size', 0)}")
        report_lines.append(f"**活跃贡献者**: {metrics.get('active_contributors', 0)}")
        report_lines.append(f"**协作指数**: {metrics.get('collaboration_index', 0):.2f}/100")
        report_lines.append(f"**知识分布**: {metrics.get('knowledge_distribution', 0):.2f}/100")
        report_lines.append(f"**Bus Factor**: {metrics.get('bus_factor', 0)}")
        
        # 贡献者分析
        if contributors:
            report_lines.append("\n## 🏆 贡献者分析")
            
            # 排序贡献者
            sorted_contributors = sorted(contributors, key=lambda x: x['commits_count'], reverse=True)
            
            report_lines.append("\n### 主要贡献者")
            for contributor in sorted_contributors[:5]:
                report_lines.append(f"\n#### {contributor['name']}")
                report_lines.append(f"- **提交数**: {contributor['commits_count']}")
                report_lines.append(f"- **代码行数**: +{contributor['lines_added']} -{contributor['lines_deleted']}")
                report_lines.append(f"- **修改文件**: {contributor['files_modified']}")
                report_lines.append(f"- **专业领域**: {', '.join(contributor['expertise_areas'])}")
                report_lines.append(f"- **协作分数**: {contributor['collaboration_score']}/100")
            
            # 专业领域分布
            expertise_count = defaultdict(int)
            for contributor in contributors:
                for area in contributor['expertise_areas']:
                    expertise_count[area] += 1
            
            if expertise_count:
                report_lines.append("\n### 专业领域分布")
                for area, count in sorted(expertise_count.items(), key=lambda x: x[1], reverse=True):
                    report_lines.append(f"- **{area}**: {count} 人")
        
        # 代码所有权分析
        ownership_data = results.get('code_ownership', [])
        if ownership_data:
            report_lines.append("\n## 🔐 代码所有权分析")
            
            # 风险统计
            risk_stats = defaultdict(int)
            for ownership in ownership_data:
                risk_stats[ownership['risk_level']] += 1
            
            report_lines.append(f"\n**总文件数**: {len(ownership_data)}")
            for risk, count in risk_stats.items():
                emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(risk, '⚪')
                report_lines.append(f"- {emoji} {risk.title()}: {count}")
            
            # 高风险文件
            high_risk_files = [o for o in ownership_data if o['risk_level'] in ['critical', 'high']]
            if high_risk_files:
                report_lines.append("\n### ⚠️ 高风险文件")
                for ownership in high_risk_files[:10]:
                    report_lines.append(f"\n- **{ownership['file_path']}**")
                    report_lines.append(f"  - 主要维护者: {ownership['primary_owner']}")
                    report_lines.append(f"  - 风险等级: {ownership['risk_level']}")
                    report_lines.append(f"  - 修改频率: {ownership['modification_frequency']}")
        
        # 知识缺口分析
        knowledge_gaps = results.get('knowledge_gaps', [])
        if knowledge_gaps:
            report_lines.append("\n## 🧠 知识缺口分析")
            
            for gap in knowledge_gaps:
                risk_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(gap['risk_level'], '⚪')
                report_lines.append(f"\n### {risk_emoji} {gap['area']}")
                report_lines.append(f"- **描述**: {gap['description']}")
                report_lines.append(f"- **风险等级**: {gap['risk_level']}")
                report_lines.append(f"- **专家**: {', '.join(gap['experts'])}")
                
                if gap['recommended_actions']:
                    report_lines.append("- **建议行动**:")
                    for action in gap['recommended_actions']:
                        report_lines.append(f"  - {action}")
        
        # 文档质量分析
        doc_analysis = results.get('documentation_analysis', {})
        if doc_analysis:
            report_lines.append("\n## 📚 文档质量分析")
            report_lines.append(f"\n**总体覆盖率**: {doc_analysis.get('overall_coverage', 0):.2f}%")
            report_lines.append(f"**函数文档覆盖率**: {doc_analysis.get('function_coverage', 0):.2f}%")
            report_lines.append(f"**类文档覆盖率**: {doc_analysis.get('class_coverage', 0):.2f}%")
            report_lines.append(f"**Python文件数**: {doc_analysis.get('total_python_files', 0)}")
            report_lines.append(f"**文档文件数**: {doc_analysis.get('total_doc_files', 0)}")
        
        # 改进建议
        recommendations = results.get('recommendations', [])
        if recommendations:
            report_lines.append("\n## 💡 改进建议")
            
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"\n{i}. {rec}")
        
        # 行动计划
        report_lines.append("\n## 🎯 行动计划")
        
        if knowledge_gaps:
            critical_gaps = [g for g in knowledge_gaps if g['risk_level'] == 'critical']
            if critical_gaps:
                report_lines.append("\n1. **紧急行动**: 解决关键知识缺口")
                for gap in critical_gaps[:3]:
                    report_lines.append(f"   - {gap['area']}: {gap['description']}")
        
        if doc_analysis.get('overall_coverage', 0) < 70:
            report_lines.append("\n2. **文档改进**: 提高代码文档覆盖率")
            report_lines.append("   - 为关键函数和类添加文档字符串")
            report_lines.append("   - 创建技术设计文档")
        
        if metrics.get('collaboration_index', 0) < 50:
            report_lines.append("\n3. **协作优化**: 改善团队协作模式")
            report_lines.append("   - 鼓励代码审查和结对编程")
            report_lines.append("   - 建立知识分享机制")
        
        report_lines.append("\n4. **持续改进**: 建立定期评估机制")
        report_lines.append("   - 每月生成团队协作质量报告")
        report_lines.append("   - 跟踪关键指标变化趋势")
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="VIVTransformer 团队协作质量分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --analyze .
  %(prog)s --contributors --since-days 90
  %(prog)s --ownership --risk-level critical
        """
    )
    
    parser.add_argument(
        '--analyze',
        metavar='PATH',
        help='分析指定项目的团队协作质量'
    )
    
    parser.add_argument(
        '--contributors',
        action='store_true',
        help='分析贡献者信息'
    )
    
    parser.add_argument(
        '--ownership',
        action='store_true',
        help='分析代码所有权'
    )
    
    parser.add_argument(
        '--knowledge-gaps',
        action='store_true',
        help='识别知识缺口'
    )
    
    parser.add_argument(
        '--since-days',
        type=int,
        default=365,
        help='分析时间范围（天数）'
    )
    
    parser.add_argument(
        '--risk-level',
        choices=['critical', 'high', 'medium', 'low'],
        help='过滤指定风险等级的问题'
    )
    
    parser.add_argument(
        '--output',
        default='reports/team_collaboration_report.md',
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
        target_path = args.analyze or '.'
        
        if not os.path.exists(target_path):
            print(f"❌ 路径不存在: {target_path}")
            sys.exit(1)
        
        print(f"🔍 分析团队协作质量: {target_path}")
        
        # 初始化分析器
        analyzer = TeamCollaborationAnalyzer(target_path)
        
        # 执行分析
        results = analyzer.analyze_team_collaboration()
        
        # 显示结果摘要
        print("\n" + "="*60)
        print("👥 团队协作分析结果")
        
        metrics = results.get('collaboration_metrics', {})
        print(f"👥 团队规模: {metrics.get('team_size', 0)}")
        print(f"🏃 活跃贡献者: {metrics.get('active_contributors', 0)}")
        print(f"🤝 协作指数: {metrics.get('collaboration_index', 0):.2f}/100")
        print(f"🧠 知识分布: {metrics.get('knowledge_distribution', 0):.2f}/100")
        print(f"🚌 Bus Factor: {metrics.get('bus_factor', 0)}")
        
        knowledge_gaps = results.get('knowledge_gaps', [])
        critical_gaps = [g for g in knowledge_gaps if g.get('risk_level') == 'critical']
        print(f"⚠️  知识缺口: {len(critical_gaps)}/{len(knowledge_gaps)}")
        
        doc_analysis = results.get('documentation_analysis', {})
        print(f"📚 文档覆盖率: {doc_analysis.get('overall_coverage', 0):.2f}%")
        
        # 生成报告
        if args.format == 'json':
            output_content = json.dumps(results, indent=2, ensure_ascii=False)
        else:
            output_content = analyzer.generate_collaboration_report(results)
        
        # 保存报告
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        print(f"\n📋 详细报告已保存: {args.output}")
        
        # 显示关键建议
        recommendations = results.get('recommendations', [])
        if recommendations:
            print("\n💡 关键建议:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⏹️  操作被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"团队协作分析失败: {e}")
        print(f"❌ 团队协作分析失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()