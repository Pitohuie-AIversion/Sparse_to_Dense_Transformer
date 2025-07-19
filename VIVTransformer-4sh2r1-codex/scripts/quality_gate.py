#!/usr/bin/env python3
"""
质量门禁脚本 - 强制执行代码质量标准

这个脚本用于在CI/CD流水线中实施质量门禁，确保代码满足预定义的质量标准。
它会运行各种代码质量检查工具，并根据配置的阈值决定是否通过质量门禁。

作者: VIVTransformer Team
日期: 2024
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from code_quality import CodeQualityChecker, QualityReport, Severity


@dataclass
class QualityThresholds:
    """质量门禁阈值配置"""
    max_critical_issues: int = 0
    max_high_issues: int = 5
    max_medium_issues: int = 20
    max_low_issues: int = 50
    min_test_coverage: float = 80.0
    max_complexity: int = 10
    max_duplicated_lines: int = 100
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QualityThresholds':
        """从字典创建阈值配置"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class QualityGate:
    """质量门禁管理器"""
    
    def __init__(self, 
                 thresholds: QualityThresholds,
                 target_dirs: List[str],
                 python_executable: str = "python"):
        self.thresholds = thresholds
        self.target_dirs = target_dirs
        self.python_executable = python_executable
        self.checker = CodeQualityChecker(target_dirs, python_executable)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def run_quality_checks(self) -> Dict[str, QualityReport]:
        """运行所有质量检查"""
        self.logger.info("开始运行质量检查...")
        
        checks = {
            'formatting': self.checker.check_formatting,
            'imports': self.checker.check_imports,
            'flake8': self.checker.check_flake8,
            'mypy': self.checker.check_mypy,
            'bandit': self.checker.check_bandit,
            'pylint': self.checker.check_pylint,
            'complexity': self.checker.check_complexity,
        }
        
        reports = {}
        for check_name, check_func in checks.items():
            try:
                self.logger.info(f"运行 {check_name} 检查...")
                reports[check_name] = check_func()
            except Exception as e:
                self.logger.error(f"{check_name} 检查失败: {e}")
                # 创建一个失败的报告
                reports[check_name] = QualityReport(
                    tool=check_name,
                    issues=[],
                    summary={'error': str(e)}
                )
        
        return reports
    
    def check_test_coverage(self) -> Tuple[float, bool]:
        """检查测试覆盖率"""
        try:
            import subprocess
            result = subprocess.run(
                [self.python_executable, '-m', 'pytest', '--cov=modify_multi_attention', 
                 '--cov-report=json', '--cov-report=term-missing'],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            # 读取覆盖率报告
            coverage_file = Path('coverage.json')
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                coverage_percent = coverage_data.get('totals', {}).get('percent_covered', 0.0)
            else:
                coverage_percent = 0.0
            
            passed = coverage_percent >= self.thresholds.min_test_coverage
            return coverage_percent, passed
            
        except Exception as e:
            self.logger.error(f"测试覆盖率检查失败: {e}")
            return 0.0, False
    
    def evaluate_quality_gate(self, reports: Dict[str, QualityReport]) -> Tuple[bool, Dict]:
        """评估质量门禁"""
        self.logger.info("评估质量门禁...")
        
        # 统计问题数量
        issue_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
        
        failed_checks = []
        
        for tool_name, report in reports.items():
            if 'error' in report.summary:
                failed_checks.append(f"{tool_name}: {report.summary['error']}")
                continue
                
            for issue in report.issues:
                if issue.severity == Severity.CRITICAL:
                    issue_counts['critical'] += 1
                elif issue.severity == Severity.HIGH:
                    issue_counts['high'] += 1
                elif issue.severity == Severity.MEDIUM:
                    issue_counts['medium'] += 1
                elif issue.severity == Severity.LOW:
                    issue_counts['low'] += 1
        
        # 检查测试覆盖率
        coverage_percent, coverage_passed = self.check_test_coverage()
        
        # 评估是否通过
        gate_passed = True
        failure_reasons = []
        
        if issue_counts['critical'] > self.thresholds.max_critical_issues:
            gate_passed = False
            failure_reasons.append(
                f"严重问题过多: {issue_counts['critical']} > {self.thresholds.max_critical_issues}"
            )
        
        if issue_counts['high'] > self.thresholds.max_high_issues:
            gate_passed = False
            failure_reasons.append(
                f"高级问题过多: {issue_counts['high']} > {self.thresholds.max_high_issues}"
            )
        
        if issue_counts['medium'] > self.thresholds.max_medium_issues:
            gate_passed = False
            failure_reasons.append(
                f"中级问题过多: {issue_counts['medium']} > {self.thresholds.max_medium_issues}"
            )
        
        if issue_counts['low'] > self.thresholds.max_low_issues:
            gate_passed = False
            failure_reasons.append(
                f"低级问题过多: {issue_counts['low']} > {self.thresholds.max_low_issues}"
            )
        
        if not coverage_passed:
            gate_passed = False
            failure_reasons.append(
                f"测试覆盖率不足: {coverage_percent:.1f}% < {self.thresholds.min_test_coverage}%"
            )
        
        if failed_checks:
            gate_passed = False
            failure_reasons.extend(failed_checks)
        
        result = {
            'passed': gate_passed,
            'issue_counts': issue_counts,
            'test_coverage': coverage_percent,
            'failure_reasons': failure_reasons,
            'thresholds': {
                'max_critical_issues': self.thresholds.max_critical_issues,
                'max_high_issues': self.thresholds.max_high_issues,
                'max_medium_issues': self.thresholds.max_medium_issues,
                'max_low_issues': self.thresholds.max_low_issues,
                'min_test_coverage': self.thresholds.min_test_coverage,
            }
        }
        
        return gate_passed, result
    
    def generate_report(self, 
                       reports: Dict[str, QualityReport], 
                       gate_result: Dict,
                       output_file: Optional[str] = None) -> str:
        """生成质量门禁报告"""
        report_lines = [
            "# 质量门禁报告",
            "",
            f"## 总体结果: {'✅ 通过' if gate_result['passed'] else '❌ 失败'}",
            ""
        ]
        
        # 问题统计
        report_lines.extend([
            "## 问题统计",
            "",
            f"- 严重问题: {gate_result['issue_counts']['critical']} / {gate_result['thresholds']['max_critical_issues']}",
            f"- 高级问题: {gate_result['issue_counts']['high']} / {gate_result['thresholds']['max_high_issues']}",
            f"- 中级问题: {gate_result['issue_counts']['medium']} / {gate_result['thresholds']['max_medium_issues']}",
            f"- 低级问题: {gate_result['issue_counts']['low']} / {gate_result['thresholds']['max_low_issues']}",
            f"- 测试覆盖率: {gate_result['test_coverage']:.1f}% / {gate_result['thresholds']['min_test_coverage']}%",
            ""
        ])
        
        # 失败原因
        if gate_result['failure_reasons']:
            report_lines.extend([
                "## 失败原因",
                ""
            ])
            for reason in gate_result['failure_reasons']:
                report_lines.append(f"- {reason}")
            report_lines.append("")
        
        # 详细检查结果
        report_lines.extend([
            "## 详细检查结果",
            ""
        ])
        
        for tool_name, report in reports.items():
            status = "✅" if not report.issues else "⚠️"
            report_lines.append(f"### {status} {tool_name.title()}")
            report_lines.append("")
            
            if 'error' in report.summary:
                report_lines.append(f"❌ 错误: {report.summary['error']}")
            elif report.issues:
                for issue in report.issues[:5]:  # 只显示前5个问题
                    severity_icon = {
                        Severity.CRITICAL: "🔴",
                        Severity.HIGH: "🟠",
                        Severity.MEDIUM: "🟡",
                        Severity.LOW: "🔵"
                    }.get(issue.severity, "⚪")
                    report_lines.append(f"  {severity_icon} {issue.file}:{issue.line} - {issue.message}")
                
                if len(report.issues) > 5:
                    report_lines.append(f"  ... 还有 {len(report.issues) - 5} 个问题")
            else:
                report_lines.append("✅ 没有发现问题")
            
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.info(f"报告已保存到: {output_file}")
        
        return report_content
    
    def run(self, output_file: Optional[str] = None) -> bool:
        """运行完整的质量门禁检查"""
        self.logger.info("开始质量门禁检查...")
        
        # 运行质量检查
        reports = self.run_quality_checks()
        
        # 评估质量门禁
        gate_passed, gate_result = self.evaluate_quality_gate(reports)
        
        # 生成报告
        report_content = self.generate_report(reports, gate_result, output_file)
        
        # 输出结果
        if gate_passed:
            self.logger.info("🎉 质量门禁检查通过!")
            print("\n" + "="*50)
            print("🎉 质量门禁检查通过!")
            print("="*50)
        else:
            self.logger.error("❌ 质量门禁检查失败!")
            print("\n" + "="*50)
            print("❌ 质量门禁检查失败!")
            print("="*50)
            print("\n失败原因:")
            for reason in gate_result['failure_reasons']:
                print(f"  - {reason}")
        
        print(f"\n详细报告: {output_file or '控制台输出'}")
        if not output_file:
            print("\n" + report_content)
        
        return gate_passed


def load_config(config_file: str) -> QualityThresholds:
    """加载配置文件"""
    if not os.path.exists(config_file):
        return QualityThresholds()
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.endswith('.json'):
                config_data = json.load(f)
            else:
                import yaml
                config_data = yaml.safe_load(f)
        
        return QualityThresholds.from_dict(config_data.get('quality_gate', {}))
    except Exception as e:
        print(f"警告: 无法加载配置文件 {config_file}: {e}")
        return QualityThresholds()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="质量门禁检查工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --target-dirs modify_multi_attention tests
  %(prog)s --config quality_gate.json --output gate_report.md
  %(prog)s --strict --target-dirs modify_multi_attention
        """
    )
    
    parser.add_argument(
        '--target-dirs',
        nargs='+',
        default=['modify_multi_attention'],
        help='要检查的目标目录 (默认: modify_multi_attention)'
    )
    
    parser.add_argument(
        '--config',
        help='配置文件路径 (JSON或YAML格式)'
    )
    
    parser.add_argument(
        '--output',
        help='输出报告文件路径'
    )
    
    parser.add_argument(
        '--python',
        default='python',
        help='Python解释器路径 (默认: python)'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='使用严格模式 (更低的阈值)'
    )
    
    parser.add_argument(
        '--max-critical',
        type=int,
        help='最大严重问题数量'
    )
    
    parser.add_argument(
        '--max-high',
        type=int,
        help='最大高级问题数量'
    )
    
    parser.add_argument(
        '--max-medium',
        type=int,
        help='最大中级问题数量'
    )
    
    parser.add_argument(
        '--max-low',
        type=int,
        help='最大低级问题数量'
    )
    
    parser.add_argument(
        '--min-coverage',
        type=float,
        help='最小测试覆盖率百分比'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        thresholds = load_config(args.config)
    else:
        thresholds = QualityThresholds()
    
    # 应用命令行参数覆盖
    if args.strict:
        thresholds.max_critical_issues = 0
        thresholds.max_high_issues = 2
        thresholds.max_medium_issues = 10
        thresholds.max_low_issues = 25
        thresholds.min_test_coverage = 90.0
    
    if args.max_critical is not None:
        thresholds.max_critical_issues = args.max_critical
    if args.max_high is not None:
        thresholds.max_high_issues = args.max_high
    if args.max_medium is not None:
        thresholds.max_medium_issues = args.max_medium
    if args.max_low is not None:
        thresholds.max_low_issues = args.max_low
    if args.min_coverage is not None:
        thresholds.min_test_coverage = args.min_coverage
    
    # 创建质量门禁检查器
    gate = QualityGate(
        thresholds=thresholds,
        target_dirs=args.target_dirs,
        python_executable=args.python
    )
    
    # 运行检查
    try:
        passed = gate.run(args.output)
        sys.exit(0 if passed else 1)
    except KeyboardInterrupt:
        print("\n检查被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"检查过程中发生错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()