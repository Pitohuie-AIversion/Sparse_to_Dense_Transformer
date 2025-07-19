#!/usr/bin/env python3
"""代码质量检查脚本

集成多种代码质量工具，提供全面的代码质量分析
"""

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    """问题严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityIssue:
    """代码质量问题"""
    tool: str
    file: str
    line: Optional[int]
    column: Optional[int]
    severity: Severity
    code: str
    message: str
    rule: Optional[str] = None


@dataclass
class QualityReport:
    """质量报告"""
    tool: str
    success: bool
    duration: float
    issues: List[QualityIssue]
    summary: Dict[str, int]
    raw_output: str = ""


class CodeQualityChecker:
    """代码质量检查器"""
    
    def __init__(self, project_root: Path, target_dirs: List[str] = None):
        self.project_root = project_root
        self.target_dirs = target_dirs or ['modify_multi_attention', 'tests', 'scripts']
        self.python_executable = sys.executable
        self.reports: List[QualityReport] = []
        
    def run_command(self, cmd: List[str], cwd: Path = None) -> Tuple[int, str, str]:
        """运行命令并返回结果"""
        if cwd is None:
            cwd = self.project_root
            
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def check_flake8(self) -> QualityReport:
        """运行flake8检查"""
        print("Running flake8...")
        start_time = time.time()
        
        cmd = [
            self.python_executable, '-m', 'flake8',
            '--format=json',
            '--max-line-length=88',
            '--extend-ignore=E203,W503',
            '--exclude=.git,__pycache__,.pytest_cache,.mypy_cache,build,dist,*.egg-info'
        ] + self.target_dirs
        
        returncode, stdout, stderr = self.run_command(cmd)
        duration = time.time() - start_time
        
        issues = []
        summary = {'info': 0, 'warning': 0, 'error': 0, 'critical': 0}
        
        if returncode == 0:
            success = True
        else:
            success = False
            if stdout:
                try:
                    # flake8 JSON格式输出
                    for line in stdout.strip().split('\n'):
                        if line.strip():
                            data = json.loads(line)
                            severity = Severity.WARNING if data['code'].startswith('W') else Severity.ERROR
                            issue = QualityIssue(
                                tool='flake8',
                                file=data['filename'],
                                line=data['line_number'],
                                column=data['column_number'],
                                severity=severity,
                                code=data['code'],
                                message=data['text'],
                                rule=data['code']
                            )
                            issues.append(issue)
                            summary[severity.value] += 1
                except json.JSONDecodeError:
                    # 如果JSON解析失败，尝试解析标准格式
                    for line in stdout.strip().split('\n'):
                        if ':' in line and line.strip():
                            parts = line.split(':')
                            if len(parts) >= 4:
                                file_path = parts[0]
                                line_num = int(parts[1]) if parts[1].isdigit() else None
                                col_num = int(parts[2]) if parts[2].isdigit() else None
                                message = ':'.join(parts[3:]).strip()
                                
                                # 提取错误代码
                                code = ''
                                if ' ' in message:
                                    first_word = message.split()[0]
                                    if first_word.startswith(('E', 'W', 'F', 'C')):
                                        code = first_word
                                
                                severity = Severity.WARNING if code.startswith('W') else Severity.ERROR
                                issue = QualityIssue(
                                    tool='flake8',
                                    file=file_path,
                                    line=line_num,
                                    column=col_num,
                                    severity=severity,
                                    code=code,
                                    message=message,
                                    rule=code
                                )
                                issues.append(issue)
                                summary[severity.value] += 1
        
        return QualityReport(
            tool='flake8',
            success=success,
            duration=duration,
            issues=issues,
            summary=summary,
            raw_output=stdout + stderr
        )
    
    def check_formatting(self) -> QualityReport:
        """检查代码格式"""
        print("Checking code formatting...")
        start_time = time.time()
        
        cmd = [
            self.python_executable, '-m', 'black',
            '--check',
            '--diff'
        ] + self.target_dirs
        
        returncode, stdout, stderr = self.run_command(cmd)
        duration = time.time() - start_time
        
        issues = []
        summary = {'info': 0, 'warning': 0, 'error': 0, 'critical': 0}
        
        success = returncode == 0
        
        if not success and stdout:
            # 解析black输出
            for line in stdout.split('\n'):
                if line.startswith('would reformat'):
                    file_path = line.split()[-1]
                    issue = QualityIssue(
                        tool='black',
                        file=file_path,
                        line=None,
                        column=None,
                        severity=Severity.WARNING,
                        code='format',
                        message='File would be reformatted by black',
                        rule='code-formatting'
                    )
                    issues.append(issue)
                    summary['warning'] += 1
        
        return QualityReport(
            tool='black',
            success=success,
            duration=duration,
            issues=issues,
            summary=summary,
            raw_output=stdout + stderr
        )
    
    def check_imports(self) -> QualityReport:
        """检查导入排序"""
        print("Checking import sorting...")
        start_time = time.time()
        
        cmd = [
            self.python_executable, '-m', 'isort',
            '--check-only',
            '--diff',
            '--profile', 'black'
        ] + self.target_dirs
        
        returncode, stdout, stderr = self.run_command(cmd)
        duration = time.time() - start_time
        
        issues = []
        summary = {'info': 0, 'warning': 0, 'error': 0, 'critical': 0}
        
        success = returncode == 0
        
        if not success and stdout:
            # 解析isort输出
            current_file = None
            for line in stdout.split('\n'):
                if line.startswith('ERROR:') or line.startswith('would reformat'):
                    if 'would reformat' in line:
                        current_file = line.split()[-1]
                    
                    issue = QualityIssue(
                        tool='isort',
                        file=current_file or 'unknown',
                        line=None,
                        column=None,
                        severity=Severity.WARNING,
                        code='import-order',
                        message='Import statements are not properly sorted',
                        rule='import-sorting'
                    )
                    issues.append(issue)
                    summary['warning'] += 1
        
        return QualityReport(
            tool='isort',
            success=success,
            duration=duration,
            issues=issues,
            summary=summary,
            raw_output=stdout + stderr
        )
    
    def check_mypy(self) -> QualityReport:
        """运行mypy类型检查"""
        print("Running mypy...")
        start_time = time.time()
        
        cmd = [
            self.python_executable, '-m', 'mypy',
            '--ignore-missing-imports',
            '--show-error-codes',
            '--no-error-summary'
        ] + self.target_dirs
        
        returncode, stdout, stderr = self.run_command(cmd)
        duration = time.time() - start_time
        
        issues = []
        summary = {'info': 0, 'warning': 0, 'error': 0, 'critical': 0}
        
        success = returncode == 0
        
        if stdout:
            for line in stdout.strip().split('\n'):
                if ':' in line and 'error:' in line:
                    parts = line.split(':')
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_num = int(parts[1]) if parts[1].isdigit() else None
                        message = ':'.join(parts[2:]).strip()
                        
                        # 提取错误代码
                        code = ''
                        if '[' in message and ']' in message:
                            start = message.rfind('[')
                            end = message.rfind(']')
                            if start < end:
                                code = message[start+1:end]
                        
                        issue = QualityIssue(
                            tool='mypy',
                            file=file_path,
                            line=line_num,
                            column=None,
                            severity=Severity.ERROR,
                            code=code,
                            message=message,
                            rule=code
                        )
                        issues.append(issue)
                        summary['error'] += 1
        
        return QualityReport(
            tool='mypy',
            success=success,
            duration=duration,
            issues=issues,
            summary=summary,
            raw_output=stdout + stderr
        )
    
    def check_bandit(self) -> QualityReport:
        """运行bandit安全检查"""
        print("Running bandit...")
        start_time = time.time()
        
        cmd = [
            self.python_executable, '-m', 'bandit',
            '-r',
            '-f', 'json',
            '-x', 'tests/'
        ] + [d for d in self.target_dirs if d != 'tests']
        
        returncode, stdout, stderr = self.run_command(cmd)
        duration = time.time() - start_time
        
        issues = []
        summary = {'info': 0, 'warning': 0, 'error': 0, 'critical': 0}
        
        success = returncode == 0
        
        if stdout:
            try:
                data = json.loads(stdout)
                for result in data.get('results', []):
                    # 映射bandit严重程度
                    severity_map = {
                        'LOW': Severity.INFO,
                        'MEDIUM': Severity.WARNING,
                        'HIGH': Severity.ERROR
                    }
                    
                    severity = severity_map.get(result['issue_severity'], Severity.WARNING)
                    issue = QualityIssue(
                        tool='bandit',
                        file=result['filename'],
                        line=result['line_number'],
                        column=None,
                        severity=severity,
                        code=result['test_id'],
                        message=result['issue_text'],
                        rule=result['test_name']
                    )
                    issues.append(issue)
                    summary[severity.value] += 1
            except json.JSONDecodeError:
                success = False
        
        return QualityReport(
            tool='bandit',
            success=success,
            duration=duration,
            issues=issues,
            summary=summary,
            raw_output=stdout + stderr
        )
    
    def check_pylint(self) -> QualityReport:
        """运行pylint检查"""
        print("Running pylint...")
        start_time = time.time()
        
        cmd = [
            self.python_executable, '-m', 'pylint',
            '--output-format=json',
            '--disable=C0114,C0115,C0116',  # 禁用文档字符串检查
            '--max-line-length=88'
        ] + self.target_dirs
        
        returncode, stdout, stderr = self.run_command(cmd)
        duration = time.time() - start_time
        
        issues = []
        summary = {'info': 0, 'warning': 0, 'error': 0, 'critical': 0}
        
        success = returncode in [0, 1, 2, 4, 8, 16]  # pylint返回码含义不同
        
        if stdout:
            try:
                data = json.loads(stdout)
                for item in data:
                    # 映射pylint类型到严重程度
                    type_map = {
                        'convention': Severity.INFO,
                        'refactor': Severity.INFO,
                        'warning': Severity.WARNING,
                        'error': Severity.ERROR,
                        'fatal': Severity.CRITICAL
                    }
                    
                    severity = type_map.get(item['type'], Severity.WARNING)
                    issue = QualityIssue(
                        tool='pylint',
                        file=item['path'],
                        line=item['line'],
                        column=item['column'],
                        severity=severity,
                        code=item['message-id'],
                        message=item['message'],
                        rule=item['symbol']
                    )
                    issues.append(issue)
                    summary[severity.value] += 1
            except json.JSONDecodeError:
                success = False
        
        return QualityReport(
            tool='pylint',
            success=success,
            duration=duration,
            issues=issues,
            summary=summary,
            raw_output=stdout + stderr
        )
    
    def check_complexity(self) -> QualityReport:
        """检查代码复杂度"""
        print("Running complexity analysis...")
        start_time = time.time()
        
        cmd = [
            self.python_executable, '-m', 'radon',
            'cc',
            '--json',
            '--min', 'B'
        ] + self.target_dirs
        
        returncode, stdout, stderr = self.run_command(cmd)
        duration = time.time() - start_time
        
        issues = []
        summary = {'info': 0, 'warning': 0, 'error': 0, 'critical': 0}
        
        success = returncode == 0
        
        if stdout:
            try:
                data = json.loads(stdout)
                for file_path, functions in data.items():
                    for func in functions:
                        complexity = func['complexity']
                        
                        # 根据复杂度确定严重程度
                        if complexity >= 20:
                            severity = Severity.CRITICAL
                        elif complexity >= 15:
                            severity = Severity.ERROR
                        elif complexity >= 10:
                            severity = Severity.WARNING
                        else:
                            severity = Severity.INFO
                        
                        issue = QualityIssue(
                            tool='radon',
                            file=file_path,
                            line=func['lineno'],
                            column=None,
                            severity=severity,
                            code=f"CC{complexity}",
                            message=f"Function '{func['name']}' has complexity {complexity} (grade {func['rank']})",
                            rule='cyclomatic-complexity'
                        )
                        issues.append(issue)
                        summary[severity.value] += 1
            except json.JSONDecodeError:
                success = False
        
        return QualityReport(
            tool='radon',
            success=success,
            duration=duration,
            issues=issues,
            summary=summary,
            raw_output=stdout + stderr
        )
    
    def run_all_checks(self, tools: List[str] = None) -> List[QualityReport]:
        """运行所有检查"""
        available_tools = {
            'flake8': self.check_flake8,
            'pylint': self.check_pylint,
            'mypy': self.check_mypy,
            'bandit': self.check_bandit,
            'radon': self.check_complexity,
            'isort': self.check_imports,
            'black': self.check_formatting
        }
        
        if tools is None:
            tools = list(available_tools.keys())
        
        reports = []
        for tool in tools:
            if tool in available_tools:
                try:
                    report = available_tools[tool]()
                    reports.append(report)
                    self.reports.append(report)
                except Exception as e:
                    print(f"Error running {tool}: {e}")
                    # 创建错误报告
                    error_report = QualityReport(
                        tool=tool,
                        success=False,
                        duration=0.0,
                        issues=[],
                        summary={'info': 0, 'warning': 0, 'error': 1, 'critical': 0},
                        raw_output=str(e)
                    )
                    reports.append(error_report)
                    self.reports.append(error_report)
            else:
                print(f"Unknown tool: {tool}")
        
        return reports
    
    def generate_summary(self) -> Dict:
        """生成总结报告"""
        total_issues = 0
        total_files = set()
        severity_counts = {'info': 0, 'warning': 0, 'error': 0, 'critical': 0}
        tool_results = {}
        
        for report in self.reports:
            tool_results[report.tool] = {
                'success': report.success,
                'duration': report.duration,
                'issues': len(report.issues),
                'summary': report.summary
            }
            
            total_issues += len(report.issues)
            for issue in report.issues:
                total_files.add(issue.file)
                severity_counts[issue.severity.value] += 1
        
        return {
            'total_issues': total_issues,
            'affected_files': len(total_files),
            'severity_counts': severity_counts,
            'tool_results': tool_results,
            'overall_success': all(r.success for r in self.reports)
        }
    
    def print_report(self, detailed: bool = False, format_type: str = 'text'):
        """打印报告"""
        if format_type == 'json':
            self._print_json_report()
        else:
            self._print_text_report(detailed)
    
    def _print_text_report(self, detailed: bool = False):
        """打印文本格式报告"""
        summary = self.generate_summary()
        
        print("\n" + "="*80)
        print("📊 CODE QUALITY REPORT")
        print("="*80)
        
        # 总体统计
        print(f"\n📈 Overall Statistics:")
        print(f"   Total Issues: {summary['total_issues']}")
        print(f"   Affected Files: {summary['affected_files']}")
        print(f"   Overall Status: {'✅ PASS' if summary['overall_success'] else '❌ FAIL'}")
        
        # 严重程度统计
        print(f"\n🚨 Issues by Severity:")
        severity_icons = {
            'critical': '🔴',
            'error': '🟠', 
            'warning': '🟡',
            'info': '🔵'
        }
        
        for severity, count in summary['severity_counts'].items():
            if count > 0:
                icon = severity_icons.get(severity, '⚪')
                print(f"   {icon} {severity.upper()}: {count}")
        
        # 工具结果
        print(f"\n🔧 Tool Results:")
        for tool, result in summary['tool_results'].items():
            status = "✅" if result['success'] else "❌"
            duration = f"{result['duration']:.2f}s"
            issues = result['issues']
            print(f"   {status} {tool:<10} ({duration:>6}) - {issues} issues")
        
        # 详细问题列表
        if detailed and summary['total_issues'] > 0:
            print(f"\n📋 Detailed Issues:")
            
            # 按文件分组
            issues_by_file = {}
            for report in self.reports:
                for issue in report.issues:
                    if issue.file not in issues_by_file:
                        issues_by_file[issue.file] = []
                    issues_by_file[issue.file].append(issue)
            
            for file_path, issues in sorted(issues_by_file.items()):
                print(f"\n📄 {file_path}:")
                
                # 按行号排序
                sorted_issues = sorted(issues, key=lambda x: (x.line or 0, x.tool))
                
                for issue in sorted_issues:
                    severity_icon = severity_icons.get(issue.severity.value, '⚪')
                    line_info = f":{issue.line}" if issue.line else ""
                    col_info = f":{issue.column}" if issue.column else ""
                    location = f"{line_info}{col_info}"
                    
                    print(f"   {severity_icon} {issue.tool:<8} {location:<8} {issue.code:<12} {issue.message}")
        
        # 建议
        if summary['total_issues'] > 0:
            print(f"\n💡 Suggestions:")
            
            if summary['severity_counts']['critical'] > 0:
                print("   🔴 Critical issues found! Please fix immediately.")
            
            if summary['severity_counts']['error'] > 0:
                print("   🟠 Errors found. Consider fixing before committing.")
            
            if summary['tool_results'].get('black', {}).get('issues', 0) > 0:
                print("   🎨 Run 'python scripts/format_code.py' to fix formatting issues.")
            
            if summary['tool_results'].get('isort', {}).get('issues', 0) > 0:
                print("   📦 Run 'isort .' to fix import sorting issues.")
            
            print("   📚 Use 'pre-commit run --all-files' to fix many issues automatically.")
        else:
            print(f"\n🎉 Excellent! No issues found. Your code quality is great!")
        
        print("\n" + "="*80)
    
    def _print_json_report(self):
        """打印JSON格式报告"""
        summary = self.generate_summary()
        
        # 转换问题为可序列化格式
        json_reports = []
        for report in self.reports:
            json_issues = []
            for issue in report.issues:
                json_issues.append({
                    'tool': issue.tool,
                    'file': issue.file,
                    'line': issue.line,
                    'column': issue.column,
                    'severity': issue.severity.value,
                    'code': issue.code,
                    'message': issue.message,
                    'rule': issue.rule
                })
            
            json_reports.append({
                'tool': report.tool,
                'success': report.success,
                'duration': report.duration,
                'issues': json_issues,
                'summary': report.summary
            })
        
        result = {
            'summary': summary,
            'reports': json_reports,
            'timestamp': time.time()
        }
        
        print(json.dumps(result, indent=2))
    
    def save_report(self, output_file: Path, format_type: str = 'json'):
        """保存报告到文件"""
        if format_type == 'json':
            summary = self.generate_summary()
            
            # 转换为可序列化格式
            json_reports = []
            for report in self.reports:
                json_issues = []
                for issue in report.issues:
                    json_issues.append({
                        'tool': issue.tool,
                        'file': issue.file,
                        'line': issue.line,
                        'column': issue.column,
                        'severity': issue.severity.value,
                        'code': issue.code,
                        'message': issue.message,
                        'rule': issue.rule
                    })
                
                json_reports.append({
                    'tool': report.tool,
                    'success': report.success,
                    'duration': report.duration,
                    'issues': json_issues,
                    'summary': report.summary
                })
            
            result = {
                'summary': summary,
                'reports': json_reports,
                'timestamp': time.time()
            }
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        print(f"Report saved to {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Code Quality Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/code_quality.py                        # Run all checks
  python scripts/code_quality.py --tools flake8 mypy    # Run specific tools
  python scripts/code_quality.py --detailed             # Show detailed issues
  python scripts/code_quality.py --format json          # JSON output
        """
    )
    
    parser.add_argument(
        '--tools',
        nargs='+',
        choices=['flake8', 'pylint', 'mypy', 'bandit', 'radon', 'isort', 'black'],
        help='Specific tools to run'
    )
    
    parser.add_argument(
        '--dirs',
        nargs='+',
        default=['modify_multi_attention', 'tests', 'scripts'],
        help='Target directories to check'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file for detailed report'
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format'
    )
    
    parser.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Show detailed issue list'
    )
    
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path.cwd(),
        help='Project root directory'
    )
    
    args = parser.parse_args()
    
    # 检查项目根目录
    if not args.project_root.exists():
        print(f"Error: Project root directory does not exist: {args.project_root}")
        sys.exit(1)
    
    # 创建代码质量检查器
    checker = CodeQualityChecker(args.project_root, args.dirs)
    
    print("🔍 Starting code quality analysis...")
    print(f"📁 Target directories: {', '.join(args.dirs)}")
    
    if args.tools:
        print(f"🔧 Running tools: {', '.join(args.tools)}")
    else:
        print("🔧 Running all available tools")
    
    # 运行检查
    try:
        reports = checker.run_all_checks(args.tools)
        
        # 打印报告
        checker.print_report(detailed=args.detailed, format_type=args.format)
        
        # 保存报告
        if args.output:
            checker.save_report(args.output, format_type='json')
        
        # 生成总结
        summary = checker.generate_summary()
        
        # 根据结果设置退出码
        if summary['total_issues'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error during analysis: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()