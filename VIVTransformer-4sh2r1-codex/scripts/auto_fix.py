#!/usr/bin/env python3
"""
自动修复脚本 - 自动修复常见的代码质量问题

这个脚本可以自动修复一些常见的代码质量问题，包括：
- 代码格式化 (Black, autopep8)
- 导入排序 (isort)
- 移除未使用的导入 (unimport)
- 文档字符串格式化 (docformatter)
- 移除尾随空白
- 修复简单的语法问题

作者: VIVTransformer Team
日期: 2024
"""

import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


class AutoFixer:
    """自动修复工具"""
    
    def __init__(self, 
                 target_dirs: List[str],
                 python_executable: str = "python",
                 dry_run: bool = False):
        self.target_dirs = target_dirs
        self.python_executable = python_executable
        self.dry_run = dry_run
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        if dry_run:
            self.logger.info("运行在预览模式，不会实际修改文件")
    
    def run_command(self, cmd: List[str], cwd: Optional[str] = None) -> Tuple[bool, str, str]:
        """运行命令并返回结果"""
        try:
            if self.dry_run:
                self.logger.info(f"[预览] 将要运行: {' '.join(cmd)}")
                return True, "预览模式 - 未实际执行", ""
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd or Path.cwd()
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except Exception as e:
            return False, "", str(e)
    
    def get_python_files(self) -> List[Path]:
        """获取所有Python文件"""
        python_files = []
        
        for target_dir in self.target_dirs:
            target_path = Path(target_dir)
            if not target_path.exists():
                self.logger.warning(f"目录不存在: {target_dir}")
                continue
            
            if target_path.is_file() and target_path.suffix == '.py':
                python_files.append(target_path)
            else:
                python_files.extend(target_path.rglob('*.py'))
        
        # 过滤掉一些不需要处理的文件
        excluded_patterns = [
            '__pycache__',
            '.git',
            '.tox',
            '.venv',
            'venv',
            'build',
            'dist',
            '.pytest_cache'
        ]
        
        filtered_files = []
        for file_path in python_files:
            if not any(pattern in str(file_path) for pattern in excluded_patterns):
                filtered_files.append(file_path)
        
        return filtered_files
    
    def fix_trailing_whitespace(self, files: List[Path]) -> int:
        """移除尾随空白"""
        self.logger.info("修复尾随空白...")
        fixed_count = 0
        
        for file_path in files:
            try:
                if self.dry_run:
                    self.logger.info(f"[预览] 将检查尾随空白: {file_path}")
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 移除尾随空白
                lines = content.splitlines()
                fixed_lines = [line.rstrip() for line in lines]
                new_content = '\n'.join(fixed_lines)
                
                # 确保文件以换行符结尾
                if new_content and not new_content.endswith('\n'):
                    new_content += '\n'
                
                if content != new_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    fixed_count += 1
                    self.logger.info(f"修复尾随空白: {file_path}")
                    
            except Exception as e:
                self.logger.error(f"处理文件 {file_path} 时出错: {e}")
        
        return fixed_count
    
    def fix_line_endings(self, files: List[Path]) -> int:
        """统一行结束符为LF"""
        self.logger.info("统一行结束符...")
        fixed_count = 0
        
        for file_path in files:
            try:
                if self.dry_run:
                    self.logger.info(f"[预览] 将检查行结束符: {file_path}")
                    continue
                
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                # 将CRLF和CR转换为LF
                if b'\r\n' in content or b'\r' in content:
                    new_content = content.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
                    
                    with open(file_path, 'wb') as f:
                        f.write(new_content)
                    fixed_count += 1
                    self.logger.info(f"修复行结束符: {file_path}")
                    
            except Exception as e:
                self.logger.error(f"处理文件 {file_path} 时出错: {e}")
        
        return fixed_count
    
    def fix_encoding_declarations(self, files: List[Path]) -> int:
        """移除不必要的编码声明 (Python 3默认UTF-8)"""
        self.logger.info("检查编码声明...")
        fixed_count = 0
        
        encoding_pattern = re.compile(r'^\s*#.*?coding[:=]\s*([-\w.]+)', re.MULTILINE)
        
        for file_path in files:
            try:
                if self.dry_run:
                    self.logger.info(f"[预览] 将检查编码声明: {file_path}")
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否有编码声明
                if encoding_pattern.search(content):
                    lines = content.splitlines()
                    new_lines = []
                    
                    for i, line in enumerate(lines):
                        # 保留shebang行
                        if i == 0 and line.startswith('#!'):
                            new_lines.append(line)
                        # 移除编码声明行
                        elif not encoding_pattern.match(line):
                            new_lines.append(line)
                    
                    new_content = '\n'.join(new_lines) + '\n'
                    
                    if content != new_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        fixed_count += 1
                        self.logger.info(f"移除编码声明: {file_path}")
                        
            except Exception as e:
                self.logger.error(f"处理文件 {file_path} 时出错: {e}")
        
        return fixed_count
    
    def run_black(self) -> bool:
        """运行Black代码格式化"""
        self.logger.info("运行Black代码格式化...")
        
        cmd = [self.python_executable, '-m', 'black']
        if self.dry_run:
            cmd.append('--check')
        cmd.extend(self.target_dirs)
        
        success, stdout, stderr = self.run_command(cmd)
        
        if success:
            self.logger.info("Black格式化完成")
        else:
            self.logger.error(f"Black格式化失败: {stderr}")
        
        return success
    
    def run_isort(self) -> bool:
        """运行isort导入排序"""
        self.logger.info("运行isort导入排序...")
        
        cmd = [self.python_executable, '-m', 'isort']
        if self.dry_run:
            cmd.append('--check-only')
        cmd.extend(self.target_dirs)
        
        success, stdout, stderr = self.run_command(cmd)
        
        if success:
            self.logger.info("isort导入排序完成")
        else:
            self.logger.error(f"isort导入排序失败: {stderr}")
        
        return success
    
    def run_autopep8(self) -> bool:
        """运行autopep8修复PEP8问题"""
        self.logger.info("运行autopep8修复PEP8问题...")
        
        cmd = [self.python_executable, '-m', 'autopep8']
        if not self.dry_run:
            cmd.extend(['--in-place', '--recursive'])
        else:
            cmd.append('--diff')
        
        cmd.extend(['--aggressive', '--aggressive'])
        cmd.extend(self.target_dirs)
        
        success, stdout, stderr = self.run_command(cmd)
        
        if success:
            self.logger.info("autopep8修复完成")
        else:
            self.logger.error(f"autopep8修复失败: {stderr}")
        
        return success
    
    def run_unimport(self) -> bool:
        """运行unimport移除未使用的导入"""
        self.logger.info("运行unimport移除未使用的导入...")
        
        cmd = [self.python_executable, '-m', 'unimport']
        if not self.dry_run:
            cmd.append('--remove-all')
        else:
            cmd.append('--check')
        
        for target_dir in self.target_dirs:
            cmd.extend(['--source', target_dir])
        
        success, stdout, stderr = self.run_command(cmd)
        
        if success:
            self.logger.info("unimport清理完成")
        else:
            self.logger.warning(f"unimport清理警告: {stderr}")
            # unimport有时会返回非零退出码但仍然成功
            success = True
        
        return success
    
    def run_docformatter(self) -> bool:
        """运行docformatter格式化文档字符串"""
        self.logger.info("运行docformatter格式化文档字符串...")
        
        cmd = [self.python_executable, '-m', 'docformatter']
        if not self.dry_run:
            cmd.append('--in-place')
        else:
            cmd.append('--check')
        
        cmd.extend([
            '--wrap-summaries', '88',
            '--wrap-descriptions', '88',
            '--recursive'
        ])
        cmd.extend(self.target_dirs)
        
        success, stdout, stderr = self.run_command(cmd)
        
        if success:
            self.logger.info("docformatter格式化完成")
        else:
            self.logger.error(f"docformatter格式化失败: {stderr}")
        
        return success
    
    def run_all_fixes(self) -> dict:
        """运行所有修复"""
        self.logger.info("开始自动修复...")
        
        results = {}
        
        # 获取Python文件
        python_files = self.get_python_files()
        self.logger.info(f"找到 {len(python_files)} 个Python文件")
        
        # 1. 基础文件修复
        results['trailing_whitespace'] = self.fix_trailing_whitespace(python_files)
        results['line_endings'] = self.fix_line_endings(python_files)
        results['encoding_declarations'] = self.fix_encoding_declarations(python_files)
        
        # 2. 代码格式化工具
        results['unimport'] = self.run_unimport()
        results['isort'] = self.run_isort()
        results['autopep8'] = self.run_autopep8()
        results['black'] = self.run_black()
        results['docformatter'] = self.run_docformatter()
        
        return results
    
    def generate_report(self, results: dict) -> str:
        """生成修复报告"""
        report_lines = [
            "# 自动修复报告",
            "",
            f"## 修复模式: {'预览模式' if self.dry_run else '实际修复'}",
            ""
        ]
        
        # 文件修复统计
        report_lines.extend([
            "## 文件修复统计",
            "",
            f"- 修复尾随空白: {results.get('trailing_whitespace', 0)} 个文件",
            f"- 修复行结束符: {results.get('line_endings', 0)} 个文件",
            f"- 移除编码声明: {results.get('encoding_declarations', 0)} 个文件",
            ""
        ])
        
        # 工具执行结果
        report_lines.extend([
            "## 工具执行结果",
            ""
        ])
        
        tools = {
            'unimport': '移除未使用导入',
            'isort': '导入排序',
            'autopep8': 'PEP8修复',
            'black': '代码格式化',
            'docformatter': '文档字符串格式化'
        }
        
        for tool, description in tools.items():
            status = "✅ 成功" if results.get(tool, False) else "❌ 失败"
            report_lines.append(f"- {description}: {status}")
        
        report_lines.extend([
            "",
            "## 建议",
            "",
            "1. 运行代码质量检查确认修复效果",
            "2. 运行测试确保功能正常",
            "3. 检查Git差异确认修改内容",
            "4. 考虑配置pre-commit钩子防止问题再次出现"
        ])
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="自动修复代码质量问题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --target-dirs modify_multi_attention tests
  %(prog)s --dry-run --target-dirs modify_multi_attention
  %(prog)s --tools black isort --target-dirs .
        """
    )
    
    parser.add_argument(
        '--target-dirs',
        nargs='+',
        default=['modify_multi_attention'],
        help='要修复的目标目录 (默认: modify_multi_attention)'
    )
    
    parser.add_argument(
        '--python',
        default='python',
        help='Python解释器路径 (默认: python)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='预览模式，不实际修改文件'
    )
    
    parser.add_argument(
        '--tools',
        nargs='+',
        choices=['black', 'isort', 'autopep8', 'unimport', 'docformatter', 'basic'],
        help='指定要运行的修复工具'
    )
    
    parser.add_argument(
        '--output',
        help='输出报告文件路径'
    )
    
    parser.add_argument(
        '--aggressive',
        action='store_true',
        help='使用激进模式 (更多修复)'
    )
    
    args = parser.parse_args()
    
    # 创建自动修复器
    fixer = AutoFixer(
        target_dirs=args.target_dirs,
        python_executable=args.python,
        dry_run=args.dry_run
    )
    
    try:
        if args.tools:
            # 运行指定工具
            results = {}
            python_files = fixer.get_python_files()
            
            if 'basic' in args.tools:
                results['trailing_whitespace'] = fixer.fix_trailing_whitespace(python_files)
                results['line_endings'] = fixer.fix_line_endings(python_files)
                results['encoding_declarations'] = fixer.fix_encoding_declarations(python_files)
            
            if 'unimport' in args.tools:
                results['unimport'] = fixer.run_unimport()
            if 'isort' in args.tools:
                results['isort'] = fixer.run_isort()
            if 'autopep8' in args.tools:
                results['autopep8'] = fixer.run_autopep8()
            if 'black' in args.tools:
                results['black'] = fixer.run_black()
            if 'docformatter' in args.tools:
                results['docformatter'] = fixer.run_docformatter()
        else:
            # 运行所有修复
            results = fixer.run_all_fixes()
        
        # 生成报告
        report = fixer.generate_report(results)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"报告已保存到: {args.output}")
        else:
            print(report)
        
        # 检查是否有失败的工具
        failed_tools = [tool for tool, success in results.items() 
                       if isinstance(success, bool) and not success]
        
        if failed_tools:
            print(f"\n警告: 以下工具执行失败: {', '.join(failed_tools)}")
            sys.exit(1)
        else:
            print("\n✅ 所有修复工具执行成功!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n修复被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"修复过程中发生错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()