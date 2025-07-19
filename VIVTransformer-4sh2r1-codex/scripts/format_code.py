#!/usr/bin/env python3
"""
代码格式化脚本

这个脚本提供自动化的代码格式化功能，包括：
- Black 代码格式化
- isort 导入排序
- 自动修复简单的代码质量问题
- 批量处理多个目录

使用方法:
    python scripts/format_code.py                    # 格式化所有目录
    python scripts/format_code.py --dirs src tests   # 格式化指定目录
    python scripts/format_code.py --check            # 仅检查，不修改
    python scripts/format_code.py --aggressive       # 激进模式，包含更多修复
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple


class CodeFormatter:
    """代码格式化器"""
    
    def __init__(self, target_dirs: List[str], python_executable: str = 'python'):
        self.target_dirs = target_dirs
        self.python_executable = python_executable
        self.results = []
    
    def run_command(self, cmd: List[str]) -> Tuple[int, str, str]:
        """运行命令并返回结果"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)
    
    def format_with_black(self, check_only: bool = False) -> Dict:
        """使用 Black 格式化代码"""
        print("🎨 Running Black formatter...")
        start_time = time.time()
        
        cmd = [
            self.python_executable, '-m', 'black',
            '--line-length', '88',
            '--target-version', 'py38'
        ]
        
        if check_only:
            cmd.append('--check')
            cmd.append('--diff')
        
        cmd.extend(self.target_dirs)
        
        returncode, stdout, stderr = self.run_command(cmd)
        duration = time.time() - start_time
        
        # 解析输出
        if check_only:
            if returncode == 0:
                status = "✅ All files are properly formatted"
                files_changed = 0
            else:
                # 计算需要格式化的文件数
                files_changed = stdout.count('would reformat')
                status = f"❌ {files_changed} files need formatting"
        else:
            if returncode == 0:
                files_changed = stdout.count('reformatted')
                if files_changed > 0:
                    status = f"✅ Reformatted {files_changed} files"
                else:
                    status = "✅ All files already formatted"
            else:
                status = f"❌ Black failed: {stderr}"
                files_changed = 0
        
        result = {
            'tool': 'black',
            'success': returncode == 0 or (check_only and returncode == 1),
            'duration': duration,
            'files_changed': files_changed,
            'status': status,
            'output': stdout,
            'errors': stderr
        }
        
        print(f"   {status} ({duration:.2f}s)")
        return result
    
    def sort_imports_with_isort(self, check_only: bool = False) -> Dict:
        """使用 isort 排序导入"""
        print("📦 Running isort import sorter...")
        start_time = time.time()
        
        cmd = [
            self.python_executable, '-m', 'isort',
            '--profile', 'black',
            '--line-length', '88',
            '--multi-line', '3',
            '--trailing-comma'
        ]
        
        if check_only:
            cmd.append('--check-only')
            cmd.append('--diff')
        
        cmd.extend(self.target_dirs)
        
        returncode, stdout, stderr = self.run_command(cmd)
        duration = time.time() - start_time
        
        # 解析输出
        if check_only:
            if returncode == 0:
                status = "✅ All imports are properly sorted"
                files_changed = 0
            else:
                files_changed = stdout.count('would reformat')
                status = f"❌ {files_changed} files need import sorting"
        else:
            if returncode == 0:
                files_changed = stdout.count('Fixing')
                if files_changed > 0:
                    status = f"✅ Fixed imports in {files_changed} files"
                else:
                    status = "✅ All imports already sorted"
            else:
                status = f"❌ isort failed: {stderr}"
                files_changed = 0
        
        result = {
            'tool': 'isort',
            'success': returncode == 0 or (check_only and returncode == 1),
            'duration': duration,
            'files_changed': files_changed,
            'status': status,
            'output': stdout,
            'errors': stderr
        }
        
        print(f"   {status} ({duration:.2f}s)")
        return result
    
    def fix_with_autopep8(self, check_only: bool = False) -> Dict:
        """使用 autopep8 修复 PEP8 问题"""
        print("🔧 Running autopep8 fixer...")
        start_time = time.time()
        
        cmd = [
            self.python_executable, '-m', 'autopep8',
            '--max-line-length', '88',
            '--aggressive',
            '--aggressive'
        ]
        
        if check_only:
            cmd.append('--diff')
        else:
            cmd.append('--in-place')
        
        cmd.append('--recursive')
        cmd.extend(self.target_dirs)
        
        returncode, stdout, stderr = self.run_command(cmd)
        duration = time.time() - start_time
        
        # 解析输出
        if check_only:
            if stdout.strip():
                files_changed = stdout.count('---')
                status = f"❌ {files_changed} files need PEP8 fixes"
            else:
                status = "✅ All files are PEP8 compliant"
                files_changed = 0
        else:
            if returncode == 0:
                # autopep8 不输出修改的文件数，估算
                files_changed = len([f for f in Path().rglob('*.py') if any(d in str(f) for d in self.target_dirs)])
                status = f"✅ Applied PEP8 fixes to {files_changed} files"
            else:
                status = f"❌ autopep8 failed: {stderr}"
                files_changed = 0
        
        result = {
            'tool': 'autopep8',
            'success': returncode == 0,
            'duration': duration,
            'files_changed': files_changed,
            'status': status,
            'output': stdout,
            'errors': stderr
        }
        
        print(f"   {status} ({duration:.2f}s)")
        return result
    
    def remove_unused_imports(self, check_only: bool = False) -> Dict:
        """使用 unimport 移除未使用的导入"""
        print("🧹 Running unimport to remove unused imports...")
        start_time = time.time()
        
        cmd = [
            self.python_executable, '-m', 'unimport',
            '--remove-all'
        ]
        
        if check_only:
            cmd.append('--check')
        
        cmd.extend(self.target_dirs)
        
        returncode, stdout, stderr = self.run_command(cmd)
        duration = time.time() - start_time
        
        # 解析输出
        if returncode == 0:
            if check_only:
                status = "✅ No unused imports found"
                files_changed = 0
            else:
                files_changed = stdout.count('removed')
                if files_changed > 0:
                    status = f"✅ Removed unused imports from {files_changed} files"
                else:
                    status = "✅ No unused imports to remove"
        else:
            if 'unused import' in stdout:
                files_changed = stdout.count('unused import')
                status = f"❌ {files_changed} unused imports found"
            else:
                status = f"❌ unimport failed: {stderr}"
                files_changed = 0
        
        result = {
            'tool': 'unimport',
            'success': returncode == 0,
            'duration': duration,
            'files_changed': files_changed,
            'status': status,
            'output': stdout,
            'errors': stderr
        }
        
        print(f"   {status} ({duration:.2f}s)")
        return result
    
    def format_docstrings(self, check_only: bool = False) -> Dict:
        """使用 docformatter 格式化文档字符串"""
        print("📝 Running docformatter for docstrings...")
        start_time = time.time()
        
        cmd = [
            self.python_executable, '-m', 'docformatter',
            '--wrap-summaries', '88',
            '--wrap-descriptions', '88',
            '--make-summary-multi-line'
        ]
        
        if check_only:
            cmd.append('--check')
        else:
            cmd.append('--in-place')
        
        cmd.append('--recursive')
        cmd.extend(self.target_dirs)
        
        returncode, stdout, stderr = self.run_command(cmd)
        duration = time.time() - start_time
        
        # 解析输出
        if returncode == 0:
            if check_only:
                status = "✅ All docstrings are properly formatted"
                files_changed = 0
            else:
                files_changed = stdout.count('fixed')
                if files_changed > 0:
                    status = f"✅ Fixed docstrings in {files_changed} files"
                else:
                    status = "✅ All docstrings already formatted"
        else:
            status = f"❌ docformatter failed: {stderr}"
            files_changed = 0
        
        result = {
            'tool': 'docformatter',
            'success': returncode == 0,
            'duration': duration,
            'files_changed': files_changed,
            'status': status,
            'output': stdout,
            'errors': stderr
        }
        
        print(f"   {status} ({duration:.2f}s)")
        return result
    
    def run_all_formatters(self, check_only: bool = False, aggressive: bool = False) -> List[Dict]:
        """运行所有格式化工具"""
        print("🚀 Starting code formatting...")
        print(f"📁 Target directories: {', '.join(self.target_dirs)}")
        print(f"🔍 Mode: {'Check only' if check_only else 'Format and fix'}")
        
        if aggressive:
            print("⚡ Aggressive mode enabled")
        
        results = []
        
        # 基础格式化（总是运行）
        results.append(self.format_with_black(check_only))
        results.append(self.sort_imports_with_isort(check_only))
        
        # 激进模式的额外工具
        if aggressive:
            results.append(self.fix_with_autopep8(check_only))
            results.append(self.remove_unused_imports(check_only))
            results.append(self.format_docstrings(check_only))
        
        self.results = results
        return results
    
    def print_summary(self):
        """打印格式化总结"""
        print("\n" + "="*80)
        print("📊 CODE FORMATTING SUMMARY")
        print("="*80)
        
        total_tools = len(self.results)
        successful_tools = sum(1 for r in self.results if r['success'])
        total_files_changed = sum(r['files_changed'] for r in self.results)
        total_duration = sum(r['duration'] for r in self.results)
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Tools run: {total_tools}")
        print(f"   Successful: {successful_tools}")
        print(f"   Files modified: {total_files_changed}")
        print(f"   Total duration: {total_duration:.2f}s")
        
        print(f"\n🔧 Tool Results:")
        for result in self.results:
            status_icon = "✅" if result['success'] else "❌"
            tool = result['tool']
            duration = result['duration']
            files = result['files_changed']
            print(f"   {status_icon} {tool:<12} ({duration:>5.2f}s) - {files} files")
        
        print(f"\n📋 Detailed Status:")
        for result in self.results:
            print(f"   • {result['status']}")
        
        if successful_tools == total_tools and total_files_changed == 0:
            print(f"\n🎉 Perfect! Your code is already well-formatted!")
        elif successful_tools == total_tools:
            print(f"\n✨ Formatting completed successfully! {total_files_changed} files were improved.")
        else:
            failed_tools = [r['tool'] for r in self.results if not r['success']]
            print(f"\n⚠️  Some tools failed: {', '.join(failed_tools)}")
            print("   Please check the error messages above.")
        
        print("\n" + "="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Automatic code formatting and fixing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/format_code.py                    # Format all directories
  python scripts/format_code.py --dirs src tests   # Format specific directories
  python scripts/format_code.py --check            # Check only, don't modify
  python scripts/format_code.py --aggressive       # Aggressive mode with more tools
        """
    )
    
    parser.add_argument(
        '--dirs',
        nargs='+',
        default=['modify_multi_attention/', 'scripts/', 'tests/'],
        help='Target directories to format (default: modify_multi_attention/ scripts/ tests/)'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check formatting without making changes'
    )
    
    parser.add_argument(
        '--aggressive',
        action='store_true',
        help='Use aggressive mode with additional formatting tools'
    )
    
    parser.add_argument(
        '--python',
        default='python',
        help='Python executable to use (default: python)'
    )
    
    parser.add_argument(
        '--fail-on-changes',
        action='store_true',
        help='Exit with error code if changes are needed (useful for CI)'
    )
    
    args = parser.parse_args()
    
    # 创建格式化器
    formatter = CodeFormatter(
        target_dirs=args.dirs,
        python_executable=args.python
    )
    
    try:
        # 运行格式化
        results = formatter.run_all_formatters(
            check_only=args.check,
            aggressive=args.aggressive
        )
        
        # 打印总结
        formatter.print_summary()
        
        # 设置退出码
        if args.fail_on_changes:
            total_changes = sum(r['files_changed'] for r in results)
            if total_changes > 0:
                print(f"\n❌ Exiting with error code due to {total_changes} files needing changes.")
                sys.exit(1)
        
        # 检查是否有工具失败
        failed_tools = [r for r in results if not r['success']]
        if failed_tools:
            print(f"\n❌ Some formatting tools failed.")
            sys.exit(1)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n⚠️  Formatting interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Error during formatting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()