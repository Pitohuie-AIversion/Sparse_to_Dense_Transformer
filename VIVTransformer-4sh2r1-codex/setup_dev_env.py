#!/usr/bin/env python3
"""
开发环境设置脚本 - 一键配置完整的开发环境
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional


class DevEnvironmentSetup:
    """开发环境设置类"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.scripts_dir = self.project_root / "scripts"
        self.python_executable = sys.executable
    
    def run_command(self, cmd: List[str], description: str, check: bool = True) -> bool:
        """运行命令并显示结果"""
        print(f"\n🔧 {description}...")
        try:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True, cwd=self.project_root)
            if result.returncode == 0:
                print(f"✅ {description} 完成")
                return True
            else:
                print(f"⚠️ {description} 完成但有警告")
                if result.stdout:
                    print("输出:", result.stdout[:500])
                return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} 失败: {e}")
            if e.stdout:
                print("输出:", e.stdout[:500])
            if e.stderr:
                print("错误:", e.stderr[:500])
            return False
        except FileNotFoundError:
            print(f"❌ {description} 失败: 命令未找到")
            return False
    
    def check_python_version(self) -> bool:
        """检查Python版本"""
        print("\n🐍 检查Python版本...")
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} 版本符合要求")
            return True
        else:
            print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro}，需要Python 3.8+")
            return False
    
    def install_dependencies(self) -> bool:
        """安装项目依赖"""
        print("\n📦 安装项目依赖...")
        
        # 升级pip
        self.run_command([self.python_executable, "-m", "pip", "install", "--upgrade", "pip"], "升级pip")
        
        # 安装基础依赖
        if (self.project_root / "requirements.txt").exists():
            success = self.run_command(
                [self.python_executable, "-m", "pip", "install", "-r", "requirements.txt"],
                "安装requirements.txt依赖"
            )
        else:
            print("⚠️ requirements.txt不存在，跳过基础依赖安装")
            success = True
        
        # 安装开发依赖
        dev_packages = [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "pytest-mock>=3.10.0",
            "bandit>=1.7.0",
            "safety>=2.3.0",
            "pylint>=2.17.0",
            "autopep8>=2.0.0",
            "docformatter>=1.7.0",
            "unimport>=0.15.0",
            "radon>=6.0.0",
            "vulture>=2.7.0",
            "pydocstyle>=6.3.0",
            "xenon>=0.9.0",
            "pre-commit>=3.0.0",
            "matplotlib>=3.5.0",
            "plotly>=5.0.0"
        ]
        
        success &= self.run_command(
            [self.python_executable, "-m", "pip", "install"] + dev_packages,
            "安装开发依赖包"
        )
        
        return success
    
    def setup_pre_commit(self) -> bool:
        """设置pre-commit钩子"""
        print("\n🪝 设置pre-commit钩子...")
        
        if not (self.project_root / ".pre-commit-config.yaml").exists():
            print("⚠️ .pre-commit-config.yaml不存在，跳过pre-commit设置")
            return True
        
        success = self.run_command(
            [self.python_executable, "-m", "pre_commit", "install"],
            "安装pre-commit钩子"
        )
        
        if success:
            success &= self.run_command(
                [self.python_executable, "-m", "pre_commit", "install", "--hook-type", "pre-push"],
                "安装pre-push钩子"
            )
        
        return success
    
    def create_directories(self) -> bool:
        """创建必要的目录结构"""
        print("\n📁 创建目录结构...")
        
        directories = [
            "scripts",
            "tests",
            "docs",
            "reports",
            "logs",
            ".github/workflows"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ 创建目录: {directory}")
        
        return True
    
    def verify_scripts(self) -> bool:
        """验证关键脚本是否存在"""
        print("\n📋 验证开发脚本...")
        
        required_scripts = [
            "code_quality.py",
            "format_code.py",
            "auto_fix.py",
            "quality_gate.py",
            "quality_dashboard.py"
        ]
        
        missing_scripts = []
        for script in required_scripts:
            script_path = self.scripts_dir / script
            if script_path.exists():
                print(f"✅ {script} 存在")
            else:
                print(f"❌ {script} 缺失")
                missing_scripts.append(script)
        
        if missing_scripts:
            print(f"\n⚠️ 缺失脚本: {', '.join(missing_scripts)}")
            print("请确保所有开发脚本都已创建")
            return False
        
        return True
    
    def run_initial_checks(self) -> bool:
        """运行初始检查"""
        print("\n🔍 运行初始检查...")
        
        # 运行代码格式化检查
        success = self.run_command(
            [self.python_executable, "scripts/format_code.py", "--check", "--target-dirs", "modify_multi_attention"],
            "代码格式化检查",
            check=False
        )
        
        # 运行基础质量检查
        success &= self.run_command(
            [self.python_executable, "scripts/code_quality.py", "--tools", "flake8,mypy", "--target-dirs", "modify_multi_attention"],
            "基础代码质量检查",
            check=False
        )
        
        return success
    
    def create_development_scripts(self):
        """创建开发辅助脚本"""
        print("\n📝 创建开发辅助脚本...")
        
        scripts_dir = Path("scripts")
        scripts_dir.mkdir(exist_ok=True)
        
        # 快速测试脚本
        quick_test_script = scripts_dir / "quick_test.py"
        quick_test_content = '''#!/usr/bin/env python3
"""
快速测试脚本 - 运行核心功能的快速测试
"""

import sys
import subprocess
from pathlib import Path

def run_quick_tests():
    """运行快速测试"""
    print("🧪 运行快速测试...")
    
    # 检查导入
    try:
        import modify_multi_attention
        print("✅ 核心模块导入成功")
    except ImportError as e:
        print(f"❌ 核心模块导入失败: {e}")
        return False
    
    # 运行基础测试
    test_files = [
        "tests/test_basic.py",
        "tests/test_attention.py"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            result = subprocess.run([sys.executable, "-m", "pytest", test_file, "-v"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {test_file} 通过")
            else:
                print(f"❌ {test_file} 失败")
                print(result.stdout)
                return False
    
    print("🎉 所有快速测试通过!")
    return True

if __name__ == "__main__":
    success = run_quick_tests()
    sys.exit(0 if success else 1)
'''
        
        with open(quick_test_script, 'w', encoding='utf-8') as f:
            f.write(quick_test_content)
        
        # 综合开发工具脚本
        dev_tools_script = scripts_dir / "dev_tools.py"
        dev_tools_content = '''#!/usr/bin/env python3
"""
综合开发工具脚本 - 一键运行所有开发工具
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        if e.stdout:
            print("输出:", e.stdout)
        if e.stderr:
            print("错误:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="综合开发工具")
    parser.add_argument("--format", action="store_true", help="格式化代码")
    parser.add_argument("--quality", action="store_true", help="代码质量检查")
    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--fix", action="store_true", help="自动修复问题")
    parser.add_argument("--dashboard", action="store_true", help="生成质量仪表板")
    parser.add_argument("--all", action="store_true", help="运行所有工具")
    
    args = parser.parse_args()
    
    if not any([args.format, args.quality, args.test, args.fix, args.dashboard, args.all]):
        args.all = True
    
    success = True
    
    if args.all or args.fix:
        success &= run_command(
            [sys.executable, "scripts/auto_fix.py", "--target-dirs", "modify_multi_attention", "tests", "scripts"],
            "自动修复代码问题"
        )
    
    if args.all or args.format:
        success &= run_command(
            [sys.executable, "scripts/format_code.py", "--target-dirs", "modify_multi_attention", "tests", "scripts"],
            "代码格式化检查"
        )
    
    if args.all or args.quality:
        success &= run_command(
            [sys.executable, "scripts/code_quality.py", "--tools", "all", "--show-details"],
            "代码质量检查"
        )
    
    if args.all or args.test:
        success &= run_command(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--cov=modify_multi_attention"],
            "运行测试套件"
        )
    
    if args.all or args.dashboard:
        success &= run_command(
            [sys.executable, "scripts/quality_dashboard.py", "--target-dirs", "modify_multi_attention", "tests", "scripts"],
            "生成质量仪表板"
        )
    
    if success:
        print("\n🎉 所有开发工具执行成功!")
    else:
        print("\n❌ 部分工具执行失败，请检查上述错误信息")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
        
        with open(dev_tools_script, 'w', encoding='utf-8') as f:
            f.write(dev_tools_content)
        
        print(f"✅ 开发脚本已创建在 {scripts_dir}/")
        print("   - quick_test.py: 快速测试")
        print("   - dev_tools.py: 综合开发工具")
        print("   - code_quality.py: 代码质量检查")
        print("   - format_code.py: 代码格式化")
        print("   - auto_fix.py: 自动修复")
        print("   - quality_gate.py: 质量门禁")
        print("   - quality_dashboard.py: 质量仪表板")
    
    def setup_complete_environment(self) -> bool:
        """完整的环境设置流程"""
        print("🚀 开始设置VIVTransformer开发环境...")
        print(f"项目根目录: {self.project_root}")
        
        steps = [
            ("检查Python版本", self.check_python_version),
            ("创建目录结构", self.create_directories),
            ("安装依赖包", self.install_dependencies),
            ("设置pre-commit钩子", self.setup_pre_commit),
            ("创建开发脚本", self.create_development_scripts),
            ("验证脚本完整性", self.verify_scripts),
            ("运行初始检查", self.run_initial_checks)
        ]
        
        failed_steps = []
        for step_name, step_func in steps:
            print(f"\n{'='*50}")
            print(f"步骤: {step_name}")
            print(f"{'='*50}")
            
            try:
                if not step_func():
                    failed_steps.append(step_name)
                    print(f"⚠️ 步骤 '{step_name}' 未完全成功")
            except Exception as e:
                failed_steps.append(step_name)
                print(f"❌ 步骤 '{step_name}' 执行失败: {e}")
        
        print(f"\n{'='*60}")
        print("🎯 环境设置完成总结")
        print(f"{'='*60}")
        
        if not failed_steps:
            print("🎉 所有步骤都成功完成!")
            print("\n📋 下一步操作建议:")
            print("1. 运行 'python scripts/dev_tools.py --all' 进行全面检查")
            print("2. 运行 'python scripts/quick_test.py' 进行快速测试")
            print("3. 运行 'python scripts/quality_dashboard.py' 生成质量报告")
            print("4. 使用 'pre-commit run --all-files' 验证pre-commit配置")
            return True
        else:
            print(f"⚠️ 以下步骤需要手动检查: {', '.join(failed_steps)}")
            print("\n🔧 建议的修复操作:")
            for step in failed_steps:
                if "依赖" in step:
                    print(f"- {step}: 检查网络连接，手动安装缺失的包")
                elif "pre-commit" in step:
                    print(f"- {step}: 确保.pre-commit-config.yaml文件存在且格式正确")
                elif "脚本" in step:
                    print(f"- {step}: 检查scripts目录下的脚本文件是否完整")
                else:
                    print(f"- {step}: 查看上述错误信息进行排查")
            return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VIVTransformer开发环境设置")
    parser.add_argument("--project-root", type=Path, help="项目根目录路径")
    parser.add_argument("--skip-deps", action="store_true", help="跳过依赖安装")
    parser.add_argument("--skip-precommit", action="store_true", help="跳过pre-commit设置")
    
    args = parser.parse_args()
    
    setup = DevEnvironmentSetup(args.project_root)
    
    if args.skip_deps:
        setup.install_dependencies = lambda: True
    if args.skip_precommit:
        setup.setup_pre_commit = lambda: True
    
    success = setup.setup_complete_environment()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()