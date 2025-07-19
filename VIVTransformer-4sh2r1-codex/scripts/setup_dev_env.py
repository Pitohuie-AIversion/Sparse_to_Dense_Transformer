#!/usr/bin/env python3
"""开发环境设置脚本

自动化配置VIVTransformer开发环境，包括依赖安装、工具配置等
"""

import argparse
import subprocess
import sys
import os
import platform
from pathlib import Path
import json
import shutil
from typing import List, Dict, Optional


class DevEnvironmentSetup:
    """开发环境设置器"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.python_executable = sys.executable
        self.system = platform.system().lower()
        
    def run_command(self, cmd: List[str], cwd: Path = None, check: bool = True) -> subprocess.CompletedProcess:
        """运行命令"""
        if cwd is None:
            cwd = self.project_root
            
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=check
            )
            if result.stdout:
                print(result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
            if check:
                raise
            return e
    
    def check_python_version(self) -> bool:
        """检查Python版本"""
        print("\n=== Checking Python Version ===")
        
        version = sys.version_info
        print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major != 3 or version.minor < 8:
            print("❌ Python 3.8+ is required")
            return False
        
        print("✅ Python version is compatible")
        return True
    
    def check_git(self) -> bool:
        """检查Git是否安装"""
        print("\n=== Checking Git ===")
        
        try:
            result = self.run_command(['git', '--version'])
            print(f"✅ Git is available: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Git is not installed or not in PATH")
            return False
    
    def setup_virtual_environment(self, venv_name: str = 'venv') -> bool:
        """设置虚拟环境"""
        print(f"\n=== Setting up Virtual Environment ({venv_name}) ===")
        
        venv_path = self.project_root / venv_name
        
        if venv_path.exists():
            print(f"Virtual environment already exists at {venv_path}")
            response = input("Do you want to recreate it? (y/N): ")
            if response.lower() == 'y':
                print("Removing existing virtual environment...")
                shutil.rmtree(venv_path)
            else:
                print("Using existing virtual environment")
                return True
        
        try:
            # 创建虚拟环境
            self.run_command([self.python_executable, '-m', 'venv', str(venv_path)])
            print(f"✅ Virtual environment created at {venv_path}")
            
            # 激活脚本路径
            if self.system == 'windows':
                activate_script = venv_path / 'Scripts' / 'activate.bat'
                pip_executable = venv_path / 'Scripts' / 'pip.exe'
            else:
                activate_script = venv_path / 'bin' / 'activate'
                pip_executable = venv_path / 'bin' / 'pip'
            
            print(f"To activate the virtual environment, run:")
            if self.system == 'windows':
                print(f"  {activate_script}")
            else:
                print(f"  source {activate_script}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def install_dependencies(self, dev: bool = True, gpu: bool = False) -> bool:
        """安装依赖"""
        print("\n=== Installing Dependencies ===")
        
        try:
            # 升级pip
            self.run_command([self.python_executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
            
            # 安装PyTorch
            if gpu and self.system != 'darwin':  # macOS不支持CUDA
                print("Installing PyTorch with CUDA support...")
                torch_cmd = [
                    self.python_executable, '-m', 'pip', 'install',
                    'torch', 'torchvision', 'torchaudio',
                    '--index-url', 'https://download.pytorch.org/whl/cu118'
                ]
            else:
                print("Installing PyTorch (CPU version)...")
                torch_cmd = [
                    self.python_executable, '-m', 'pip', 'install',
                    'torch', 'torchvision', 'torchaudio',
                    '--index-url', 'https://download.pytorch.org/whl/cpu'
                ]
            
            self.run_command(torch_cmd)
            
            # 安装项目依赖
            print("Installing project dependencies...")
            self.run_command([self.python_executable, '-m', 'pip', 'install', '-e', '.'])
            
            # 安装开发依赖
            if dev:
                print("Installing development dependencies...")
                self.run_command([
                    self.python_executable, '-m', 'pip', 'install',
                    '-r', 'requirements-dev.txt'
                ])
            
            print("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def setup_pre_commit(self) -> bool:
        """设置pre-commit钩子"""
        print("\n=== Setting up Pre-commit Hooks ===")
        
        try:
            # 安装pre-commit钩子
            self.run_command([self.python_executable, '-m', 'pre_commit', 'install'])
            
            # 运行一次检查
            print("Running pre-commit on all files (this may take a while)...")
            result = self.run_command(
                [self.python_executable, '-m', 'pre_commit', 'run', '--all-files'],
                check=False
            )
            
            if result.returncode == 0:
                print("✅ Pre-commit hooks installed and all checks passed")
            else:
                print("⚠️ Pre-commit hooks installed but some checks failed")
                print("This is normal for the first run. Files have been auto-fixed.")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to setup pre-commit: {e}")
            return False
    
    def setup_ide_config(self) -> bool:
        """设置IDE配置文件"""
        print("\n=== Setting up IDE Configuration ===")
        
        # VS Code配置
        vscode_dir = self.project_root / '.vscode'
        vscode_dir.mkdir(exist_ok=True)
        
        # settings.json
        settings = {
            "python.defaultInterpreterPath": "./venv/bin/python" if self.system != 'windows' else "./venv/Scripts/python.exe",
            "python.linting.enabled": True,
            "python.linting.pylintEnabled": True,
            "python.linting.flake8Enabled": True,
            "python.linting.mypyEnabled": True,
            "python.formatting.provider": "black",
            "python.sortImports.args": ["--profile", "black"],
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {
                "source.organizeImports": True
            },
            "python.testing.pytestEnabled": True,
            "python.testing.unittestEnabled": False,
            "python.testing.pytestArgs": [
                "tests"
            ],
            "files.exclude": {
                "**/__pycache__": True,
                "**/*.pyc": True,
                "**/.pytest_cache": True,
                "**/.mypy_cache": True,
                "**/.coverage": True,
                "**/htmlcov": True
            }
        }
        
        with open(vscode_dir / 'settings.json', 'w') as f:
            json.dump(settings, f, indent=2)
        
        # launch.json
        launch_config = {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Python: Current File",
                    "type": "python",
                    "request": "launch",
                    "program": "${file}",
                    "console": "integratedTerminal",
                    "justMyCode": True
                },
                {
                    "name": "Python: Run Tests",
                    "type": "python",
                    "request": "launch",
                    "module": "pytest",
                    "args": ["tests/", "-v"],
                    "console": "integratedTerminal",
                    "justMyCode": False
                },
                {
                    "name": "Python: Debug Tests",
                    "type": "python",
                    "request": "launch",
                    "module": "pytest",
                    "args": ["tests/", "-v", "-s"],
                    "console": "integratedTerminal",
                    "justMyCode": False
                }
            ]
        }
        
        with open(vscode_dir / 'launch.json', 'w') as f:
            json.dump(launch_config, f, indent=2)
        
        # extensions.json
        extensions = {
            "recommendations": [
                "ms-python.python",
                "ms-python.flake8",
                "ms-python.black-formatter",
                "ms-python.isort",
                "ms-python.mypy-type-checker",
                "ms-toolsai.jupyter",
                "ms-vscode.test-adapter-converter",
                "littlefoxteam.vscode-python-test-adapter",
                "ms-vscode.vscode-json",
                "redhat.vscode-yaml",
                "ms-vscode.vscode-markdown",
                "yzhang.markdown-all-in-one",
                "ms-vscode.vscode-github-issue-notebooks",
                "github.vscode-pull-request-github",
                "ms-vscode.vscode-git-graph",
                "eamodio.gitlens",
                "ms-vscode.vscode-docker",
                "ms-azuretools.vscode-docker"
            ]
        }
        
        with open(vscode_dir / 'extensions.json', 'w') as f:
            json.dump(extensions, f, indent=2)
        
        print("✅ VS Code configuration files created")
        
        # PyCharm配置提示
        print("\nFor PyCharm users:")
        print("1. Open the project in PyCharm")
        print("2. Go to File > Settings > Project > Python Interpreter")
        print("3. Add a new interpreter and point to the virtual environment")
        print("4. Enable code inspections for Python")
        
        return True
    
    def create_development_scripts(self) -> bool:
        """创建开发脚本"""
        print("\n=== Creating Development Scripts ===")
        
        scripts_dir = self.project_root / 'scripts'
        scripts_dir.mkdir(exist_ok=True)
        
        # 快速测试脚本
        quick_test_script = '''#!/usr/bin/env python3
"""快速测试脚本"""
import subprocess
import sys

def main():
    """运行快速测试"""
    print("Running quick tests...")
    
    # 运行单元测试
    result = subprocess.run([
        sys.executable, '-m', 'pytest',
        'tests/unit/',
        '-v',
        '--tb=short',
        '--durations=10'
    ])
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''
        
        with open(scripts_dir / 'quick_test.py', 'w') as f:
            f.write(quick_test_script)
        
        # 代码格式化脚本
        format_script = '''#!/usr/bin/env python3
"""代码格式化脚本"""
import subprocess
import sys

def main():
    """格式化代码"""
    print("Formatting code...")
    
    # Black格式化
    subprocess.run([
        sys.executable, '-m', 'black',
        'modify_multi_attention/',
        'tests/',
        'scripts/'
    ])
    
    # isort导入排序
    subprocess.run([
        sys.executable, '-m', 'isort',
        'modify_multi_attention/',
        'tests/',
        'scripts/'
    ])
    
    print("✅ Code formatting completed!")

if __name__ == '__main__':
    main()
'''
        
        with open(scripts_dir / 'format_code.py', 'w') as f:
            f.write(format_script)
        
        # 使脚本可执行
        if self.system != 'windows':
            os.chmod(scripts_dir / 'quick_test.py', 0o755)
            os.chmod(scripts_dir / 'format_code.py', 0o755)
        
        print("✅ Development scripts created")
        return True
    
    def verify_installation(self) -> bool:
        """验证安装"""
        print("\n=== Verifying Installation ===")
        
        try:
            # 检查PyTorch安装
            result = self.run_command([
                self.python_executable, '-c',
                'import torch; print(f"PyTorch {torch.__version__} installed")'  
            ])
            
            # 检查CUDA可用性
            result = self.run_command([
                self.python_executable, '-c',
                'import torch; print(f"CUDA available: {torch.cuda.is_available()}")'
            ])
            
            # 检查项目模块
            result = self.run_command([
                self.python_executable, '-c',
                'from modify_multi_attention.attention_test import StandardAttention; print("Project modules importable")'
            ])
            
            # 运行快速测试
            print("Running a quick smoke test...")
            result = self.run_command([
                self.python_executable, '-m', 'pytest',
                'tests/', '-x', '--tb=short', '-q'
            ], check=False)
            
            if result.returncode == 0:
                print("✅ All verification checks passed!")
                return True
            else:
                print("⚠️ Some tests failed, but installation seems OK")
                return True
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    def print_next_steps(self):
        """打印后续步骤"""
        print("\n" + "="*60)
        print("🎉 Development Environment Setup Complete!")
        print("="*60)
        
        print("\nNext steps:")
        print("1. Activate your virtual environment:")
        if self.system == 'windows':
            print("   .\\venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        
        print("\n2. Open your IDE and configure the Python interpreter")
        print("\n3. Start developing! Some useful commands:")
        print("   - Run tests: python scripts/run_tests.py")
        print("   - Quick test: python scripts/quick_test.py")
        print("   - Format code: python scripts/format_code.py")
        print("   - Code quality: python scripts/code_quality.py")
        print("   - Benchmark: python scripts/benchmark.py")
        
        print("\n4. Before committing:")
        print("   - Pre-commit hooks will run automatically")
        print("   - Or run manually: pre-commit run --all-files")
        
        print("\n5. Documentation:")
        print("   - Build docs: cd docs && make html")
        print("   - View docs: open docs/_build/html/index.html")
        
        print("\n6. Jupyter notebooks:")
        print("   - Start JupyterLab: jupyter lab")
        print("   - Create notebooks in the notebooks/ directory")
        
        print("\nHappy coding! 🚀")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="VIVTransformer Development Environment Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--no-venv',
        action='store_true',
        help='Skip virtual environment creation'
    )
    
    parser.add_argument(
        '--no-deps',
        action='store_true',
        help='Skip dependency installation'
    )
    
    parser.add_argument(
        '--no-dev',
        action='store_true',
        help='Skip development dependencies'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Install GPU version of PyTorch'
    )
    
    parser.add_argument(
        '--no-precommit',
        action='store_true',
        help='Skip pre-commit setup'
    )
    
    parser.add_argument(
        '--no-ide',
        action='store_true',
        help='Skip IDE configuration'
    )
    
    parser.add_argument(
        '--no-scripts',
        action='store_true',
        help='Skip development scripts creation'
    )
    
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip installation verification'
    )
    
    parser.add_argument(
        '--venv-name',
        default='venv',
        help='Virtual environment name (default: venv)'
    )
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    # 创建设置器
    setup = DevEnvironmentSetup(project_root)
    
    print("VIVTransformer Development Environment Setup")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Python executable: {setup.python_executable}")
    print(f"System: {setup.system}")
    
    success = True
    
    # 检查基础要求
    if not setup.check_python_version():
        print("\n❌ Python version check failed")
        sys.exit(1)
    
    if not setup.check_git():
        print("\n⚠️ Git not found, some features may not work")
    
    # 设置虚拟环境
    if not args.no_venv:
        if not setup.setup_virtual_environment(args.venv_name):
            success = False
    
    # 安装依赖
    if not args.no_deps and success:
        if not setup.install_dependencies(dev=not args.no_dev, gpu=args.gpu):
            success = False
    
    # 设置pre-commit
    if not args.no_precommit and success:
        if not setup.setup_pre_commit():
            print("⚠️ Pre-commit setup failed, but continuing...")
    
    # 设置IDE配置
    if not args.no_ide and success:
        if not setup.setup_ide_config():
            print("⚠️ IDE configuration failed, but continuing...")
    
    # 创建开发脚本
    if not args.no_scripts and success:
        if not setup.create_development_scripts():
            print("⚠️ Development scripts creation failed, but continuing...")
    
    # 验证安装
    if not args.no_verify and success:
        if not setup.verify_installation():
            print("⚠️ Verification failed, but setup may still be usable")
    
    # 打印后续步骤
    if success:
        setup.print_next_steps()
    else:
        print("\n❌ Setup failed. Please check the errors above and try again.")
        sys.exit(1)


if __name__ == '__main__':
    main()