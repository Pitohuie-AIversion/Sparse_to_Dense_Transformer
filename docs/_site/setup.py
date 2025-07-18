#!/usr/bin/env python3
"""
GitHub Pages 一键配置脚本
自动化配置VIVTransformer项目的GitHub Pages文档网站
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import re

class Colors:
    """终端颜色常量"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class GitHubPagesSetup:
    """GitHub Pages配置管理器"""
    
    def __init__(self, project_dir: Path = None):
        self.project_dir = project_dir or Path.cwd()
        self.docs_dir = self.project_dir / "docs"
        self.config_file = self.docs_dir / "_config.yml"
        self.git_config = {}
        
    def print_colored(self, message: str, color: str = Colors.OKGREEN):
        """打印彩色消息"""
        print(f"{color}{message}{Colors.ENDC}")
        
    def print_step(self, step: int, total: int, message: str):
        """打印步骤信息"""
        self.print_colored(f"\n[{step}/{total}] {message}", Colors.OKBLUE)
        
    def run_command(self, command: str, capture_output: bool = True) -> tuple:
        """运行命令并返回结果"""
        try:
            if capture_output:
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True
                )
                return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
            else:
                result = subprocess.run(command, shell=True)
                return result.returncode == 0, "", ""
        except Exception as e:
            return False, "", str(e)
            
    def detect_git_info(self) -> Dict[str, str]:
        """检测Git仓库信息"""
        info = {}
        
        # 检测远程仓库URL
        success, output, _ = self.run_command("git remote get-url origin")
        if success and output:
            # 解析GitHub URL
            if "github.com" in output:
                # 支持HTTPS和SSH格式
                if output.startswith("https://"):
                    # https://github.com/username/repo.git
                    match = re.search(r'github\.com/([^/]+)/([^/]+?)(?:\.git)?$', output)
                elif output.startswith("git@"):
                    # git@github.com:username/repo.git
                    match = re.search(r'github\.com:([^/]+)/([^/]+?)(?:\.git)?$', output)
                else:
                    match = None
                    
                if match:
                    info['username'] = match.group(1)
                    info['repository'] = match.group(2)
                    
        # 检测当前分支
        success, output, _ = self.run_command("git branch --show-current")
        if success and output:
            info['branch'] = output
        else:
            info['branch'] = 'main'
            
        # 检测用户信息
        success, output, _ = self.run_command("git config user.name")
        if success and output:
            info['author_name'] = output
            
        success, output, _ = self.run_command("git config user.email")
        if success and output:
            info['author_email'] = output
            
        return info
        
    def interactive_config(self) -> Dict[str, str]:
        """交互式配置"""
        self.print_colored("\n🔧 配置GitHub Pages信息", Colors.HEADER)
        
        git_info = self.detect_git_info()
        config = {}
        
        # GitHub用户名
        default_username = git_info.get('username', '')
        username = input(f"GitHub用户名 [{default_username}]: ").strip()
        config['username'] = username or default_username
        
        # 仓库名
        default_repo = git_info.get('repository', 'VIVTransformer')
        repository = input(f"仓库名 [{default_repo}]: ").strip()
        config['repository'] = repository or default_repo
        
        # 项目标题
        default_title = "VIVTransformer Documentation"
        title = input(f"项目标题 [{default_title}]: ").strip()
        config['title'] = title or default_title
        
        # 项目描述
        default_desc = "Advanced Transformer Architecture with Vision Integration for Vortex-Induced Vibration Analysis"
        description = input(f"项目描述 [{default_desc}]: ").strip()
        config['description'] = description or default_desc
        
        # 作者姓名
        default_author = git_info.get('author_name', '')
        author_name = input(f"作者姓名 [{default_author}]: ").strip()
        config['author_name'] = author_name or default_author
        
        # 作者邮箱
        default_email = git_info.get('author_email', '')
        author_email = input(f"作者邮箱 [{default_email}]: ").strip()
        config['author_email'] = author_email or default_email
        
        # 分支名
        default_branch = git_info.get('branch', 'main')
        branch = input(f"部署分支 [{default_branch}]: ").strip()
        config['branch'] = branch or default_branch
        
        return config
        
    def update_config_file(self, config: Dict[str, str]):
        """更新Jekyll配置文件"""
        if not self.config_file.exists():
            self.print_colored("❌ 配置文件不存在，请先运行基础配置", Colors.FAIL)
            return False
            
        # 读取现有配置
        with open(self.config_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 更新配置项
        replacements = {
            r'title: .*': f'title: {config["title"]}',
            r'description: .*': f'description: {config["description"]}',
            r'baseurl: .*': f'baseurl: "/{config["repository"]}"',
            r'url: .*': f'url: "https://{config["username"]}.github.io"',
            r'name: .*  # 作者姓名': f'name: {config["author_name"]}',
            r'email: .*  # 作者邮箱': f'email: {config["author_email"]}',
            r'- https://github.com/yourusername/VIVTransformer': f'- https://github.com/{config["username"]}/{config["repository"]}'
        }
        
        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content)
            
        # 写回文件
        with open(self.config_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        self.print_colored("✅ 配置文件更新完成", Colors.OKGREEN)
        return True
        
    def check_environment(self) -> Dict[str, bool]:
        """检查环境依赖"""
        checks = {}
        
        # 检查Git
        success, _, _ = self.run_command("git --version")
        checks['git'] = success
        
        # 检查Ruby
        success, output, _ = self.run_command("ruby --version")
        checks['ruby'] = success
        if success:
            version_match = re.search(r'ruby (\d+\.\d+)', output)
            if version_match:
                version = float(version_match.group(1))
                checks['ruby_version_ok'] = version >= 3.0
            else:
                checks['ruby_version_ok'] = False
                
        # 检查Bundler
        success, _, _ = self.run_command("bundle --version")
        checks['bundler'] = success
        
        # 检查Jekyll
        success, _, _ = self.run_command("jekyll --version")
        checks['jekyll'] = success
        
        # 检查Node.js (可选)
        success, _, _ = self.run_command("node --version")
        checks['nodejs'] = success
        
        return checks
        
    def install_dependencies(self):
        """安装依赖"""
        if not (self.docs_dir / "Gemfile").exists():
            self.print_colored("❌ Gemfile不存在", Colors.FAIL)
            return False
            
        self.print_colored("📦 安装Ruby依赖...", Colors.OKBLUE)
        
        # 切换到docs目录
        original_dir = os.getcwd()
        os.chdir(self.docs_dir)
        
        try:
            # 安装bundler（如果需要）
            success, _, error = self.run_command("bundle --version")
            if not success:
                self.print_colored("安装Bundler...", Colors.WARNING)
                success, _, error = self.run_command("gem install bundler")
                if not success:
                    self.print_colored(f"❌ Bundler安装失败: {error}", Colors.FAIL)
                    return False
                    
            # 安装依赖
            success, output, error = self.run_command("bundle install", capture_output=False)
            if success:
                self.print_colored("✅ 依赖安装完成", Colors.OKGREEN)
                return True
            else:
                self.print_colored(f"❌ 依赖安装失败: {error}", Colors.FAIL)
                return False
                
        finally:
            os.chdir(original_dir)
            
    def test_local_build(self) -> bool:
        """测试本地构建"""
        self.print_colored("🧪 测试本地构建...", Colors.OKBLUE)
        
        original_dir = os.getcwd()
        os.chdir(self.docs_dir)
        
        try:
            success, output, error = self.run_command("bundle exec jekyll build --dry-run")
            if success:
                self.print_colored("✅ 本地构建测试通过", Colors.OKGREEN)
                return True
            else:
                self.print_colored(f"❌ 本地构建测试失败: {error}", Colors.FAIL)
                return False
        finally:
            os.chdir(original_dir)
            
    def create_github_pages_config(self, config: Dict[str, str]):
        """创建GitHub Pages配置说明"""
        instructions = f"""
# 🚀 GitHub Pages 部署说明

## 自动配置已完成！

### 📋 配置信息
- **GitHub用户名**: {config['username']}
- **仓库名**: {config['repository']}
- **部署分支**: {config['branch']}
- **网站地址**: https://{config['username']}.github.io/{config['repository']}

### 🔧 下一步操作

#### 1. 提交代码到GitHub
```bash
git add .
git commit -m "Configure GitHub Pages documentation site"
git push origin {config['branch']}
```

#### 2. 启用GitHub Pages
1. 访问: https://github.com/{config['username']}/{config['repository']}/settings/pages
2. 在"Source"部分选择"Deploy from a branch"
3. 选择"{config['branch']}"分支和"/docs"文件夹
4. 点击"Save"

#### 3. 等待部署完成
- 部署通常需要2-10分钟
- 可以在Actions标签页查看部署状态
- 部署完成后访问: https://{config['username']}.github.io/{config['repository']}

### 🛠️ 本地开发
```bash
# 进入docs目录
cd docs

# 启动本地服务器
bundle exec jekyll serve

# 访问 http://localhost:4000
```

### 📚 更多帮助
- 查看 docs/README.md 获取详细说明
- 查看 docs/DEPLOYMENT_CHECKLIST.md 获取部署检查清单
- 如有问题，请查看 GitHub Actions 日志
"""
        
        instructions_file = self.docs_dir / "GITHUB_PAGES_SETUP.md"
        with open(instructions_file, 'w', encoding='utf-8') as f:
            f.write(instructions)
            
        self.print_colored(f"✅ 部署说明已保存到: {instructions_file}", Colors.OKGREEN)
        
    def run_setup(self, auto_mode: bool = False, config_data: Dict[str, str] = None):
        """运行完整配置流程"""
        self.print_colored("🚀 VIVTransformer GitHub Pages 一键配置", Colors.HEADER)
        self.print_colored("=" * 50, Colors.HEADER)
        
        total_steps = 7
        
        # 步骤1: 检查环境
        self.print_step(1, total_steps, "检查环境依赖")
        env_checks = self.check_environment()
        
        for check, status in env_checks.items():
            status_icon = "✅" if status else "❌"
            self.print_colored(f"  {status_icon} {check}: {'OK' if status else 'Missing'}", 
                             Colors.OKGREEN if status else Colors.WARNING)
                             
        if not env_checks.get('git', False):
            self.print_colored("❌ Git未安装，请先安装Git", Colors.FAIL)
            return False
            
        # 步骤2: 检查项目结构
        self.print_step(2, total_steps, "检查项目结构")
        if not self.docs_dir.exists():
            self.print_colored("❌ docs目录不存在，请先运行基础配置", Colors.FAIL)
            return False
            
        required_files = ['_config.yml', 'index.md', 'Gemfile']
        for file in required_files:
            if (self.docs_dir / file).exists():
                self.print_colored(f"  ✅ {file}", Colors.OKGREEN)
            else:
                self.print_colored(f"  ❌ {file} 缺失", Colors.FAIL)
                return False
                
        # 步骤3: 获取配置信息
        self.print_step(3, total_steps, "配置项目信息")
        if auto_mode and config_data:
            config = config_data
            self.print_colored("使用提供的配置数据", Colors.OKGREEN)
        elif auto_mode:
            config = self.detect_git_info()
            # 使用默认值填充缺失的配置
            config.setdefault('repository', 'VIVTransformer')
            config.setdefault('title', 'VIVTransformer Documentation')
            config.setdefault('description', 'Advanced Transformer Architecture with Vision Integration')
            config.setdefault('author_name', 'VIVTransformer Team')
            config.setdefault('author_email', 'team@vivtransformer.com')
            config.setdefault('branch', 'main')
            self.print_colored("使用自动检测的配置", Colors.OKGREEN)
        else:
            config = self.interactive_config()
            
        # 步骤4: 更新配置文件
        self.print_step(4, total_steps, "更新配置文件")
        if not self.update_config_file(config):
            return False
            
        # 步骤5: 安装依赖
        self.print_step(5, total_steps, "安装依赖")
        if env_checks.get('ruby', False):
            if not self.install_dependencies():
                self.print_colored("⚠️  依赖安装失败，但可以继续", Colors.WARNING)
        else:
            self.print_colored("⚠️  Ruby未安装，跳过依赖安装", Colors.WARNING)
            
        # 步骤6: 测试构建
        self.print_step(6, total_steps, "测试构建")
        if env_checks.get('ruby', False) and env_checks.get('bundler', False):
            self.test_local_build()
        else:
            self.print_colored("⚠️  跳过构建测试（缺少Ruby环境）", Colors.WARNING)
            
        # 步骤7: 生成部署说明
        self.print_step(7, total_steps, "生成部署说明")
        self.create_github_pages_config(config)
        
        # 完成
        self.print_colored("\n🎉 配置完成！", Colors.OKGREEN)
        self.print_colored("=" * 50, Colors.OKGREEN)
        
        self.print_colored("\n📋 下一步操作:", Colors.OKBLUE)
        self.print_colored(f"1. 提交代码: git add . && git commit -m 'Configure GitHub Pages' && git push", Colors.OKCYAN)
        self.print_colored(f"2. 启用GitHub Pages: https://github.com/{config.get('username', 'USERNAME')}/{config.get('repository', 'REPO')}/settings/pages", Colors.OKCYAN)
        self.print_colored(f"3. 访问网站: https://{config.get('username', 'USERNAME')}.github.io/{config.get('repository', 'REPO')}", Colors.OKCYAN)
        
        self.print_colored("\n📚 查看详细说明: docs/GITHUB_PAGES_SETUP.md", Colors.OKBLUE)
        
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='VIVTransformer GitHub Pages 一键配置工具')
    parser.add_argument('--auto', action='store_true', help='自动模式（使用检测到的Git配置）')
    parser.add_argument('--username', help='GitHub用户名')
    parser.add_argument('--repository', help='仓库名')
    parser.add_argument('--title', help='项目标题')
    parser.add_argument('--description', help='项目描述')
    parser.add_argument('--author-name', help='作者姓名')
    parser.add_argument('--author-email', help='作者邮箱')
    parser.add_argument('--branch', help='部署分支', default='main')
    parser.add_argument('--project-dir', help='项目目录路径')
    
    args = parser.parse_args()
    
    # 设置项目目录
    project_dir = Path(args.project_dir) if args.project_dir else Path.cwd()
    
    # 创建配置管理器
    setup = GitHubPagesSetup(project_dir)
    
    # 准备配置数据
    config_data = None
    if any([args.username, args.repository, args.title, args.description, 
            args.author_name, args.author_email]):
        config_data = {
            'username': args.username,
            'repository': args.repository or 'VIVTransformer',
            'title': args.title or 'VIVTransformer Documentation',
            'description': args.description or 'Advanced Transformer Architecture with Vision Integration',
            'author_name': args.author_name or 'VIVTransformer Team',
            'author_email': args.author_email or 'team@vivtransformer.com',
            'branch': args.branch
        }
    
    # 运行配置
    try:
        success = setup.run_setup(auto_mode=args.auto, config_data=config_data)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        setup.print_colored("\n\n⚠️  配置被用户中断", Colors.WARNING)
        sys.exit(1)
    except Exception as e:
        setup.print_colored(f"\n❌ 配置过程中出现错误: {str(e)}", Colors.FAIL)
        sys.exit(1)

if __name__ == "__main__":
    main()