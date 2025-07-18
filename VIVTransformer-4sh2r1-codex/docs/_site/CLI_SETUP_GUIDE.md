# 📋 命令行配置指南

本指南介绍如何使用命令行工具一键配置 VIVTransformer 项目的 GitHub Pages 文档网站。

## 🚀 快速开始

### Windows 用户

#### 方法1: 使用批处理脚本（推荐新手）
```cmd
# 进入docs目录
cd docs

# 双击运行或命令行执行
setup.bat
```

#### 方法2: 使用PowerShell脚本（推荐）
```powershell
# 进入docs目录
cd docs

# 运行PowerShell脚本
.\setup.ps1
```

#### 方法3: 使用Python脚本（高级用户）
```bash
# 进入docs目录
cd docs

# 运行Python脚本
python setup.py
```

### Linux/macOS 用户

```bash
# 进入docs目录
cd docs

# 使用PowerShell Core（推荐）
pwsh setup.ps1

# 或使用Python脚本
python3 setup.py
```

## 🔧 配置模式

### 1. 交互式菜单模式（推荐新手）

直接运行脚本，通过菜单选择操作：

```powershell
.\setup.ps1
```

菜单选项：
- **[1] 快速配置**: 自动检测Git信息，使用默认值
- **[2] 交互式配置**: 手动输入所有配置信息
- **[3] 查看当前配置**: 显示当前配置状态
- **[4] 安装依赖环境**: 安装Ruby、Jekyll等依赖
- **[5] 测试本地预览**: 启动本地开发服务器
- **[0] 退出**: 退出配置工具

### 2. 快速自动配置

自动检测Git仓库信息，使用默认配置：

```powershell
# PowerShell
.\setup.ps1 -Mode auto

# Python
python setup.py --auto
```

### 3. 命令行参数配置

直接通过命令行参数指定配置：

```powershell
# PowerShell完整配置
.\setup.ps1 -Mode auto -Username "yourusername" -Repository "VIVTransformer" -Title "My Documentation" -AuthorName "Your Name" -AuthorEmail "your@email.com"

# Python完整配置
python setup.py --auto --username "yourusername" --repository "VIVTransformer" --title "My Documentation" --author-name "Your Name" --author-email "your@email.com"
```

### 4. 单独功能调用

```powershell
# 仅安装依赖
.\setup.ps1 -Mode install

# 仅启动本地预览
.\setup.ps1 -Mode preview

# 仅查看状态
.\setup.ps1 -Mode status
```

## 📋 参数说明

### PowerShell 脚本参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `-Mode` | 配置模式 | `auto`, `interactive`, `install`, `preview`, `status`, `menu` |
| `-Username` | GitHub用户名 | `yourusername` |
| `-Repository` | 仓库名 | `VIVTransformer` |
| `-Title` | 项目标题 | `"My Documentation"` |
| `-Description` | 项目描述 | `"Project description"` |
| `-AuthorName` | 作者姓名 | `"Your Name"` |
| `-AuthorEmail` | 作者邮箱 | `"your@email.com"` |
| `-Branch` | 部署分支 | `main` |

### Python 脚本参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--auto` | 自动配置模式 | - |
| `--username` | GitHub用户名 | `yourusername` |
| `--repository` | 仓库名 | `VIVTransformer` |
| `--title` | 项目标题 | `"My Documentation"` |
| `--description` | 项目描述 | `"Project description"` |
| `--author-name` | 作者姓名 | `"Your Name"` |
| `--author-email` | 作者邮箱 | `"your@email.com"` |
| `--branch` | 部署分支 | `main` |

## 🛠️ 环境要求

### 必需环境
- **Git**: 版本控制和仓库信息检测
- **Python 3.7+**: 运行Python配置脚本

### 可选环境（本地开发）
- **Ruby 3.0+**: Jekyll静态网站生成器
- **Bundler**: Ruby依赖管理
- **Jekyll**: 静态网站生成器
- **Node.js**: 可选的前端工具支持

### 环境安装指南

#### Windows
```powershell
# 安装Ruby（访问 https://rubyinstaller.org/）
# 下载并安装Ruby+Devkit版本

# 安装Bundler和Jekyll
gem install bundler jekyll

# 验证安装
ruby --version
bundle --version
jekyll --version
```

#### macOS
```bash
# 使用Homebrew安装Ruby
brew install ruby

# 安装Bundler和Jekyll
gem install bundler jekyll
```

#### Linux (Ubuntu/Debian)
```bash
# 安装Ruby和开发工具
sudo apt-get install ruby-full build-essential zlib1g-dev

# 配置gem安装路径
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 安装Bundler和Jekyll
gem install bundler jekyll
```

## 📝 使用示例

### 示例1: 新项目快速配置

```powershell
# 1. 进入项目docs目录
cd docs

# 2. 运行快速配置
.\setup.ps1 -Mode auto

# 3. 提交代码
git add .
git commit -m "Configure GitHub Pages"
git push

# 4. 在GitHub仓库设置中启用Pages
# 访问: https://github.com/USERNAME/REPO/settings/pages
```

### 示例2: 自定义配置

```powershell
# 使用自定义参数配置
.\setup.ps1 -Mode auto `
  -Username "myusername" `
  -Repository "MyProject" `
  -Title "My Project Documentation" `
  -AuthorName "My Name" `
  -AuthorEmail "my@email.com"
```

### 示例3: 本地开发流程

```powershell
# 1. 配置项目
.\setup.ps1 -Mode auto

# 2. 安装依赖
.\setup.ps1 -Mode install

# 3. 启动本地预览
.\setup.ps1 -Mode preview

# 4. 访问 http://localhost:4000 查看效果
```

## 🔍 故障排除

### 常见问题

#### 1. PowerShell执行策略错误
```powershell
# 临时允许脚本执行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 然后运行脚本
.\setup.ps1
```

#### 2. Ruby/Jekyll安装问题
```powershell
# 检查Ruby版本
ruby --version

# 如果版本过低，重新安装Ruby 3.0+
# Windows: https://rubyinstaller.org/
# macOS: brew install ruby
# Linux: 参考上面的安装指南
```

#### 3. Git信息检测失败
```bash
# 配置Git用户信息
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# 添加远程仓库
git remote add origin https://github.com/username/repository.git
```

#### 4. 依赖安装失败
```powershell
# 清理Bundler缓存
bundle clean --force

# 重新安装依赖
bundle install

# 如果仍然失败，删除Gemfile.lock后重试
Remove-Item Gemfile.lock
bundle install
```

### 获取帮助

```powershell
# 查看PowerShell脚本帮助
Get-Help .\setup.ps1 -Full

# 查看Python脚本帮助
python setup.py --help
```

## 📚 相关文档

- [README.md](README.md) - 详细部署指南
- [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) - 部署检查清单
- [GITHUB_PAGES_SETUP.md](GITHUB_PAGES_SETUP.md) - 配置完成后的部署说明

## 🎯 下一步

配置完成后：

1. **提交代码**: `git add . && git commit -m "Configure GitHub Pages" && git push`
2. **启用GitHub Pages**: 访问仓库设置页面启用Pages功能
3. **等待部署**: 通常需要2-10分钟
4. **访问网站**: `https://username.github.io/repository`
5. **本地开发**: 使用 `setup.ps1 -Mode preview` 启动本地服务器

---

💡 **提示**: 建议首次使用时选择交互式菜单模式，熟悉流程后可以使用命令行参数进行快速配置。