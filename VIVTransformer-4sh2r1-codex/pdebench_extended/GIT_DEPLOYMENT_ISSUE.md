# Git部署问题解决方案

## 问题描述

在服务器端部署时遇到 `ModuleNotFoundError: No module named 'multiscale.data'` 错误的根本原因是：

**`multiscale/data/` 目录被 `.gitignore` 文件意外忽略了！**

## 问题分析

### 原因

在项目根目录的 `.gitignore` 文件中，有以下规则：
```
data/raw/
data/processed/
data/external/
```

虽然这些规则本意是忽略大型数据文件目录，但Git的模式匹配可能会影响到 `multiscale/data/` 目录，导致该目录及其内容不被版本控制跟踪。

### 影响

- 本地开发环境中 `multiscale/data/` 目录存在
- 但在服务器端通过 `git clone` 或 `git pull` 时，该目录被忽略
- 导致 `multiscale.data.multiscale_adapter` 模块无法导入

## 解决方案

### 1. 修复 .gitignore 文件

已在 `.gitignore` 文件中添加了排除规则：
```gitignore
# Important: Do NOT ignore multiscale/data directory
!multiscale/data/
!multiscale/data/**
```

### 2. 确保文件被正确跟踪

在本地执行以下命令：
```bash
# 强制添加 multiscale/data 目录
git add -f multiscale/data/
git add -f multiscale/data/**

# 提交更改
git commit -m "fix: ensure multiscale/data directory is tracked by git"

# 推送到远程仓库
git push
```

### 3. 服务器端重新部署

在服务器上：
```bash
# 重新克隆仓库（推荐）
rm -rf pdebench_extended
git clone <repository_url>

# 或者强制更新现有仓库
cd pdebench_extended
git clean -fd
git reset --hard HEAD
git pull origin main
```

## 验证步骤

### 1. 检查文件是否存在
```bash
ls -la multiscale/data/
# 应该看到：
# __init__.py
# multiscale_adapter.py
```

### 2. 运行修复脚本
```bash
python fix_server_import.py
```

### 3. 测试训练脚本
```bash
python train_configurable_multiscale.py --scale_factor 4 --num_epochs 1 --batch_size 2
```

## 预防措施

### 1. 定期检查 Git 状态
```bash
# 检查哪些文件被忽略
git status --ignored

# 检查特定目录是否被跟踪
git ls-files multiscale/data/
```

### 2. 使用更精确的 .gitignore 规则

避免使用过于宽泛的模式，如：
- ❌ `data/` （会忽略所有名为 data 的目录）
- ✅ `data/raw/` （只忽略特定的数据目录）

### 3. 在 CI/CD 中添加检查

可以在部署脚本中添加文件存在性检查：
```bash
if [ ! -f "multiscale/data/multiscale_adapter.py" ]; then
    echo "错误：关键文件 multiscale_adapter.py 不存在！"
    exit 1
fi
```

## 总结

这个问题提醒我们：

1. **`.gitignore` 规则需要仔细设计**，避免意外忽略重要文件
2. **部署前应该验证所有必要文件都存在**
3. **使用排除规则（`!`）来保护重要目录**
4. **定期检查 Git 跟踪状态**

通过以上修复，现在 `multiscale/data/` 目录将被正确跟踪和部署到服务器端。