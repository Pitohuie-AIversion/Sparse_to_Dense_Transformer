# PyCharm Linux 快速运行指南

## 🚀 PyCharm 中快速运行 VIVTransformer 可视化训练

### 1. 项目导入和环境配置

#### 1.1 导入项目
1. 打开 PyCharm
2. 选择 `File` → `Open`
3. 选择 `pdebench_extended` 文件夹
4. 点击 `OK` 导入项目

#### 1.2 配置 Python 解释器
1. 打开 `File` → `Settings` (或 `Ctrl+Alt+S`)
2. 选择 `Project: pdebench_extended` → `Python Interpreter`
3. 点击齿轮图标 → `Add...`
4. 选择 `Virtualenv Environment` → `New environment`
5. 设置位置为项目目录下的 `viv_env`
6. 点击 `OK` 创建虚拟环境

#### 1.3 安装依赖包
在 PyCharm 的 Terminal 中运行：
```bash
pip install -r requirements.txt
```

### 2. 快速运行配置

#### 2.1 创建运行配置 - 增强版可视化训练

1. 点击右上角的 `Add Configuration...`
2. 点击 `+` → `Python`
3. 配置如下：
   - **Name**: `增强版可视化训练`
   - **Script path**: 选择 `enhanced_visual_training.py`
   - **Parameters**: `--epochs 20 --output-dir enhanced_training_output`
   - **Working directory**: 项目根目录
   - **Environment variables**: 
     ```
     MPLBACKEND=Agg
     KMP_DUPLICATE_LIB_OK=TRUE
     OMP_NUM_THREADS=1
     ```
4. 点击 `OK` 保存

#### 2.2 创建运行配置 - 预测可视化演示

1. 再次点击 `Add Configuration...`
2. 点击 `+` → `Python`
3. 配置如下：
   - **Name**: `预测可视化演示`
   - **Script path**: 选择 `create_prediction_visualization.py`
   - **Working directory**: 项目根目录
   - **Environment variables**: 
     ```
     MPLBACKEND=Agg
     KMP_DUPLICATE_LIB_OK=TRUE
     ```
4. 点击 `OK` 保存

#### 2.3 创建运行配置 - 自定义训练

1. 创建新的 Python 配置
2. 配置如下：
   - **Name**: `自定义训练`
   - **Script path**: 选择 `enhanced_visual_training.py`
   - **Parameters**: `--epochs 50 --batch-size 8 --output-dir custom_output`
   - **Working directory**: 项目根目录
   - **Environment variables**: 同上

### 3. 一键运行方式

#### 3.1 使用工具栏运行
1. 在右上角的运行配置下拉菜单中选择要运行的配置
2. 点击绿色的运行按钮 (▶️) 或按 `Shift+F10`
3. 查看控制台输出和运行结果

#### 3.2 使用右键菜单
1. 在项目文件树中右键点击 `enhanced_visual_training.py`
2. 选择 `Run 'enhanced_visual_training'`
3. PyCharm 会自动创建临时运行配置

#### 3.3 使用快捷键
- `Shift+F10`: 运行当前选中的配置
- `Ctrl+Shift+F10`: 运行当前文件
- `Shift+F9`: 调试运行

### 4. 终端运行方式

在 PyCharm 底部的 Terminal 标签中：

```bash
# 激活虚拟环境（如果未自动激活）
source viv_env/bin/activate

# 运行增强版训练
python enhanced_visual_training.py --epochs 20

# 运行预测可视化
python create_prediction_visualization.py

# 使用shell脚本（需要先设置权限）
chmod +x *.sh
./启动增强版可视化训练.sh
```

### 5. 调试配置

#### 5.1 设置断点
1. 在代码行号左侧点击设置断点
2. 选择调试配置运行 (🐛 图标)
3. 程序会在断点处暂停

#### 5.2 调试快捷键
- `F8`: 单步执行
- `F7`: 步入函数
- `Shift+F8`: 步出函数
- `F9`: 继续执行
- `Ctrl+F8`: 切换断点

### 6. 查看运行结果

#### 6.1 在 PyCharm 中查看
1. 运行完成后，在项目文件树中刷新
2. 展开输出目录（如 `enhanced_training_output`）
3. 双击图片文件可在 PyCharm 中预览
4. 双击 `.html` 文件可在浏览器中打开

#### 6.2 使用内置工具
- **File Watcher**: 自动监控文件变化
- **Version Control**: 查看代码变更
- **Database**: 查看训练数据（如果使用数据库）

### 7. 性能监控

#### 7.1 内存和CPU监控
1. 在 PyCharm 底部状态栏查看内存使用
2. 使用 `View` → `Tool Windows` → `Profiler` 进行性能分析

#### 7.2 GPU监控（如果有GPU）
在 Terminal 中运行：
```bash
# 监控GPU使用情况
watch -n 1 nvidia-smi

# 或者在Python中
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 8. 常用快捷键

| 功能 | 快捷键 |
|------|--------|
| 运行 | `Shift+F10` |
| 调试 | `Shift+F9` |
| 停止 | `Ctrl+F2` |
| 重新运行 | `Ctrl+F5` |
| 查找文件 | `Ctrl+Shift+N` |
| 查找类 | `Ctrl+N` |
| 全局搜索 | `Ctrl+Shift+F` |
| 替换 | `Ctrl+R` |
| 格式化代码 | `Ctrl+Alt+L` |
| 优化导入 | `Ctrl+Alt+O` |

### 9. 插件推荐

安装有用的插件来提升开发体验：

1. **Python Requirements**: 管理依赖包
2. **Matplotlib Integration**: 更好的图表支持
3. **CSV Plugin**: 查看CSV数据文件
4. **Markdown**: 查看和编辑Markdown文档
5. **Git Integration**: 版本控制

### 10. 故障排除

#### 10.1 常见问题

**问题1**: 找不到模块
```bash
# 解决方案：确保在正确的虚拟环境中
source viv_env/bin/activate
pip list  # 检查已安装的包
```

**问题2**: 图形显示错误
```python
# 在代码开头添加
import matplotlib
matplotlib.use('Agg')
```

**问题3**: 权限错误
```bash
chmod +x *.sh
```

**问题4**: 内存不足
- 减少 `--batch-size` 参数
- 减少 `--epochs` 参数
- 关闭其他应用程序

#### 10.2 日志查看

在 PyCharm 中查看日志：
1. 运行配置中启用日志记录
2. 在 `Run` 窗口查看实时输出
3. 检查输出目录中的日志文件

### 11. 最佳实践

1. **使用虚拟环境**: 避免包冲突
2. **设置环境变量**: 确保图形后端正确
3. **定期保存**: 使用 `Ctrl+S` 保存文件
4. **版本控制**: 定期提交代码变更
5. **监控资源**: 注意内存和CPU使用情况
6. **备份结果**: 及时保存训练结果

---

## 🎯 快速开始清单

- [ ] 导入项目到 PyCharm
- [ ] 配置 Python 虚拟环境
- [ ] 安装依赖包 (`pip install -r requirements.txt`)
- [ ] 创建运行配置
- [ ] 设置环境变量
- [ ] 运行第一个训练任务
- [ ] 查看生成的可视化结果

现在您可以在 PyCharm 中高效地运行和调试 VIVTransformer 可视化训练项目了！