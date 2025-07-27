# PDEBench Extended - 重新组织的目录结构

本项目已经重新组织，将原本混杂的文件按功能分类到不同的目录中，提高了项目的可维护性和可读性。

## 目录结构说明

### 📁 core_modules/
核心模块和主要代码
- `mymodels/` - 模型定义
- `PDEBench/` - PDEBench核心库
- `main.py` - 主程序入口
- `__init__.py` - 包初始化文件

### 📁 documentation/
项目文档
- 各种.md文档文件
- 部署指南、训练指南、项目总结等

### 📁 configs/
配置文件
- `config.yaml` - 主配置文件
- `loss_configs/` - 损失函数配置
- `demo_config.yaml` - 演示配置
- `.flake8`, `.gitignore` - 项目配置文件

### 📁 training_tools/
训练相关工具
- `train_*.py` - 各种训练脚本
- `multiscale/` - 多尺度训练模块
- `training/` - 训练工具和实用程序

### 📁 scripts/
脚本文件
- 各种.sh和.bat脚本
- `run_*.py` - 运行脚本
- 启动脚本和管理脚本

### 📁 tests/
测试文件
- `test_*.py` - 各种测试脚本
- `attention_test.py` - 注意力机制测试

### 📁 examples/
示例和工具
- `example_unified_usage.py` - 统一使用示例
- `analyze_dataset.py` - 数据集分析
- `create_*.py` - 数据创建工具
- `fix_*.py` - 修复工具

### 📁 monitoring_tools/
监控工具
- `gpu_monitor.sh` - GPU监控
- `system_monitor.sh` - 系统监控
- `start_*monitoring*` - 监控启动脚本
- `start_tensorboard.py` - TensorBoard启动

### 📁 deployment_tools/
部署工具
- `deploy_*.sh` - 部署脚本
- `requirements.txt` - 依赖文件
- `sever_gpu_config` - 服务器GPU配置
- `pycharm_run_configurations.xml` - PyCharm运行配置

### 📁 visualization_tools/
可视化工具
- `visualize_*.py` - 可视化脚本
- `create_prediction_visualization.py` - 预测可视化
- 各种可视化相关的启动脚本

### 📁 data/
数据文件
- 训练输出、日志文件
- 图像文件和缓存
- 临时数据和演示数据

### 📁 results/
实验结果
- `configurable_multiscale_results_*` - 多尺度实验结果
- `attention_results` - 注意力机制实验结果
- `config_training_1000_epochs` - 1000轮训练结果

### 📁 docs/
详细文档
- 技术文档和指南

### 📁 utils/
工具函数
- 通用工具和实用程序

## 使用说明

1. **开始训练**: 查看 `training_tools/` 目录中的训练脚本
2. **运行示例**: 参考 `examples/` 目录中的示例代码
3. **查看文档**: 阅读 `documentation/` 目录中的相关文档
4. **配置修改**: 编辑 `configs/` 目录中的配置文件
5. **监控训练**: 使用 `monitoring_tools/` 中的监控工具

## 重组优势

- ✅ **清晰的结构**: 文件按功能分类，易于查找
- ✅ **更好的维护性**: 相关文件集中管理
- ✅ **提高效率**: 减少查找文件的时间
- ✅ **便于协作**: 团队成员更容易理解项目结构
- ✅ **模块化**: 不同功能模块独立，便于扩展

重组完成时间: 2025年7月27日