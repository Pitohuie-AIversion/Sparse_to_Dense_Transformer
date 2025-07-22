# PDEBench Integration for Multi-Attention Transformer

## 概述

本项目已成功集成PDEBench数据集支持，允许在多种偏微分方程（PDE）数据集上训练和评估多注意力机制的Transformer模型。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

新增的PDEBench相关依赖：
- `h5py`: HDF5文件处理
- `scipy`: 科学计算
- `sklearn`: 机器学习工具
- `seaborn`: 数据可视化
- `tqdm`: 进度条显示

### 2. 数据准备

1. 下载PDEBench数据集：
   ```bash
   # 创建数据目录
   mkdir -p data/pdebench
   
   # 下载数据文件（示例）
   # 请参考PDEBench官方文档获取数据下载链接
   ```

2. 数据目录结构：
   ```
   data/pdebench/
   ├── ns_incom_inhom_2d.h5      # Navier-Stokes不可压缩流
   ├── darcy_flow_2d.h5          # Darcy流方程
   ├── shallow_water_2d.h5       # 浅水方程
   └── ...
   ```

### 3. 运行示例

```bash
# 运行PDEBench示例
python examples/pdebench_example.py

# 使用PDEBench配置训练模型
python main.py -c configs/pdebench_example.yaml

# 运行集成测试
python tests/test_pdebench_integration.py
```

## 📁 新增文件

### 核心文件
- `data/pdebench_adapter.py`: PDEBench数据集适配器
- `configs/pdebench_example.yaml`: PDEBench配置示例
- `examples/pdebench_example.py`: 使用示例脚本
- `tests/test_pdebench_integration.py`: 集成测试
- `docs/PDEBench_Integration_Guide.md`: 详细集成指南

### 修改文件
- `data/dataloader.py`: 添加PDEBench数据加载支持
- `main.py`: 集成自适应数据加载器
- `requirements.txt`: 添加新依赖

## 🔧 配置说明

### 启用PDEBench

在配置文件中设置：
```yaml
data:
  use_pdebench: true
  
pdebench:
  data_root: ./data/pdebench
  
current_pde: ns_incom  # 选择PDE类型
```

### 支持的PDE类型

1. **ns_incom**: Navier-Stokes不可压缩流
   - 空间分辨率: 64×64
   - 时间步数: 49
   - 变量: 速度场(u, v)

2. **darcy_flow**: Darcy流方程
   - 空间分辨率: 32×32
   - 时间步数: 1
   - 变量: 压力场

3. **shallow_water**: 浅水方程
   - 空间分辨率: 128×128
   - 时间步数: 40
   - 变量: 高度和速度场

## 🧪 测试验证

运行完整测试套件：
```bash
python tests/test_pdebench_integration.py
```

测试覆盖：
- ✅ PDEBench数据集创建
- ✅ 数据加载器功能
- ✅ 自适应数据加载
- ✅ 数据标准化
- ✅ 数据分割
- ✅ 错误处理
- ✅ 配置验证
- ✅ 批处理功能

## 📊 性能优化

### 数据加载优化
- 多进程数据加载
- 内存固定（pin_memory）
- 持久化工作进程
- 批量预处理

### 模型优化
- 混合精度训练
- 梯度裁剪
- 学习率调度
- 早停机制

## 🔍 可视化功能

支持的可视化类型：
- 速度场可视化
- 压力场分布
- 涡度场显示
- 流线图
- 训练过程动画

## 📈 评估指标

### 标准指标
- MSE (均方误差)
- MAE (平均绝对误差)
- RMSE (均方根误差)
- 相对误差
- 相关系数

### 物理约束指标
- 能量守恒
- 质量守恒
- 涡度保持

## 🛠️ 故障排除

### 常见问题

1. **模块导入错误**
   ```
   ModuleNotFoundError: No module named 'data.pdebench_adapter'
   ```
   解决：确保在项目根目录运行，或检查Python路径

2. **数据文件未找到**
   ```
   FileNotFoundError: [Errno 2] No such file or directory
   ```
   解决：检查数据文件路径和配置文件中的路径设置

3. **内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决：减小batch_size或使用梯度累积

### 调试模式

启用详细日志：
```yaml
logging:
  level: DEBUG
  save_logs: true
```

## 📚 参考资源

- [PDEBench官方仓库](https://github.com/pdebench/PDEBench)
- [PDEBench论文](https://arxiv.org/abs/2210.07182)
- [项目文档](docs/PDEBench_Integration_Guide.md)
- [API参考](docs/api_reference.md)

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

---

**注意**: 确保在使用PDEBench数据集时遵循相应的许可证和引用要求。