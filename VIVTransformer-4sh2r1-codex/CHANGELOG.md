# Changelog

本文档记录了VIVTransformer项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本控制](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 计划中的功能
- 添加更多注意力机制
- 性能优化和内存使用改进
- 支持分布式训练
- 添加更多数据集支持

## [1.0.0] - 2025-01-XX

### 新增
- 🎉 **初始版本发布**
- ✨ **38种注意力机制支持**
  - Self Attention (标准自注意力)
  - Simplified Self Attention (简化自注意力)
  - Efficient Attention (高效注意力)
  - Linear Attention (线性注意力)
  - Global Context Attention (全局上下文注意力)
  - Local Attention (局部注意力)
  - Sparse Attention (稀疏注意力)
  - Dilated Attention (膨胀注意力)
  - Multi-Scale Attention (多尺度注意力)
  - Hierarchical Attention (层次注意力)
  - Cross Attention (交叉注意力)
  - Causal Attention (因果注意力)
  - Bidirectional Attention (双向注意力)
  - Rotary Position Embedding (旋转位置编码)
  - Relative Position Attention (相对位置注意力)
  - Absolute Position Attention (绝对位置注意力)
  - Learned Position Attention (学习位置注意力)
  - Sinusoidal Position Attention (正弦位置注意力)
  - Mobile Attention (移动端注意力)
  - Lightweight Attention (轻量级注意力)
  - Depthwise Attention (深度可分离注意力)
  - Separable Attention (可分离注意力)
  - Low Rank Attention (低秩注意力)
  - Performer Attention (Performer注意力)
  - Linformer Attention (Linformer注意力)
  - Synthesizer Attention (合成器注意力)
  - FNet Attention (FNet注意力)
  - Reformer Attention (Reformer注意力)
  - Longformer Attention (Longformer注意力)
  - BigBird Attention (BigBird注意力)
  - Channel Attention (通道注意力)
  - Spatial Attention (空间注意力)
  - Squeeze Excitation (SE注意力)
  - CBAM Attention (CBAM注意力)
  - ECA Attention (ECA注意力)
  - Coordinate Attention (坐标注意力)
  - Triplet Attention (三元注意力)
  - Dual Attention (双重注意力)

- 🏗️ **完整的实验框架**
  - 模块化架构设计
  - 灵活的配置系统
  - 自动化实验管理
  - 详细的日志记录

- 📊 **数据处理和可视化**
  - 支持多种数据格式
  - 自动数据预处理
  - 训练过程可视化
  - 注意力权重可视化
  - 性能指标统计

- 🔧 **配置和工具**
  - YAML配置文件支持
  - 多损失函数配置
  - 灵活的模型参数设置
  - 实验结果自动保存

- 📚 **完整文档**
  - 详细的README文档
  - GitHub Wiki页面
  - API文档
  - 使用教程
  - 性能对比分析

### 技术特性
- **模块化设计**: 易于扩展和维护
- **配置驱动**: 通过配置文件控制实验
- **自动化测试**: 批量测试多种注意力机制
- **结果分析**: 自动生成性能报告
- **错误处理**: 完善的异常处理和日志记录

### 性能特点
- **内存优化**: 支持大规模数据处理
- **计算效率**: 优化的注意力计算实现
- **并行支持**: 支持GPU加速训练
- **可扩展性**: 易于添加新的注意力机制

### 开发工具
- **代码质量**: 遵循Python最佳实践
- **类型注解**: 完整的类型提示
- **文档字符串**: 详细的函数和类文档
- **模块化测试**: 独立的测试模块

## [0.9.0] - 2025-01-XX (Beta)

### 新增
- 🚧 **Beta版本发布**
- 基础注意力机制实现
- 核心训练框架
- 基本配置系统

### 修复
- 修复训练过程中的内存泄漏问题
- 解决配置文件解析错误
- 修复注意力权重计算bug

## [0.1.0] - 2025-01-XX (Alpha)

### 新增
- 🎯 **项目初始化**
- 基础项目结构
- 核心模块框架
- 初始配置系统

---

## 版本说明

### 版本类型
- **Major (主版本)**: 不兼容的API变更
- **Minor (次版本)**: 向后兼容的功能性新增
- **Patch (修订版本)**: 向后兼容的问题修正

### 变更类型
- `新增` - 新功能
- `变更` - 对现有功能的变更
- `弃用` - 即将移除的功能
- `移除` - 已移除的功能
- `修复` - 问题修复
- `安全` - 安全相关修复

### 发布周期
- **主版本**: 根据重大功能更新发布
- **次版本**: 每月发布（如有新功能）
- **修订版本**: 根据bug修复需要发布

---

## 贡献指南

如果您想为项目做出贡献，请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。