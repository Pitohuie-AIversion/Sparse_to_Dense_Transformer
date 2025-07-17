# Contributing to VIVTransformer

我们欢迎并感谢所有形式的贡献！无论您是想报告bug、提出新功能、改进文档还是提交代码，都请阅读以下指南。

## 🚀 快速开始

### 开发环境设置

1. **Fork 项目**
   ```bash
   git clone https://github.com/your-username/VIVTransformer.git
   cd VIVTransformer
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # 开发依赖
   ```

3. **运行测试**
   ```bash
   python main.py  # 基础功能测试
   ```

## 📋 贡献类型

### 🐛 Bug 报告

在提交bug报告前，请：
- 检查是否已有相关issue
- 使用最新版本重现问题
- 提供详细的复现步骤

**Bug报告模板：**
```markdown
**描述**
简要描述bug

**复现步骤**
1. 执行 '...'
2. 点击 '....'
3. 滚动到 '....'
4. 看到错误

**期望行为**
描述您期望发生的情况

**实际行为**
描述实际发生的情况

**环境信息**
- OS: [e.g. Windows 10]
- Python版本: [e.g. 3.8.5]
- PyTorch版本: [e.g. 1.9.0]
- 其他相关信息

**附加信息**
添加任何其他相关信息、截图等
```

### 💡 功能请求

**功能请求模板：**
```markdown
**功能描述**
清晰简洁地描述您想要的功能

**问题背景**
描述这个功能要解决的问题

**解决方案**
描述您希望的解决方案

**替代方案**
描述您考虑过的其他解决方案

**附加信息**
添加任何其他相关信息
```

### 🔧 代码贡献

#### 开发流程

1. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

2. **编写代码**
   - 遵循项目代码风格
   - 添加必要的注释
   - 编写或更新测试

3. **测试代码**
   ```bash
   # 运行基础测试
   python main.py
   
   # 测试特定注意力机制
   python main.py --attention_types self_attention
   ```

4. **提交代码**
   ```bash
   git add .
   git commit -m "feat: add new attention mechanism"
   ```

5. **推送并创建PR**
   ```bash
   git push origin feature/your-feature-name
   ```

#### 代码规范

**Python代码风格：**
- 使用4个空格缩进
- 行长度不超过88字符
- 使用有意义的变量名
- 添加类型注解（推荐）

**注释规范：**
```python
def attention_mechanism(query, key, value):
    """
    实现注意力机制
    
    Args:
        query: 查询张量 [batch_size, seq_len, d_model]
        key: 键张量 [batch_size, seq_len, d_model]
        value: 值张量 [batch_size, seq_len, d_model]
    
    Returns:
        output: 注意力输出 [batch_size, seq_len, d_model]
        attention_weights: 注意力权重 [batch_size, seq_len, seq_len]
    """
    pass
```

**提交信息规范：**
```
type(scope): description

[optional body]

[optional footer]
```

类型：
- `feat`: 新功能
- `fix`: bug修复
- `docs`: 文档更新
- `style`: 代码格式化
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

示例：
```
feat(attention): add ECA attention mechanism

- Implement Efficient Channel Attention
- Add configuration support
- Update documentation

Closes #123
```

## 🧪 添加新的注意力机制

### 步骤指南

1. **在 `models/attention/` 目录下创建新文件**
   ```python
   # models/attention/your_attention.py
   import torch
   import torch.nn as nn
   
   class YourAttention(nn.Module):
       def __init__(self, d_model, **kwargs):
           super().__init__()
           # 实现初始化
       
       def forward(self, x):
           # 实现前向传播
           return x
   ```

2. **在 `models/attention/__init__.py` 中注册**
   ```python
   from .your_attention import YourAttention
   
   ATTENTION_REGISTRY = {
       # ... 其他注意力机制
       'your_attention': YourAttention,
   }
   ```

3. **更新配置文件**
   ```yaml
   # configs/config.yaml
   attention_test:
     attention_types:
       - your_attention
   ```

4. **添加测试**
   ```bash
   python main.py --attention_types your_attention
   ```

5. **更新文档**
   - 在 `Attention-Mechanisms-Guide.md` 中添加说明
   - 更新 README.md 中的支持列表

### 注意力机制要求

- **输入输出兼容性**: 确保与现有模型架构兼容
- **参数配置**: 支持通过配置文件调整参数
- **内存效率**: 考虑大规模数据的内存使用
- **文档完整**: 提供清晰的使用说明和参数解释

## 📚 文档贡献

### Wiki页面

我们使用GitHub Wiki来维护详细文档：
- **Home.md**: 项目主页
- **Architecture-Overview.md**: 架构概览
- **Attention-Mechanisms-Guide.md**: 注意力机制指南
- **Quick-Start-Tutorial.md**: 快速入门
- **Performance-Comparison.md**: 性能对比
- **FAQ.md**: 常见问题

### 文档规范

- 使用清晰的标题层次
- 提供代码示例
- 包含必要的图表和截图
- 保持内容更新

## 🔍 代码审查

### PR审查清单

**功能性：**
- [ ] 代码实现了预期功能
- [ ] 没有引入新的bug
- [ ] 与现有代码兼容

**代码质量：**
- [ ] 遵循项目代码风格
- [ ] 有适当的注释和文档
- [ ] 变量命名清晰
- [ ] 没有重复代码

**测试：**
- [ ] 包含必要的测试
- [ ] 所有测试通过
- [ ] 覆盖边界情况

**文档：**
- [ ] 更新相关文档
- [ ] 提交信息清晰
- [ ] PR描述详细

## 🏷️ 发布流程

### 版本号规范

我们使用语义化版本控制 (SemVer)：
- `MAJOR.MINOR.PATCH`
- `MAJOR`: 不兼容的API变更
- `MINOR`: 向后兼容的功能性新增
- `PATCH`: 向后兼容的问题修正

### 发布检查清单

- [ ] 所有测试通过
- [ ] 文档更新完整
- [ ] 版本号正确
- [ ] 更新日志完整
- [ ] 性能基准测试

## 🤝 社区准则

### 行为准则

我们致力于为每个人提供友好、安全和欢迎的环境：

- **尊重**: 尊重不同的观点和经验
- **包容**: 欢迎所有背景的贡献者
- **建设性**: 提供建设性的反馈
- **专业**: 保持专业和礼貌的交流

### 沟通渠道

- **Issues**: 用于bug报告和功能请求
- **Discussions**: 用于一般讨论和问题
- **PR**: 用于代码审查和讨论

## 📞 获取帮助

如果您在贡献过程中遇到问题：

1. **查看文档**: 首先查看项目文档和Wiki
2. **搜索Issues**: 查看是否有相关的已知问题
3. **创建Issue**: 如果找不到答案，创建新的issue
4. **参与讨论**: 在GitHub Discussions中提问

## 🙏 致谢

感谢所有为VIVTransformer项目做出贡献的开发者！您的贡献让这个项目变得更好。

---

**记住**: 每个贡献都很重要，无论大小。我们期待您的参与！