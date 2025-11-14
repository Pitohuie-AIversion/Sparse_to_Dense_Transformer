# VIVTransformer 文档网站 - 运行指南

## 🚀 快速启动

### 前置要求
- Ruby 2.7+ 
- Jekyll 4.4+
- Bundler

### 安装步骤

1. **进入文档目录**
```bash
cd docs
```

2. **安装依赖**
```bash
bundle install
```

3. **启动开发服务器**
```bash
bundle exec jekyll serve --host 0.0.0.0 --port 4000
```

4. **访问网站**
- 英文版: http://localhost:4000/Sparse_to_Dense_Transformer/
- 中文版: http://localhost:4000/Sparse_to_Dense_Transformer/zh/

## 🌐 多语言功能

### 语言切换
- 页面右上角语言切换器
- 支持中英文无缝切换
- 自动记忆语言偏好

### 翻译内容
- 网站标题和描述
- 导航菜单
- 按钮文本
- 页脚信息
- 核心文档页面

## 📱 响应式特性

### 设备适配
- 桌面端: 完整功能展示
- 平板端: 优化布局
- 手机端: 触摸友好界面

### 浏览器支持
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- IE11 (基础功能)

## 🔧 高级功能

### 主题切换
- **快捷键**: 按 `T` 键切换主题
- **自动模式**: 跟随系统主题
- **手动模式**: 用户自定义选择

### 链接验证
- **触发器**: 页面右下角 🔍 按钮
- **功能**: 自动验证所有链接
- **报告**: 导出验证结果

### 导航增强
- **平滑滚动**: 页面内平滑跳转
- **键盘导航**: 完整的键盘支持
- **自动隐藏**: 滚动时智能隐藏

## 🛠️ 故障排除

### 常见问题

#### Jekyll 启动失败
```bash
# 检查Ruby版本
ruby --version

# 更新Bundler
gem install bundler

# 重新安装依赖
bundle install
```

#### 多语言插件问题
```bash
# 确保插件已安装
gem list jekyll-multiple-languages-plugin

# 清理缓存
bundle clean --force
bundle install
```

#### 端口占用
```bash
# 使用不同端口
bundle exec jekyll serve --port 4001
```

### 性能优化

#### 本地开发
- 使用 `--incremental` 参数加速构建
- 启用 `--livereload` 自动刷新
- 配置 `--future` 显示未来日期内容

#### 生产环境
- 启用 Jekyll 缓存
- 压缩 HTML/CSS/JS
- 优化图片资源

## 📊 监控和调试

### 开发工具
- **浏览器 DevTools**: 检查元素和网络请求
- **Jekyll 日志**: 查看构建过程和错误信息
- **链接验证器**: 检查链接有效性

### 性能指标
- 页面加载时间: < 2秒
- CSS文件大小: ~15KB
- JavaScript文件: ~8KB
- 图片优化: WebP格式

## 🔄 部署选项

### GitHub Pages
```bash
# 推送到GitHub自动部署
git add .
git commit -m "Update documentation"
git push origin main
```

### 本地构建
```bash
# 生成静态文件
bundle exec jekyll build

# 输出目录: _site/
```

### Docker部署
```dockerfile
FROM ruby:3.0
WORKDIR /app
COPY Gemfile* ./
RUN bundle install
COPY . .
EXPOSE 4000
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0"]
```

## 📞 获取帮助

### 文档资源
- [Jekyll文档](https://jekyllrb.com/docs/)
- [Just the Docs主题](https://just-the-docs.github.io/just-the-docs/)
- [多语言插件](https://github.com/kurtsson/jekyll-multiple-languages-plugin)

### 社区支持
- [GitHub Issues](https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer/issues)
- [Jekyll社区](https://talk.jekyllrb.com/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/jekyll)

---

**最后更新**: 2024年11月14日  
**文档版本**: v1.0  
**维护团队**: VIVTransformer项目团队