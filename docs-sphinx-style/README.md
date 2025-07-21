# VIVTransformer Documentation Site

这是VIVTransformer项目的GitHub Pages文档网站。本文档网站基于Jekyll构建，提供了完整的项目文档、API参考和使用指南。

## 🚀 快速部署

### 方法一：GitHub Pages自动部署（推荐）

1. **更新配置文件**
   ```bash
   # 编辑 _config.yml
   # 将 'yourusername' 替换为您的GitHub用户名
   # 将 'VIVTransformer' 替换为您的仓库名
   ```

2. **提交代码到GitHub**
   ```bash
   git add .
   git commit -m "Add GitHub Pages documentation site"
   git push origin main
   ```

3. **启用GitHub Pages**
   - 进入GitHub仓库设置页面
   - 滚动到"Pages"部分
   - 在"Source"下选择"Deploy from a branch"
   - 选择"main"分支和"/docs"文件夹
   - 点击"Save"

4. **访问网站**
   - 网站将在几分钟内可用
   - 访问地址：`https://yourusername.github.io/VIVTransformer`

### 方法二：本地开发和预览

1. **安装依赖**
   ```bash
   # 确保已安装Ruby 3.0+
   cd docs
   bundle install
   ```

2. **本地运行**
   ```bash
   bundle exec jekyll serve
   # 或者指定端口
   bundle exec jekyll serve --port 4001
   ```

3. **访问本地网站**
   - 打开浏览器访问：`http://localhost:4000`

## 📁 目录结构

```
docs/
├── _config.yml          # Jekyll配置文件
├── _layouts/             # 页面布局模板
│   └── default.html      # 默认布局
├── assets/               # 静态资源
│   └── css/
│       └── custom.css    # 自定义样式
├── pages/                # 文档页面
│   ├── quick-start-tutorial.md
│   ├── architecture-overview.md
│   └── ...
├── index.md              # 首页
├── Gemfile               # Ruby依赖
├── convert_wiki.py       # Wiki转换脚本
└── README.md             # 本文件
```

## 🔧 自定义配置

### 更新站点信息

编辑 `_config.yml` 文件：

```yaml
# 基本信息
title: "您的项目名称"
description: "项目描述"
baseurl: "/您的仓库名"
url: "https://您的用户名.github.io"

# 作者信息
author:
  name: "您的姓名"
  email: "您的邮箱"

# 社交链接
social:
  name: "项目名称"
  links:
    - https://github.com/您的用户名/您的仓库名
```

### 自定义样式

编辑 `assets/css/custom.css` 文件来自定义网站外观：

```css
/* 自定义主色调 */
:root {
  --primary-color: #your-color;
  --secondary-color: #your-secondary-color;
}

/* 自定义字体 */
body {
  font-family: 'Your Font', sans-serif;
}
```

### 添加新页面

1. 在 `pages/` 目录下创建新的 `.md` 文件
2. 添加Jekyll Front Matter：
   ```yaml
   ---
   layout: default
   title: "页面标题"
   description: "页面描述"
   permalink: /pages/your-page/
   ---
   ```
3. 编写Markdown内容
4. 更新导航菜单（在 `_layouts/default.html` 中）

## 🔄 从Wiki同步内容

如果您有现有的Wiki内容，可以使用提供的转换脚本：

```bash
cd docs
python convert_wiki.py
```

这个脚本会：
- 将Wiki文件转换为Jekyll格式
- 添加适当的Front Matter
- 修复内部链接
- 生成站点地图

## 🎨 主题和样式

### 当前主题特性

- 📱 响应式设计，支持移动端
- 🌙 自动深色模式支持
- 🔍 自动生成目录
- 💫 平滑滚动和动画效果
- 📊 代码高亮和数学公式支持
- 🔗 面包屑导航
- 📋 侧边栏快速链接

### 可用的CSS类

```html
<!-- 警告框 -->
<div class="alert alert-info">信息提示</div>
<div class="alert alert-warning">警告信息</div>
<div class="alert alert-error">错误信息</div>
<div class="alert alert-success">成功信息</div>

<!-- 工具类 -->
<p class="text-center">居中文本</p>
<p class="text-muted">灰色文本</p>
```

## 🚀 高级功能

### 启用Google Analytics

在 `_config.yml` 中添加：

```yaml
google_analytics: UA-XXXXXXXX-X
```

### 添加搜索功能

可以集成以下搜索解决方案：
- [Algolia DocSearch](https://docsearch.algolia.com/)
- [Simple Jekyll Search](https://github.com/christian-fei/Simple-Jekyll-Search)

### 多语言支持

可以使用 [jekyll-multiple-languages-plugin](https://github.com/kurtsson/jekyll-multiple-languages-plugin) 添加多语言支持。

## 🔧 故障排除

### 常见问题

1. **页面404错误**
   - 检查 `baseurl` 配置是否正确
   - 确认文件路径和permalink设置

2. **样式不加载**
   - 检查CSS文件路径
   - 确认 `_config.yml` 中的URL设置

3. **构建失败**
   - 检查Gemfile依赖
   - 查看GitHub Actions日志
   - 确认Markdown语法正确

4. **链接失效**
   - 使用相对路径而非绝对路径
   - 检查文件名大小写

### 调试命令

```bash
# 检查Jekyll配置
bundle exec jekyll doctor

# 详细构建信息
bundle exec jekyll build --verbose

# 检查依赖
bundle outdated
```

## 📚 相关资源

- [Jekyll官方文档](https://jekyllrb.com/docs/)
- [GitHub Pages文档](https://docs.github.com/en/pages)
- [Markdown语法指南](https://www.markdownguide.org/)
- [Jekyll主题库](https://jekyllthemes.io/)
- [Liquid模板语言](https://shopify.github.io/liquid/)

## 🤝 贡献

欢迎贡献文档内容！请：

1. Fork本仓库
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📄 许可证

本文档网站采用与主项目相同的许可证。

---

*需要帮助？请查看 [GitHub Issues](https://github.com/yourusername/VIVTransformer/issues) 或联系维护者。*