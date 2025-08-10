# 多语言配置指南 / Multilingual Setup Guide

本文档说明如何为 VIVTransformer 文档网站配置和使用多语言功能。

This document explains how to configure and use multilingual features for the VIVTransformer documentation site.

## 🌐 配置概览 / Configuration Overview

### 已配置的语言 / Configured Languages
- **English (en)** - 默认语言 / Default language
- **中文 (zh)** - 中文支持 / Chinese support

### 插件配置 / Plugin Configuration

我们使用 `jekyll-multiple-languages-plugin` 插件来实现多语言支持：

We use the `jekyll-multiple-languages-plugin` to enable multilingual support:

```yaml
# _config.yml
plugins:
  - jekyll-multiple-languages-plugin

# Multi-language configuration
languages: ["en", "zh"]
default_lang: "en"
exclude_from_localizations: ["assets", "_site"]
parallel_localization: true
```

## 📁 文件结构 / File Structure

```
docs/
├── _i18n/
│   ├── en.yml          # 英文翻译文件 / English translations
│   └── zh.yml          # 中文翻译文件 / Chinese translations
├── _includes/
│   └── language-switcher.html  # 语言切换组件 / Language switcher
├── pages/
│   ├── en/             # 英文页面 / English pages (可选)
│   └── zh/             # 中文页面 / Chinese pages (可选)
└── index.md            # 多语言主页 / Multilingual homepage
```

## 🔧 使用方法 / Usage

### 1. 在页面中使用翻译 / Using Translations in Pages

```liquid
<!-- 使用翻译键 / Use translation keys -->
<h1>{% t site.title %}</h1>
<p>{% t site.description %}</p>

<!-- 条件语言内容 / Conditional language content -->
{% if site.active_lang == 'zh' %}
  <p>这是中文内容</p>
{% else %}
  <p>This is English content</p>
{% endif %}
```

### 2. 添加语言切换器 / Adding Language Switcher

在页面顶部添加语言切换器：

Add the language switcher at the top of pages:

```liquid
{% include language-switcher.html %}
```

### 3. 翻译文件格式 / Translation File Format

在 `_i18n/en.yml` 和 `_i18n/zh.yml` 中定义翻译：

Define translations in `_i18n/en.yml` and `_i18n/zh.yml`:

```yaml
# en.yml
site:
  title: "VIVTransformer Documentation"
  description: "Advanced Transformer Architecture"

navigation:
  home: "Home"
  getting_started: "Getting Started"

common:
  read_more: "Read More"
  back_to_top: "Back to Top"
```

```yaml
# zh.yml
site:
  title: "VIVTransformer 文档"
  description: "先进的Transformer架构"

navigation:
  home: "首页"
  getting_started: "快速开始"

common:
  read_more: "了解更多"
  back_to_top: "返回顶部"
```

## 📝 为现有页面添加多语言支持 / Adding Multilingual Support to Existing Pages

### 步骤 1：更新页面头部 / Step 1: Update Page Header

```markdown
---
layout: default
title: {% t navigation.page_title %}  # 使用翻译键
description: {% t pages.page_description %}
---

{% include language-switcher.html %}
```

### 步骤 2：添加条件内容 / Step 2: Add Conditional Content

```markdown
{% if site.active_lang == 'zh' %}
# 中文标题

这里是中文内容...

{% else %}
# English Title

Here is English content...

{% endif %}
```

### 步骤 3：更新翻译文件 / Step 3: Update Translation Files

在 `_i18n/en.yml` 和 `_i18n/zh.yml` 中添加新的翻译键：

Add new translation keys to `_i18n/en.yml` and `_i18n/zh.yml`:

```yaml
pages:
  page_title: "Page Title"
  page_description: "Page description"
```

## 🚀 本地开发 / Local Development

### 安装依赖 / Install Dependencies

```bash
cd docs
bundle install
```

### 启动开发服务器 / Start Development Server

```bash
bundle exec jekyll serve --host 0.0.0.0 --port 4000
```

### 访问多语言页面 / Access Multilingual Pages

- 英文版本 / English: `http://localhost:4000/Sparse_to_Dense_Transformer/`
- 中文版本 / Chinese: `http://localhost:4000/Sparse_to_Dense_Transformer/zh/`

## 🎨 自定义样式 / Custom Styling

语言切换器的样式已包含在组件中，支持：

The language switcher styling is included in the component and supports:

- 响应式设计 / Responsive design
- 主题适配 / Theme adaptation
- 悬停效果 / Hover effects
- 活动状态指示 / Active state indication

## 📋 最佳实践 / Best Practices

1. **保持翻译文件同步** / Keep translation files synchronized
   - 确保所有语言版本都有相同的键 / Ensure all language versions have the same keys
   - 定期检查缺失的翻译 / Regularly check for missing translations

2. **使用语义化的翻译键** / Use semantic translation keys
   - 使用描述性的键名 / Use descriptive key names
   - 按功能分组组织 / Organize by functional groups

3. **条件内容的使用** / Using conditional content
   - 对于复杂内容使用条件语句 / Use conditionals for complex content
   - 对于简单文本使用翻译键 / Use translation keys for simple text

4. **测试所有语言版本** / Test all language versions
   - 确保所有链接在不同语言下正常工作 / Ensure all links work in different languages
   - 检查布局在不同语言下的表现 / Check layout performance in different languages

## 🔧 故障排除 / Troubleshooting

### 常见问题 / Common Issues

1. **翻译不显示** / Translations not showing
   - 检查翻译文件语法 / Check translation file syntax
   - 确认翻译键存在 / Confirm translation keys exist
   - 重启开发服务器 / Restart development server

2. **语言切换不工作** / Language switching not working
   - 检查 URL 结构 / Check URL structure
   - 确认插件正确安装 / Confirm plugin is properly installed
   - 检查 _config.yml 配置 / Check _config.yml configuration

3. **样式问题** / Styling issues
   - 检查 CSS 变量支持 / Check CSS variable support
   - 确认主题兼容性 / Confirm theme compatibility
   - 测试不同浏览器 / Test different browsers

## 📚 参考资源 / References

- [Jekyll Multiple Languages Plugin](https://github.com/kurtsson/jekyll-multiple-languages-plugin)
- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [Liquid Template Language](https://shopify.github.io/liquid/)
- [Just the Docs Theme](https://just-the-docs.github.io/just-the-docs/)

---

如有问题，请在 [GitHub Issues](https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer/issues) 中报告。

For issues, please report in [GitHub Issues](https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer/issues).