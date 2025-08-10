# 📋 GitHub Pages 部署检查清单

在部署VIVTransformer文档网站之前，请按照此检查清单确保所有配置都正确。

## ✅ 部署前检查

### 1. 基础配置

- [ ] **更新 `_config.yml` 中的用户信息**
  ```yaml
  baseurl: "/您的仓库名"  # 例如："/VIVTransformer"
  url: "https://您的用户名.github.io"  # 例如："https://username.github.io"
  ```

- [ ] **更新作者信息**
  ```yaml
  author:
    name: "您的姓名"
    email: "您的邮箱@example.com"
  ```

- [ ] **更新社交链接**
  ```yaml
  social:
    links:
      - https://github.com/您的用户名/您的仓库名
  ```

### 2. 文件结构检查

- [ ] **确认目录结构完整**
  ```
  docs/
  ├── _config.yml ✓
  ├── _layouts/default.html ✓
  ├── assets/css/custom.css ✓
  ├── pages/ ✓
  ├── index.md ✓
  ├── Gemfile ✓
  └── README.md ✓
  ```

- [ ] **检查pages目录下的文件**
  - [ ] 所有.md文件都有正确的Front Matter
  - [ ] 文件名使用小写和连字符（如：quick-start-tutorial.md）
  - [ ] 没有中文文件名或特殊字符

### 3. 内容检查

- [ ] **首页内容**
  - [ ] `index.md` 包含项目介绍
  - [ ] 导航链接指向正确的页面
  - [ ] 所有链接都使用相对路径

- [ ] **页面链接**
  - [ ] 内部链接格式：`[文本](page-name.html)`
  - [ ] 锚点链接格式：`[文本](#anchor-name)`
  - [ ] 没有指向.md文件的链接

- [ ] **图片和资源**
  - [ ] 所有图片都放在 `assets/images/` 目录
  - [ ] 图片链接使用相对路径
  - [ ] 图片文件大小合理（<1MB）

### 4. GitHub配置

- [ ] **仓库设置**
  - [ ] 仓库是公开的（Public）
  - [ ] 代码已推送到main/master分支
  - [ ] docs目录在根目录下

- [ ] **GitHub Pages设置**
  - [ ] 进入仓库 Settings → Pages
  - [ ] Source选择"Deploy from a branch"
  - [ ] Branch选择"main"或"master"
  - [ ] Folder选择"/docs"
  - [ ] 点击Save保存设置

### 5. GitHub Actions（可选）

- [ ] **工作流文件**
  - [ ] `.github/workflows/deploy-docs.yml` 存在
  - [ ] 工作流权限配置正确
  - [ ] 触发条件设置合理

## 🚀 部署步骤

### 步骤1：本地测试

```bash
# 进入docs目录
cd docs

# 安装依赖（首次运行）
bundle install

# 启动本地服务器
bundle exec jekyll serve

# 访问 http://localhost:4000 检查网站
```

**检查项目：**
- [ ] 网站能正常加载
- [ ] 所有页面都能访问
- [ ] 导航菜单工作正常
- [ ] 样式显示正确
- [ ] 没有404错误

### 步骤2：提交代码

```bash
# 添加所有文件
git add .

# 提交更改
git commit -m "Add GitHub Pages documentation site"

# 推送到远程仓库
git push origin main
```

### 步骤3：启用GitHub Pages

1. 进入GitHub仓库页面
2. 点击 **Settings** 标签
3. 滚动到 **Pages** 部分
4. 在 **Source** 下选择 **Deploy from a branch**
5. 选择 **main** 分支和 **/docs** 文件夹
6. 点击 **Save**

### 步骤4：等待部署

- [ ] **检查Actions标签页**
  - 查看部署工作流是否成功运行
  - 如有错误，查看日志并修复

- [ ] **访问网站**
  - 等待2-10分钟
  - 访问：`https://您的用户名.github.io/您的仓库名`
  - 检查所有功能是否正常

## 🔧 常见问题排查

### 问题1：页面显示404

**可能原因：**
- [ ] baseurl配置错误
- [ ] 文件路径不正确
- [ ] 分支或文件夹选择错误

**解决方案：**
```yaml
# 检查_config.yml中的配置
baseurl: "/exact-repository-name"  # 必须与仓库名完全一致
url: "https://username.github.io"   # 必须与GitHub用户名一致
```

### 问题2：样式不加载

**可能原因：**
- [ ] CSS文件路径错误
- [ ] baseurl配置问题

**解决方案：**
```html
<!-- 检查_layouts/default.html中的CSS链接 -->
<link rel="stylesheet" href="{{ "/assets/css/custom.css" | relative_url }}">
```

### 问题3：构建失败

**检查项目：**
- [ ] Gemfile语法正确
- [ ] Markdown语法无误
- [ ] 没有中文文件名
- [ ] Front Matter格式正确

### 问题4：链接失效

**检查项目：**
- [ ] 使用相对路径：`[文本](page-name.html)`
- [ ] 不要使用：`[文本](page-name.md)`
- [ ] 锚点链接：`[文本](#section-name)`

## 📊 部署后验证

### 功能测试

- [ ] **导航测试**
  - 所有导航链接都能正常工作
  - 面包屑导航显示正确
  - 侧边栏链接有效

- [ ] **内容测试**
  - 所有页面内容显示完整
  - 代码高亮正常工作
  - 数学公式渲染正确（如果有）
  - 图片正常加载

- [ ] **响应式测试**
  - 在手机上浏览正常
  - 平板设备显示良好
  - 桌面端布局合理

### 性能测试

- [ ] **加载速度**
  - 首页加载时间 < 3秒
  - 页面切换流畅
  - 图片加载及时

- [ ] **SEO检查**
  - 页面标题正确
  - Meta描述存在
  - 结构化数据完整

## 🎉 部署完成

恭喜！您的VIVTransformer文档网站已成功部署到GitHub Pages。

**网站地址：** `https://您的用户名.github.io/您的仓库名`

### 后续维护

- [ ] **定期更新内容**
  - 同步Wiki更改
  - 添加新功能文档
  - 修复用户反馈的问题

- [ ] **监控网站状态**
  - 检查链接有效性
  - 监控加载性能
  - 关注用户反馈

- [ ] **备份重要数据**
  - 定期备份配置文件
  - 保存自定义样式
  - 记录重要更改

---

**需要帮助？**
- 📖 查看 [Jekyll文档](https://jekyllrb.com/docs/)
- 🔧 查看 [GitHub Pages文档](https://docs.github.com/en/pages)
- 💬 在 [GitHub Issues](https://github.com/yourusername/VIVTransformer/issues) 提问