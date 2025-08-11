# VIVTransformer Documentation Site

This is the GitHub Pages documentation site for the VIVTransformer project. The documentation site is built with Jekyll and provides comprehensive project documentation, API reference, and usage guides.

## 🚀 Quick Deployment

### Method 1: Automatic GitHub Pages Deployment (Recommended)

1. **Update Configuration File**
```bash
# Edit _config.yml
# Replace 'yourusername' with your GitHub username
# Replace 'VIVTransformer' with your repository name
```

2. **Commit Code to GitHub**
```bash
git add .
git commit -m "Setup GitHub Pages documentation"
git push origin main
```

3. **Enable GitHub Pages**
- Go to GitHub repository settings page
- Scroll to "Pages" section
- Under "Source", select "Deploy from a branch"
- Choose "main" branch and "/docs" folder
- Click "Save"

4. **Access Website**
- The website will be available in a few minutes
- Access URL: `https://yourusername.github.io/VIVTransformer`

### Method 2: Local Development and Preview

1. **Install Dependencies**
```bash
# Ensure Ruby 3.0+ is installed
bundle install
```

2. **Local Run**
```bash
bundle exec jekyll serve
# Or specify port
bundle exec jekyll serve --port 4000
```

3. **Access Local Website**
- Open browser and visit: `http://localhost:4000`

## 📁 Directory Structure

```
docs/
├── _config.yml          # Jekyll configuration file
├── _layouts/             # Page layout templates
│   └── default.html      # Default layout
├── assets/               # Static resources
│   └── css/
│       └── custom.css    # Custom styles
├── pages/                # Documentation pages
│   ├── *.md             # Various documentation files
│   └── ...
├── index.md              # Homepage
├── Gemfile               # Ruby dependencies
├── convert_wiki.py       # Wiki conversion script
└── README.md             # This file
```

## 🔧 Custom Configuration

### Update Site Information

Edit `_config.yml` file:

```yaml
# Basic information
title: "Your Project Name"
description: "Project description"
baseurl: "/YourRepositoryName"
url: "https://yourusername.github.io"

# Author information
author:
  name: "Your Name"
  email: "your.email@example.com"

# Social links
github:
  name: "Project Name"
  repository_url: "https://github.com/yourusername/yourrepository"
```

### Custom Styles

Edit `assets/css/custom.css` file to customize website appearance:

```css
/* Custom primary color */
:root {
  --primary-color: #your-color;
}

/* Custom font */
body {
  font-family: 'Your Font', sans-serif;
}
```

### Add New Pages

1. Create new `.md` file in `pages/` directory
2. Add Jekyll Front Matter:

```yaml
---
title: "Page Title"
description: "Page description"
---
```

3. Write Markdown content
4. Update navigation menu (in `_layouts/default.html`)

## 🔄 Sync Content from Wiki

If you have existing Wiki content, you can use the provided conversion script:

```bash
python convert_wiki.py
```

This script will:
- Convert Wiki files to Jekyll format
- Add appropriate Front Matter
- Fix internal links
- Generate sitemap

## 🎨 Theme and Styles

### Current Theme Features

- 📱 Responsive design with mobile support
- 🌙 Automatic dark mode support
- 🔍 Auto-generated table of contents
- 💫 Smooth scrolling and animation effects
- 📊 Code highlighting and math formula support
- 🔗 Breadcrumb navigation
- 📋 Sidebar quick links

### Available CSS Classes

```html
<!-- Alert boxes -->
<div class="alert alert-info">Information message</div>
<div class="alert alert-warning">Warning message</div>
<div class="alert alert-error">Error message</div>
<div class="alert alert-success">Success message</div>

<!-- Utility classes -->
<p class="text-center">Centered text</p>
<p class="text-muted">Muted text</p>
```

## 🚀 Advanced Features

### Enable Google Analytics

Add to `_config.yml`:

```yaml
google_analytics: UA-XXXXXXXX-X
```

### Add Search Functionality

You can integrate the following search solutions:
- Algolia DocSearch
- Lunr.js
- Google Custom Search

### Multi-language Support

You can use [jekyll-multiple-languages-plugin](https://github.com/kurtsson/jekyll-multiple-languages-plugin) to add multi-language support.

## 🔧 Troubleshooting

### Common Issues

1. **Page 404 Error**
- Check if `baseurl` configuration is correct
- Confirm file path and permalink settings

2. **Styles Not Loading**
- Check CSS file paths
- Confirm URL settings in `_config.yml`

3. **Build Failure**
- Check Gemfile dependencies
- View GitHub Actions logs
- Confirm Markdown syntax is correct

4. **Broken Links**
- Use relative paths instead of absolute paths
- Check file name case sensitivity

### Debug Commands

```bash
# Check Jekyll configuration
bundle exec jekyll doctor

# Detailed build information
bundle exec jekyll build --verbose

# Check dependencies
bundle check
```

## 📚 Related Resources

- [Jekyll Official Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Markdown Syntax Guide](https://www.markdownguide.org/)
- [Jekyll Themes](https://jekyllthemes.io/)
- [Liquid Template Language](https://shopify.github.io/liquid/)

## 🤝 Contributing

Welcome to contribute documentation content! Please:

1. Fork this repository
2. Create a feature branch
3. Submit changes
4. Create a Pull Request

## 📄 License

This documentation site uses the same license as the main project.

---

*Need help? Please check [GitHub Issues](https://github.com/yourusername/VIVTransformer/issues) or contact maintainers.*