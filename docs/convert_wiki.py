#!/usr/bin/env python3
"""
Wiki to Jekyll Pages Converter
将现有的Wiki文件转换为Jekyll格式的页面
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# 配置
WIKI_DIR = Path("../wiki")
PAGES_DIR = Path("pages")
BASE_URL = "/VIVTransformer"

# 页面元数据映射
PAGE_METADATA = {
    "Architecture-Overview.md": {
        "title": "Architecture Overview",
        "description": "VIVTransformer整体架构设计和核心组件",
        "permalink": "/pages/architecture-overview/"
    },
    "Attention-Mechanisms-Guide.md": {
        "title": "Attention Mechanisms Guide",
        "description": "多头注意力机制的实现和使用指南",
        "permalink": "/pages/attention-mechanisms-guide/"
    },
    "Configuration-System.md": {
        "title": "Configuration System",
        "description": "配置文件系统的使用和管理",
        "permalink": "/pages/configuration-system/"
    },
    "Convergence-Analysis.md": {
        "title": "Convergence Analysis",
        "description": "训练收敛性分析方法和工具",
        "permalink": "/pages/convergence-analysis/"
    },
    "Custom-Attention.md": {
        "title": "Custom Attention",
        "description": "自定义注意力机制的实现指南",
        "permalink": "/pages/custom-attention/"
    },
    "Data-Pipeline.md": {
        "title": "Data Pipeline",
        "description": "数据处理管道的设计和实现",
        "permalink": "/pages/data-pipeline/"
    },
    "Deployment-Guide.md": {
        "title": "Deployment Guide",
        "description": "生产环境部署的完整指南",
        "permalink": "/pages/deployment-guide/"
    },
    "Development-Guide.md": {
        "title": "Development Guide",
        "description": "开发环境配置和贡献指南",
        "permalink": "/pages/development-guide/"
    },
    "Evaluation-Metrics.md": {
        "title": "Evaluation Metrics",
        "description": "模型评估指标和方法",
        "permalink": "/pages/evaluation-metrics/"
    },
    "Experimental-Results.md": {
        "title": "Experimental Results",
        "description": "实验结果和性能分析",
        "permalink": "/pages/experimental-results/"
    },
    "FAQ.md": {
        "title": "FAQ",
        "description": "常见问题和解答",
        "permalink": "/pages/faq/"
    },
    "Hyperparameter-Tuning.md": {
        "title": "Hyperparameter Tuning",
        "description": "超参数调优策略和工具",
        "permalink": "/pages/hyperparameter-tuning/"
    },
    "Implementation-Details.md": {
        "title": "Implementation Details",
        "description": "核心实现细节和技术说明",
        "permalink": "/pages/implementation-details/"
    },
    "Loss-Functions.md": {
        "title": "Loss Functions",
        "description": "损失函数的设计和实现",
        "permalink": "/pages/loss-functions/"
    },
    "Model-Design.md": {
        "title": "Model Design",
        "description": "模型设计原理和架构细节",
        "permalink": "/pages/model-design/"
    },
    "Multi-Loss-Strategy.md": {
        "title": "Multi-Loss Strategy",
        "description": "多损失函数策略和配置",
        "permalink": "/pages/multi-loss-strategy/"
    },
    "Performance-Comparison.md": {
        "title": "Performance Comparison",
        "description": "性能对比和基准测试",
        "permalink": "/pages/performance-comparison/"
    },
    "Quick-Start-Tutorial.md": {
        "title": "Quick Start Tutorial",
        "description": "5分钟快速上手指南",
        "permalink": "/pages/quick-start-tutorial/"
    },
    "SVD-Loss-Functions.md": {
        "title": "SVD Loss Functions",
        "description": "SVD损失函数的理论和实现",
        "permalink": "/pages/svd-loss-functions/"
    },
    "Training-Guide.md": {
        "title": "Training Guide",
        "description": "完整的模型训练指南",
        "permalink": "/pages/training-guide/"
    },
    "Troubleshooting.md": {
        "title": "Troubleshooting",
        "description": "故障排除和问题解决",
        "permalink": "/pages/troubleshooting/"
    }
}

def create_front_matter(filename: str) -> str:
    """创建Jekyll Front Matter"""
    metadata = PAGE_METADATA.get(filename, {
        "title": filename.replace("-", " ").replace(".md", ""),
        "description": f"{filename.replace('.md', '')} documentation",
        "permalink": f"/pages/{filename.lower().replace('.md', '').replace('_', '-')}/"
    })
    
    front_matter = "---\n"
    front_matter += "layout: default\n"
    front_matter += f"title: {metadata['title']}\n"
    front_matter += f"description: {metadata['description']}\n"
    front_matter += f"permalink: {metadata['permalink']}\n"
    front_matter += "---\n\n"
    
    return front_matter

def fix_internal_links(content: str) -> str:
    """修复内部链接格式"""
    # Wiki链接格式: [Text](Page-Name.md) -> Jekyll格式: [Text](page-name.html)
    def replace_wiki_link(match):
        text = match.group(1)
        filename = match.group(2)
        
        # 跳过外部链接
        if filename.startswith(('http://', 'https://', 'mailto:')):
            return match.group(0)
        
        # 转换为Jekyll格式
        if filename.endswith('.md'):
            page_name = filename.replace('.md', '').lower().replace('_', '-')
            return f"[{text}]({page_name}.html)"
        
        return match.group(0)
    
    # 匹配Markdown链接
    content = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', replace_wiki_link, content)
    
    return content

def fix_anchor_links(content: str) -> str:
    """修复锚点链接"""
    # 将 {#anchor} 格式转换为标准的锚点
    content = re.sub(r'\{#([^}]+)\}', r'', content)
    
    # 为标题添加ID
    def add_header_id(match):
        level = len(match.group(1))
        title = match.group(2).strip()
        header_id = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]+', '-', title).lower().strip('-')
        return f"{'#' * level} {title} {{#{header_id}}}"
    
    content = re.sub(r'^(#{1,6})\s+(.+)$', add_header_id, content, flags=re.MULTILINE)
    
    return content

def process_code_blocks(content: str) -> str:
    """处理代码块"""
    # 确保代码块有正确的语言标识
    def fix_code_block(match):
        lang = match.group(1) or ''
        code = match.group(2)
        
        # 自动检测语言
        if not lang:
            if 'import ' in code and ('torch' in code or 'numpy' in code):
                lang = 'python'
            elif 'function' in code or 'const ' in code or 'let ' in code:
                lang = 'javascript'
            elif 'class ' in code and '{' in code:
                lang = 'java'
            elif '#include' in code:
                lang = 'cpp'
            elif 'def ' in code or 'import ' in code:
                lang = 'python'
            elif code.strip().startswith('$') or 'sudo' in code:
                lang = 'bash'
            elif code.strip().startswith('<') and code.strip().endswith('>'):
                lang = 'xml'
        
        return f"```{lang}\n{code}\n```"
    
    content = re.sub(r'```([^\n]*)\n(.*?)\n```', fix_code_block, content, flags=re.DOTALL)
    
    return content

def add_navigation_hints(content: str, filename: str) -> str:
    """添加导航提示"""
    # 在文档末尾添加相关链接
    related_links = {
        "Quick-Start-Tutorial.md": [
            ("Architecture Overview", "architecture-overview.html"),
            ("Training Guide", "training-guide.html"),
            ("Configuration System", "configuration-system.html")
        ],
        "Architecture-Overview.md": [
            ("Model Design", "model-design.html"),
            ("Implementation Details", "implementation-details.html"),
            ("Attention Mechanisms", "attention-mechanisms-guide.html")
        ],
        "Training-Guide.md": [
            ("Loss Functions", "loss-functions.html"),
            ("Hyperparameter Tuning", "hyperparameter-tuning.html"),
            ("Convergence Analysis", "convergence-analysis.html")
        ],
        "Deployment-Guide.md": [
            ("Performance Comparison", "performance-comparison.html"),
            ("Troubleshooting", "troubleshooting.html"),
            ("Configuration System", "configuration-system.html")
        ]
    }
    
    if filename in related_links:
        content += "\n\n## 📚 相关文档\n\n"
        for title, link in related_links[filename]:
            content += f"- [{title}]({link})\n"
    
    # 添加通用的帮助链接
    content += "\n\n---\n\n"
    content += "*需要帮助？查看 [FAQ](faq.html) 或 [故障排除](troubleshooting.html) 页面。*\n"
    
    return content

def convert_file(wiki_file: Path, output_file: Path) -> bool:
    """转换单个文件"""
    try:
        # 读取原文件
        with open(wiki_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 跳过特殊文件
        if wiki_file.name in ['Home.md', '_Footer.md']:
            print(f"跳过特殊文件: {wiki_file.name}")
            return True
        
        # 创建Front Matter
        front_matter = create_front_matter(wiki_file.name)
        
        # 处理内容
        content = fix_internal_links(content)
        content = fix_anchor_links(content)
        content = process_code_blocks(content)
        content = add_navigation_hints(content, wiki_file.name)
        
        # 组合最终内容
        final_content = front_matter + content
        
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print(f"✅ 转换完成: {wiki_file.name} -> {output_file.name}")
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {wiki_file.name} - {str(e)}")
        return False

def create_sitemap() -> str:
    """创建站点地图"""
    sitemap_content = "---\nlayout: default\ntitle: Site Map\npermalink: /sitemap/\n---\n\n"
    sitemap_content += "# 站点地图\n\n"
    
    categories = {
        "🚀 快速开始": ["quick-start-tutorial.html"],
        "🏗️ 架构设计": [
            "architecture-overview.html",
            "model-design.html",
            "attention-mechanisms-guide.html",
            "custom-attention.html"
        ],
        "📚 训练指南": [
            "training-guide.html",
            "loss-functions.html",
            "svd-loss-functions.html",
            "multi-loss-strategy.html",
            "hyperparameter-tuning.html",
            "convergence-analysis.html"
        ],
        "⚙️ 配置与实现": [
            "configuration-system.html",
            "implementation-details.html",
            "data-pipeline.html"
        ],
        "📊 评估与部署": [
            "evaluation-metrics.html",
            "experimental-results.html",
            "performance-comparison.html",
            "deployment-guide.html"
        ],
        "🛠️ 开发与维护": [
            "development-guide.html",
            "troubleshooting.html",
            "faq.html"
        ]
    }
    
    for category, pages in categories.items():
        sitemap_content += f"\n## {category}\n\n"
        for page in pages:
            title = PAGE_METADATA.get(page.replace('.html', '.md'), {}).get('title', page)
            sitemap_content += f"- [{title}]({page})\n"
    
    return sitemap_content

def main():
    """主函数"""
    print("🚀 开始转换Wiki文件到Jekyll格式...\n")
    
    # 检查目录
    if not WIKI_DIR.exists():
        print(f"❌ Wiki目录不存在: {WIKI_DIR}")
        return
    
    # 创建输出目录
    PAGES_DIR.mkdir(exist_ok=True)
    
    # 转换文件
    success_count = 0
    total_count = 0
    
    for wiki_file in WIKI_DIR.glob("*.md"):
        if wiki_file.name.startswith('_') or wiki_file.name == 'Home.md':
            continue
            
        total_count += 1
        output_filename = wiki_file.name.lower().replace('_', '-')
        output_file = PAGES_DIR / output_filename
        
        if convert_file(wiki_file, output_file):
            success_count += 1
    
    # 创建站点地图
    sitemap_content = create_sitemap()
    with open(PAGES_DIR / "sitemap.md", 'w', encoding='utf-8') as f:
        f.write(sitemap_content)
    print("✅ 创建站点地图: sitemap.md")
    
    # 总结
    print(f"\n📊 转换完成统计:")
    print(f"   成功: {success_count}/{total_count} 个文件")
    print(f"   输出目录: {PAGES_DIR.absolute()}")
    
    if success_count == total_count:
        print("\n🎉 所有文件转换成功！")
        print("\n📋 下一步操作:")
        print("1. 检查生成的页面文件")
        print("2. 更新 _config.yml 中的 GitHub 用户名和仓库名")
        print("3. 提交代码到 GitHub")
        print("4. 在仓库设置中启用 GitHub Pages")
        print("5. 选择 'Deploy from a branch' 并选择 main 分支的 /docs 文件夹")
    else:
        print(f"\n⚠️  有 {total_count - success_count} 个文件转换失败，请检查错误信息")

if __name__ == "__main__":
    main()