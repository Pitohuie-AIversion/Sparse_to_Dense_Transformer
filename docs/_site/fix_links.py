#!/usr/bin/env python3
"""
修复Jekyll文档中的.html链接问题
"""

import os
import re
from pathlib import Path

def fix_html_links_in_file(file_path):
    """修复单个文件中的.html链接"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修复Markdown链接中的.html扩展名
        # 匹配模式: [text](filename.html)
        pattern = r'\[([^\]]+)\]\(([^\)]+)\.html\)'
        
        def replace_link(match):
            text = match.group(1)
            link = match.group(2)
            # 如果链接包含路径分隔符，保留路径部分
            if '/' in link:
                return f'[{text}]({link})'
            else:
                return f'[{text}]({link})'
        
        new_content = re.sub(pattern, replace_link, content)
        
        # 只有内容发生变化时才写入文件
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅ 修复了 {file_path}")
            return True
        else:
            print(f"⏭️  跳过 {file_path} (无需修复)")
            return False
            
    except Exception as e:
        print(f"❌ 处理 {file_path} 时出错: {e}")
        return False

def main():
    """主函数"""
    docs_dir = Path(__file__).parent
    pages_dir = docs_dir / 'pages'
    
    if not pages_dir.exists():
        print(f"❌ 页面目录不存在: {pages_dir}")
        return
    
    print("🔧 开始修复Jekyll文档中的.html链接...")
    print(f"📁 扫描目录: {pages_dir}")
    
    fixed_count = 0
    total_count = 0
    
    # 遍历所有.md文件
    for md_file in pages_dir.glob('*.md'):
        total_count += 1
        if fix_html_links_in_file(md_file):
            fixed_count += 1
    
    print(f"\n📊 修复完成!")
    print(f"   总文件数: {total_count}")
    print(f"   修复文件数: {fixed_count}")
    print(f"   跳过文件数: {total_count - fixed_count}")
    
    if fixed_count > 0:
        print("\n🎉 所有.html链接已修复为Jekyll兼容格式!")
        print("💡 提示: 重新构建Jekyll站点以查看更改")
    else:
        print("\n✨ 所有链接已经是正确格式!")

if __name__ == '__main__':
    main()