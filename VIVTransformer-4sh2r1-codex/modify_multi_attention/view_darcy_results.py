#!/usr/bin/env python3
"""
查看Darcy Flow可视化结果的简单脚本
"""

import os
from pathlib import Path
import subprocess
import sys

def list_visualization_files():
    """
    列出所有生成的可视化文件
    """
    current_dir = Path(".")
    png_files = list(current_dir.glob("darcy*.png"))
    
    print("=== Darcy Flow 可视化文件列表 ===")
    print(f"找到 {len(png_files)} 个可视化文件:\n")
    
    for i, file in enumerate(sorted(png_files), 1):
        file_size = file.stat().st_size / 1024  # KB
        print(f"{i:2d}. {file.name:<35} ({file_size:.1f} KB)")
    
    return sorted(png_files)

def open_file(file_path):
    """
    使用系统默认程序打开文件
    """
    try:
        if sys.platform.startswith('win'):
            os.startfile(file_path)
        elif sys.platform.startswith('darwin'):
            subprocess.run(['open', file_path])
        else:
            subprocess.run(['xdg-open', file_path])
        print(f"已打开文件: {file_path}")
    except Exception as e:
        print(f"无法打开文件 {file_path}: {e}")

def show_file_info(file_path):
    """
    显示文件的详细信息
    """
    file_path = Path(file_path)
    if file_path.exists():
        stat = file_path.stat()
        print(f"\n=== 文件信息: {file_path.name} ===")
        print(f"文件大小: {stat.st_size / 1024:.1f} KB")
        print(f"创建时间: {stat.st_ctime}")
        print(f"修改时间: {stat.st_mtime}")
        print(f"完整路径: {file_path.absolute()}")
    else:
        print(f"文件不存在: {file_path}")

def main():
    print("Darcy Flow 可视化结果查看器")
    print("=" * 50)
    
    files = list_visualization_files()
    
    if not files:
        print("\n没有找到可视化文件。请先运行可视化脚本。")
        return
    
    print("\n推荐查看顺序:")
    print("1. darcy_flow_multiple_samples.png - 多样本对比")
    print("2. darcy_flow_sample_0.png - 样本0详细视图")
    print("3. darcy_flow_sample_100.png - 样本100详细视图")
    
    while True:
        print("\n=== 操作选项 ===")
        print("1. 查看所有文件列表")
        print("2. 打开指定文件")
        print("3. 查看文件信息")
        print("4. 打开多样本对比图")
        print("5. 打开样本0详细图")
        print("6. 打开文件夹")
        print("0. 退出")
        
        choice = input("\n请选择操作 (0-6): ").strip()
        
        if choice == '0':
            print("再见！")
            break
        elif choice == '1':
            list_visualization_files()
        elif choice == '2':
            try:
                file_num = int(input(f"请输入文件编号 (1-{len(files)}): "))
                if 1 <= file_num <= len(files):
                    open_file(files[file_num - 1])
                else:
                    print("无效的文件编号")
            except ValueError:
                print("请输入有效的数字")
        elif choice == '3':
            try:
                file_num = int(input(f"请输入文件编号 (1-{len(files)}): "))
                if 1 <= file_num <= len(files):
                    show_file_info(files[file_num - 1])
                else:
                    print("无效的文件编号")
            except ValueError:
                print("请输入有效的数字")
        elif choice == '4':
            multi_file = Path("darcy_flow_multiple_samples.png")
            if multi_file.exists():
                open_file(multi_file)
            else:
                print("多样本对比图不存在")
        elif choice == '5':
            sample0_file = Path("darcy_flow_sample_0.png")
            if sample0_file.exists():
                open_file(sample0_file)
            else:
                print("样本0详细图不存在")
        elif choice == '6':
            current_dir = Path(".").absolute()
            open_file(current_dir)
        else:
            print("无效的选择，请重新输入")

if __name__ == '__main__':
    main()