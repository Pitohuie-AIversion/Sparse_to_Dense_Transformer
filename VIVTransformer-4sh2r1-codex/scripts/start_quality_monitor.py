#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIVTransformer 质量监控启动脚本

功能:
- 快速启动质量监控
- 环境检查和依赖安装
- 配置文件生成
- 服务管理

作者: VIVTransformer Team
日期: 2024
"""

import os
import sys
import json
import yaml
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QualityMonitorStarter:
    """质量监控启动器"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.scripts_dir = Path(__file__).parent
        self.config_file = self.project_path / 'quality_monitoring_config.yaml'
    
    def check_environment(self) -> bool:
        """检查环境"""
        print("🔍 检查环境...")
        
        # 检查Python版本
        if sys.version_info < (3, 7):
            print("❌ Python版本过低，需要Python 3.7+")
            return False
        
        print(f"✅ Python版本: {sys.version}")
        
        # 检查项目路径
        if not self.project_path.exists():
            print(f"❌ 项目路径不存在: {self.project_path}")
            return False
        
        print(f"✅ 项目路径: {self.project_path}")
        
        # 检查Git仓库
        if not (self.project_path / '.git').exists():
            print("⚠️  警告: 不是Git仓库，某些功能可能受限")
        else:
            print("✅ Git仓库检查通过")
        
        # 检查必要的脚本文件
        required_scripts = [
            'quality_monitor_system.py',
            'integrated_quality_system.py',
            'quality_dashboard.py',
            'quality_trend_analyzer.py'
        ]
        
        missing_scripts = []
        for script in required_scripts:
            if not (self.scripts_dir / script).exists():
                missing_scripts.append(script)
        
        if missing_scripts:
            print(f"❌ 缺少必要脚本: {', '.join(missing_scripts)}")
            return False
        
        print("✅ 脚本文件检查通过")
        
        return True
    
    def check_dependencies(self) -> bool:
        """检查依赖"""
        print("📦 检查依赖...")
        
        required_packages = [
            'pyyaml',
            'schedule',
            'requests',
            'matplotlib',
            'plotly',
            'pandas',
            'numpy'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package}")
        
        if missing_packages:
            print(f"\n缺少依赖包: {', '.join(missing_packages)}")
            
            if input("是否自动安装缺少的依赖? (y/n): ").lower() == 'y':
                return self._install_dependencies(missing_packages)
            else:
                print("请手动安装缺少的依赖包")
                return False
        
        return True
    
    def _install_dependencies(self, packages: List[str]) -> bool:
        """安装依赖"""
        try:
            print("📥 安装依赖包...")
            
            for package in packages:
                print(f"安装 {package}...")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"❌ 安装 {package} 失败: {result.stderr}")
                    return False
                
                print(f"✅ {package} 安装成功")
            
            print("✅ 所有依赖安装完成")
            return True
            
        except Exception as e:
            print(f"❌ 安装依赖失败: {e}")
            return False
    
    def generate_config(self) -> bool:
        """生成配置文件"""
        if self.config_file.exists():
            if input(f"配置文件已存在: {self.config_file}\n是否覆盖? (y/n): ").lower() != 'y':
                print("使用现有配置文件")
                return True
        
        print("📝 生成配置文件...")
        
        try:
            config = {
                'project': {
                    'name': 'VIVTransformer',
                    'path': str(self.project_path),
                    'description': 'VIVTransformer项目质量监控'
                },
                'monitoring': {
                    'interval': 300,
                    'auto_collect': True,
                    'retention_days': 90,
                    'branches': ['main', 'develop'],
                    'file_types': ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h'],
                    'exclude_patterns': [
                        '*/node_modules/*',
                        '*/venv/*',
                        '*/env/*',
                        '*/__pycache__/*',
                        '*/build/*',
                        '*/dist/*',
                        '*/target/*'
                    ]
                },
                'thresholds': {
                    'overall_score': {
                        'excellent': 90.0,
                        'good': 80.0,
                        'acceptable': 70.0,
                        'poor': 60.0,
                        'critical': 50.0
                    },
                    'code_quality': {
                        'min_score': 70.0,
                        'max_issues': 100,
                        'max_critical': 5
                    },
                    'security': {
                        'min_score': 80.0,
                        'max_vulnerabilities': 10,
                        'max_high_severity': 2
                    },
                    'performance': {
                        'min_score': 75.0,
                        'max_complexity': 10,
                        'max_response_time': 1000
                    },
                    'coverage': {
                        'min_line_coverage': 80.0,
                        'min_branch_coverage': 70.0,
                        'min_function_coverage': 85.0
                    },
                    'documentation': {
                        'min_coverage': 60.0,
                        'min_api_docs': 80.0
                    },
                    'team_collaboration': {
                        'min_score': 70.0,
                        'max_knowledge_gaps': 5
                    }
                },
                'alerts': {
                    'enabled': True,
                    'levels': ['critical', 'warning', 'info'],
                    'rules': [
                        {
                            'name': '总体质量过低',
                            'condition': 'overall_score < 60',
                            'level': 'critical',
                            'message': '项目总体质量分数过低，需要立即关注',
                            'cooldown': 3600
                        },
                        {
                            'name': '严重问题过多',
                            'condition': 'critical_issues > 5',
                            'level': 'critical',
                            'message': '发现过多严重问题，需要立即处理',
                            'cooldown': 1800
                        },
                        {
                            'name': '安全分数过低',
                            'condition': 'security_score < 70',
                            'level': 'warning',
                            'message': '安全分数过低，存在安全风险',
                            'cooldown': 3600
                        },
                        {
                            'name': '测试覆盖率不足',
                            'condition': 'coverage < 70',
                            'level': 'warning',
                            'message': '测试覆盖率不足，建议增加测试',
                            'cooldown': 7200
                        }
                    ],
                    'channels': {
                        'email': {
                            'enabled': False,
                            'smtp_server': 'smtp.gmail.com',
                            'smtp_port': 587,
                            'from_address': '',
                            'to_addresses': [],
                            'username': '',
                            'password': ''
                        },
                        'slack': {
                            'enabled': False,
                            'webhook_url': '',
                            'channel': '#quality-alerts',
                            'username': 'QualityBot',
                            'icon_emoji': ':robot_face:'
                        },
                        'webhook': {
                            'enabled': False,
                            'url': '',
                            'headers': {
                                'Content-Type': 'application/json'
                            }
                        }
                    }
                },
                'dashboard': {
                    'server': {
                        'host': '127.0.0.1',
                        'port': 8080,
                        'auto_open': True
                    },
                    'ui': {
                        'theme': 'light',
                        'refresh_interval': 30,
                        'show_trends': True,
                        'show_details': True
                    },
                    'charts': {
                        'enabled': True,
                        'types': ['line', 'bar', 'pie'],
                        'time_range': '30d'
                    },
                    'export': {
                        'formats': ['html', 'pdf', 'json'],
                        'auto_export': False,
                        'schedule': 'daily'
                    }
                },
                'database': {
                    'type': 'sqlite',
                    'path': 'quality_monitoring.db',
                    'backup': {
                        'enabled': True,
                        'interval': 'daily',
                        'retention': 30
                    },
                    'cleanup': {
                        'enabled': True,
                        'older_than_days': 90
                    }
                },
                'integrations': {
                    'git': {
                        'enabled': True,
                        'track_commits': True,
                        'track_branches': True
                    },
                    'ci_cd': {
                        'jenkins': {
                            'enabled': False,
                            'url': '',
                            'token': ''
                        },
                        'github_actions': {
                            'enabled': False,
                            'token': ''
                        }
                    }
                },
                'advanced': {
                    'parallel_processing': {
                        'enabled': True,
                        'max_workers': 4
                    },
                    'caching': {
                        'enabled': True,
                        'ttl': 3600
                    },
                    'performance': {
                        'timeout': 600,
                        'memory_limit': '2GB'
                    },
                    'debug': {
                        'enabled': False,
                        'log_level': 'INFO',
                        'save_logs': True
                    }
                }
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            print(f"✅ 配置文件已生成: {self.config_file}")
            return True
            
        except Exception as e:
            print(f"❌ 生成配置文件失败: {e}")
            return False
    
    def start_monitor(self, mode: str = 'check') -> bool:
        """启动监控"""
        try:
            monitor_script = self.scripts_dir / 'quality_monitor_system.py'
            
            if mode == 'monitor':
                print("🔄 启动持续监控...")
                cmd = [
                    sys.executable,
                    str(monitor_script),
                    '--monitor',
                    '--project', str(self.project_path),
                    '--config', str(self.config_file),
                    '--verbose'
                ]
                
                # 启动监控进程
                process = subprocess.Popen(cmd)
                
                print(f"✅ 监控已启动 (PID: {process.pid})")
                print("按 Ctrl+C 停止监控")
                
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\n⏹️  停止监控...")
                    process.terminate()
                    process.wait()
                
            elif mode == 'check':
                print("🔍 执行质量检查...")
                cmd = [
                    sys.executable,
                    str(monitor_script),
                    '--check',
                    '--project', str(self.project_path),
                    '--config', str(self.config_file)
                ]
                
                result = subprocess.run(cmd)
                return result.returncode == 0
                
            elif mode == 'dashboard':
                print("🚀 启动仪表板...")
                cmd = [
                    sys.executable,
                    str(monitor_script),
                    '--dashboard',
                    '--project', str(self.project_path),
                    '--config', str(self.config_file)
                ]
                
                subprocess.run(cmd)
                
            elif mode == 'report':
                print("📋 生成质量报告...")
                output_file = self.project_path / 'quality_report.html'
                
                cmd = [
                    sys.executable,
                    str(monitor_script),
                    '--report',
                    '--project', str(self.project_path),
                    '--config', str(self.config_file),
                    '--output', str(output_file),
                    '--format', 'html'
                ]
                
                result = subprocess.run(cmd)
                if result.returncode == 0:
                    print(f"✅ 报告已生成: {output_file}")
                return result.returncode == 0
            
            return True
            
        except Exception as e:
            print(f"❌ 启动监控失败: {e}")
            return False
    
    def show_status(self):
        """显示状态"""
        try:
            monitor_script = self.scripts_dir / 'quality_monitor_system.py'
            
            cmd = [
                sys.executable,
                str(monitor_script),
                '--status',
                '--project', str(self.project_path),
                '--config', str(self.config_file)
            ]
            
            subprocess.run(cmd)
            
        except Exception as e:
            print(f"❌ 获取状态失败: {e}")
    
    def interactive_setup(self):
        """交互式设置"""
        print("\n" + "="*60)
        print("🚀 VIVTransformer 质量监控系统 - 交互式设置")
        print("="*60)
        
        # 环境检查
        if not self.check_environment():
            print("❌ 环境检查失败")
            return False
        
        # 依赖检查
        if not self.check_dependencies():
            print("❌ 依赖检查失败")
            return False
        
        # 配置生成
        if not self.generate_config():
            print("❌ 配置生成失败")
            return False
        
        print("\n✅ 设置完成！")
        
        # 选择操作
        while True:
            print("\n" + "-"*40)
            print("请选择操作:")
            print("1. 执行质量检查")
            print("2. 启动持续监控")
            print("3. 启动仪表板")
            print("4. 生成质量报告")
            print("5. 显示系统状态")
            print("6. 退出")
            print("-"*40)
            
            choice = input("请输入选择 (1-6): ").strip()
            
            if choice == '1':
                self.start_monitor('check')
            elif choice == '2':
                self.start_monitor('monitor')
            elif choice == '3':
                self.start_monitor('dashboard')
            elif choice == '4':
                self.start_monitor('report')
            elif choice == '5':
                self.show_status()
            elif choice == '6':
                print("👋 再见！")
                break
            else:
                print("❌ 无效选择，请重新输入")
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="VIVTransformer 质量监控启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --setup .
  %(prog)s --quick-start .
  %(prog)s --check .
  %(prog)s --monitor .
        """
    )
    
    parser.add_argument(
        'project_path',
        nargs='?',
        default='.',
        help='项目路径'
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='交互式设置'
    )
    
    parser.add_argument(
        '--quick-start',
        action='store_true',
        help='快速启动（自动设置并执行检查）'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='执行质量检查'
    )
    
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='启动持续监控'
    )
    
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='启动仪表板'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='生成质量报告'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='显示系统状态'
    )
    
    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='仅安装依赖'
    )
    
    parser.add_argument(
        '--generate-config',
        action='store_true',
        help='仅生成配置文件'
    )
    
    args = parser.parse_args()
    
    try:
        starter = QualityMonitorStarter(args.project_path)
        
        if args.setup:
            starter.interactive_setup()
        
        elif args.quick_start:
            print("🚀 快速启动质量监控...")
            
            if (starter.check_environment() and 
                starter.check_dependencies() and 
                starter.generate_config()):
                
                print("\n✅ 设置完成，开始质量检查...")
                starter.start_monitor('check')
            else:
                print("❌ 快速启动失败")
                sys.exit(1)
        
        elif args.install_deps:
            starter.check_dependencies()
        
        elif args.generate_config:
            starter.generate_config()
        
        elif args.check:
            if not starter.start_monitor('check'):
                sys.exit(1)
        
        elif args.monitor:
            starter.start_monitor('monitor')
        
        elif args.dashboard:
            starter.start_monitor('dashboard')
        
        elif args.report:
            if not starter.start_monitor('report'):
                sys.exit(1)
        
        elif args.status:
            starter.show_status()
        
        else:
            # 默认交互式设置
            starter.interactive_setup()
    
    except KeyboardInterrupt:
        print("\n⏹️  操作被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"启动脚本运行失败: {e}")
        print(f"❌ 启动脚本运行失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()