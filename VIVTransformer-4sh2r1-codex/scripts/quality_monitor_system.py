#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIVTransformer 质量监控系统

功能:
- 统一质量监控入口
- 实时质量数据收集
- 自动化质量检查
- 质量趋势分析
- 报警和通知
- 质量报告生成
- Web仪表板服务

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
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    schedule = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """监控配置"""
    project_path: str
    config_file: str = 'quality_monitoring_config.yaml'
    interval: int = 300  # 监控间隔（秒）
    auto_collect: bool = True
    retention_days: int = 90
    branches: List[str] = None
    
    def __post_init__(self):
        if self.branches is None:
            self.branches = ['main', 'develop']


@dataclass
class QualityMetrics:
    """质量指标"""
    timestamp: str
    overall_score: float
    dimensions: Dict[str, float]
    issues_count: int
    critical_issues: int
    security_score: float
    performance_score: float
    coverage: float
    technical_debt: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlertRule:
    """报警规则"""
    name: str
    condition: str
    level: str  # critical, warning, info
    message: str
    cooldown: int = 3600  # 冷却时间（秒）
    last_triggered: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_file):
            logger.warning(f"配置文件不存在: {self.config_file}，使用默认配置")
            return self._get_default_config()
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            logger.info(f"配置文件加载成功: {self.config_file}")
            return config
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'monitoring': {
                'interval': 300,
                'auto_collect': True,
                'retention_days': 90,
                'branches': ['main', 'develop']
            },
            'thresholds': {
                'overall_score': {
                    'excellent': 90.0,
                    'good': 80.0,
                    'acceptable': 70.0,
                    'poor': 60.0,
                    'critical': 50.0
                }
            },
            'alerts': {
                'enabled': True,
                'rules': []
            },
            'dashboard': {
                'server': {
                    'host': '127.0.0.1',
                    'port': 8080
                }
            }
        }
    
    def get(self, key: str, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_alert_rules(self) -> List[AlertRule]:
        """获取报警规则"""
        rules_config = self.get('alerts.rules', [])
        rules = []
        
        for rule_config in rules_config:
            rule = AlertRule(
                name=rule_config.get('name', ''),
                condition=rule_config.get('condition', ''),
                level=rule_config.get('level', 'info'),
                message=rule_config.get('message', ''),
                cooldown=rule_config.get('cooldown', 3600)
            )
            rules.append(rule)
        
        return rules


class QualityDataCollector:
    """质量数据收集器"""
    
    def __init__(self, project_path: str, config_manager: ConfigManager):
        self.project_path = project_path
        self.config_manager = config_manager
        self.scripts_dir = Path(__file__).parent
    
    def collect_current_metrics(self) -> Optional[QualityMetrics]:
        """收集当前质量指标"""
        try:
            logger.info("开始收集质量指标...")
            
            # 使用绝对路径的临时文件
            temp_file = os.path.abspath('temp_metrics.json')
            
            # 运行集成质量分析
            result = subprocess.run([
                sys.executable,
                str(self.scripts_dir / 'integrated_quality_system.py'),
                '--analyze', self.project_path,
                '--format', 'json',
                '--output', temp_file
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"质量分析失败: {result.stderr}")
                logger.error(f"质量分析输出: {result.stdout}")
                # 即使返回码非零，如果文件存在也继续处理
                if not os.path.exists(temp_file):
                    return None
            
            if not os.path.exists(temp_file):
                logger.error(f"质量分析结果文件不存在: {temp_file}")
                return None
            
            # 读取分析结果
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 清理临时文件
            try:
                os.remove(temp_file)
            except:
                pass  # 忽略删除失败
            
            # 提取指标
            dimensions = {}
            for dimension in data.get('dimensions', []):
                dimension_name = dimension['name'].lower().replace(' ', '_')
                dimensions[dimension_name] = dimension['score']
            
            summary = data.get('summary', {})
            
            metrics = QualityMetrics(
                timestamp=datetime.now().isoformat(),
                overall_score=data.get('overall_score', 0.0),
                dimensions=dimensions,
                issues_count=summary.get('total_issues', 0),
                critical_issues=summary.get('critical_issues', 0),
                security_score=dimensions.get('security', 0.0),
                performance_score=dimensions.get('performance', 0.0),
                coverage=summary.get('coverage', 0.0),
                technical_debt=summary.get('technical_debt', 0.0)
            )
            
            logger.info(f"质量指标收集完成: {metrics.overall_score:.2f}分")
            return metrics
            
        except Exception as e:
            logger.error(f"收集质量指标失败: {e}")
            return None
    
    def save_metrics(self, metrics: QualityMetrics):
        """保存质量指标"""
        try:
            # 使用趋势分析器保存数据
            from quality_trend_analyzer import QualityDatabase, QualitySnapshot
            
            db = QualityDatabase()
            
            snapshot = QualitySnapshot(
                timestamp=metrics.timestamp,
                commit_hash=self._get_current_commit(),
                branch=self._get_current_branch(),
                overall_score=metrics.overall_score,
                dimensions=metrics.dimensions,
                metrics=metrics.to_dict(),
                issues_count=metrics.issues_count,
                critical_issues=metrics.critical_issues
            )
            
            db.save_snapshot(snapshot)
            logger.info("质量指标已保存到数据库")
            
        except Exception as e:
            logger.error(f"保存质量指标失败: {e}")
    
    def _get_current_commit(self) -> str:
        """获取当前提交哈希"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.project_path
            )
            return result.stdout.strip() if result.returncode == 0 else ''
        except:
            return ''
    
    def _get_current_branch(self) -> str:
        """获取当前分支"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.project_path
            )
            return result.stdout.strip() if result.returncode == 0 else ''
        except:
            return ''


class AlertManager:
    """报警管理器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.alert_rules = config_manager.get_alert_rules()
        self.alert_history = []
    
    def check_alerts(self, metrics: QualityMetrics) -> List[Dict[str, Any]]:
        """检查报警条件"""
        triggered_alerts = []
        
        if not self.config_manager.get('alerts.enabled', True):
            return triggered_alerts
        
        current_time = datetime.now()
        
        for rule in self.alert_rules:
            try:
                # 检查冷却时间
                if rule.last_triggered:
                    last_time = datetime.fromisoformat(rule.last_triggered)
                    if (current_time - last_time).total_seconds() < rule.cooldown:
                        continue
                
                # 评估条件
                if self._evaluate_condition(rule.condition, metrics):
                    alert = {
                        'rule_name': rule.name,
                        'level': rule.level,
                        'message': rule.message,
                        'timestamp': current_time.isoformat(),
                        'metrics': metrics.to_dict()
                    }
                    
                    triggered_alerts.append(alert)
                    rule.last_triggered = current_time.isoformat()
                    
                    logger.warning(f"触发报警: {rule.name} - {rule.message}")
            
            except Exception as e:
                logger.error(f"评估报警规则失败 {rule.name}: {e}")
        
        # 保存报警历史
        self.alert_history.extend(triggered_alerts)
        
        return triggered_alerts
    
    def _evaluate_condition(self, condition: str, metrics: QualityMetrics) -> bool:
        """评估报警条件"""
        try:
            # 创建安全的评估环境
            context = {
                'overall_score': metrics.overall_score,
                'issues_count': metrics.issues_count,
                'critical_issues': metrics.critical_issues,
                'security_score': metrics.security_score,
                'performance_score': metrics.performance_score,
                'coverage': metrics.coverage,
                'technical_debt': metrics.technical_debt
            }
            
            # 添加维度数据
            for key, value in metrics.dimensions.items():
                context[key] = value
            
            # 安全评估条件
            allowed_names = {
                '__builtins__': {},
                **context
            }
            
            return eval(condition, allowed_names)
            
        except Exception as e:
            logger.error(f"评估条件失败 '{condition}': {e}")
            return False
    
    def send_alerts(self, alerts: List[Dict[str, Any]]):
        """发送报警"""
        if not alerts:
            return
        
        for alert in alerts:
            try:
                # 发送邮件报警
                if self.config_manager.get('alerts.channels.email.enabled', False):
                    self._send_email_alert(alert)
                
                # 发送Slack报警
                if self.config_manager.get('alerts.channels.slack.enabled', False):
                    self._send_slack_alert(alert)
                
                # 发送Webhook报警
                if self.config_manager.get('alerts.channels.webhook.enabled', False):
                    self._send_webhook_alert(alert)
                
            except Exception as e:
                logger.error(f"发送报警失败: {e}")
    
    def _send_email_alert(self, alert: Dict[str, Any]):
        """发送邮件报警"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # 获取邮件配置
            email_config = self.config_manager.get('alerts.channels.email', {})
            
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = email_config.get('from_address', '')
            msg['To'] = ', '.join(email_config.get('to_addresses', []))
            msg['Subject'] = f"[{alert['level'].upper()}] VIVTransformer 质量报警"
            
            # 邮件内容
            body = f"""
质量报警通知

报警规则: {alert['rule_name']}
报警级别: {alert['level'].upper()}
报警消息: {alert['message']}
触发时间: {alert['timestamp']}

当前质量指标:
- 总体分数: {alert['metrics']['overall_score']:.2f}
- 问题数量: {alert['metrics']['issues_count']}
- 严重问题: {alert['metrics']['critical_issues']}
- 安全分数: {alert['metrics']['security_score']:.2f}
- 性能分数: {alert['metrics']['performance_score']:.2f}

请及时处理相关问题。

-- VIVTransformer 质量监控系统
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 发送邮件
            server = smtplib.SMTP(email_config.get('smtp_server', ''), email_config.get('smtp_port', 587))
            server.starttls()
            server.login(email_config.get('username', ''), email_config.get('password', ''))
            server.send_message(msg)
            server.quit()
            
            logger.info(f"邮件报警已发送: {alert['rule_name']}")
            
        except Exception as e:
            logger.error(f"发送邮件报警失败: {e}")
    
    def _send_slack_alert(self, alert: Dict[str, Any]):
        """发送Slack报警"""
        try:
            import requests
            
            slack_config = self.config_manager.get('alerts.channels.slack', {})
            webhook_url = slack_config.get('webhook_url', '')
            
            if not webhook_url:
                return
            
            # 确定颜色
            color_map = {
                'critical': '#FF0000',
                'warning': '#FFA500',
                'info': '#0000FF'
            }
            color = color_map.get(alert['level'], '#808080')
            
            # 构建消息
            payload = {
                'channel': slack_config.get('channel', '#quality-alerts'),
                'username': slack_config.get('username', 'QualityBot'),
                'icon_emoji': slack_config.get('icon_emoji', ':robot_face:'),
                'attachments': [{
                    'color': color,
                    'title': f"[{alert['level'].upper()}] 质量报警",
                    'text': alert['message'],
                    'fields': [
                        {
                            'title': '报警规则',
                            'value': alert['rule_name'],
                            'short': True
                        },
                        {
                            'title': '总体分数',
                            'value': f"{alert['metrics']['overall_score']:.2f}",
                            'short': True
                        },
                        {
                            'title': '问题数量',
                            'value': str(alert['metrics']['issues_count']),
                            'short': True
                        },
                        {
                            'title': '严重问题',
                            'value': str(alert['metrics']['critical_issues']),
                            'short': True
                        }
                    ],
                    'footer': 'VIVTransformer 质量监控',
                    'ts': int(datetime.now().timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack报警已发送: {alert['rule_name']}")
            
        except Exception as e:
            logger.error(f"发送Slack报警失败: {e}")
    
    def _send_webhook_alert(self, alert: Dict[str, Any]):
        """发送Webhook报警"""
        try:
            import requests
            
            webhook_config = self.config_manager.get('alerts.channels.webhook', {})
            url = webhook_config.get('url', '')
            
            if not url:
                return
            
            headers = webhook_config.get('headers', {})
            
            response = requests.post(url, json=alert, headers=headers, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Webhook报警已发送: {alert['rule_name']}")
            
        except Exception as e:
            logger.error(f"发送Webhook报警失败: {e}")


class QualityMonitorSystem:
    """质量监控系统主类"""
    
    def __init__(self, project_path: str, config_file: str = None):
        self.project_path = project_path
        self.config_manager = ConfigManager(config_file or 'quality_monitoring_config.yaml')
        self.data_collector = QualityDataCollector(project_path, self.config_manager)
        self.alert_manager = AlertManager(self.config_manager)
        self.running = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """启动监控"""
        if self.running:
            logger.warning("监控已在运行中")
            return
        
        if not SCHEDULE_AVAILABLE:
            logger.error("缺少schedule模块，请安装: pip install schedule")
            return
        
        self.running = True
        
        # 配置定时任务
        interval = self.config_manager.get('monitoring.interval', 300)
        schedule.every(interval).seconds.do(self._collect_and_check)
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"质量监控已启动，监控间隔: {interval}秒")
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        
        if SCHEDULE_AVAILABLE and schedule:
            schedule.clear()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("质量监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                if SCHEDULE_AVAILABLE and schedule:
                    schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(5)
    
    def _collect_and_check(self):
        """收集数据并检查报警"""
        try:
            logger.info("执行定时质量检查...")
            
            # 收集质量指标
            metrics = self.data_collector.collect_current_metrics()
            if not metrics:
                logger.error("质量指标收集失败")
                return
            
            # 保存指标
            self.data_collector.save_metrics(metrics)
            
            # 检查报警
            alerts = self.alert_manager.check_alerts(metrics)
            
            # 发送报警
            if alerts:
                self.alert_manager.send_alerts(alerts)
                logger.info(f"触发了 {len(alerts)} 个报警")
            
            logger.info(f"质量检查完成: {metrics.overall_score:.2f}分")
            
        except Exception as e:
            logger.error(f"定时质量检查失败: {e}")
    
    def run_single_check(self) -> Optional[QualityMetrics]:
        """运行单次检查"""
        logger.info("执行单次质量检查...")
        
        # 收集质量指标
        metrics = self.data_collector.collect_current_metrics()
        if not metrics:
            logger.error("质量指标收集失败")
            return None
        
        # 保存指标
        self.data_collector.save_metrics(metrics)
        
        # 检查报警
        alerts = self.alert_manager.check_alerts(metrics)
        
        # 发送报警
        if alerts:
            self.alert_manager.send_alerts(alerts)
            logger.info(f"触发了 {len(alerts)} 个报警")
        
        logger.info(f"单次质量检查完成: {metrics.overall_score:.2f}分")
        return metrics
    
    def generate_report(self, output_path: str, format_type: str = 'html'):
        """生成质量报告"""
        try:
            logger.info(f"生成质量报告: {format_type}")
            
            scripts_dir = Path(__file__).parent
            
            if format_type == 'dashboard':
                # 生成仪表板
                subprocess.run([
                    sys.executable,
                    str(scripts_dir / 'quality_dashboard.py'),
                    '--static',
                    '--output', output_path
                ], check=True)
            
            elif format_type == 'trend':
                # 生成趋势报告
                subprocess.run([
                    sys.executable,
                    str(scripts_dir / 'quality_trend_analyzer.py'),
                    '--report',
                    '--output', output_path
                ], check=True)
            
            else:
                # 生成集成报告
                subprocess.run([
                    sys.executable,
                    str(scripts_dir / 'integrated_quality_system.py'),
                    '--analyze', self.project_path,
                    '--format', format_type,
                    '--output', output_path
                ], check=True)
            
            logger.info(f"质量报告已生成: {output_path}")
            
        except Exception as e:
            logger.error(f"生成质量报告失败: {e}")
    
    def start_dashboard_server(self):
        """启动仪表板服务器"""
        try:
            logger.info("启动仪表板服务器...")
            
            scripts_dir = Path(__file__).parent
            host = self.config_manager.get('dashboard.server.host', '127.0.0.1')
            port = self.config_manager.get('dashboard.server.port', 8080)
            
            subprocess.run([
                sys.executable,
                str(scripts_dir / 'quality_dashboard.py'),
                '--serve',
                '--host', host,
                '--port', str(port),
                self.project_path
            ])
            
        except Exception as e:
            logger.error(f"启动仪表板服务器失败: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'running': self.running,
            'project_path': self.project_path,
            'config_file': self.config_manager.config_file,
            'monitoring_interval': self.config_manager.get('monitoring.interval', 300),
            'alerts_enabled': self.config_manager.get('alerts.enabled', True),
            'alert_rules_count': len(self.alert_manager.alert_rules),
            'alert_history_count': len(self.alert_manager.alert_history)
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="VIVTransformer 质量监控系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --monitor .
  %(prog)s --check --project .
  %(prog)s --dashboard --project .
  %(prog)s --report --output report.html
        """
    )
    
    parser.add_argument(
        '--project',
        default='.',
        help='项目路径'
    )
    
    parser.add_argument(
        '--config',
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='启动持续监控'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='执行单次质量检查'
    )
    
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='启动仪表板服务器'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='生成质量报告'
    )
    
    parser.add_argument(
        '--output',
        default='quality_report.html',
        help='报告输出路径'
    )
    
    parser.add_argument(
        '--format',
        choices=['html', 'json', 'markdown', 'dashboard', 'trend'],
        default='html',
        help='报告格式'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='显示系统状态'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if not os.path.exists(args.project):
            print(f"❌ 项目路径不存在: {args.project}")
            sys.exit(1)
        
        # 初始化监控系统
        monitor_system = QualityMonitorSystem(args.project, args.config)
        
        if args.status:
            status = monitor_system.get_system_status()
            print("\n" + "="*50)
            print("📊 质量监控系统状态")
            print("="*50)
            for key, value in status.items():
                print(f"{key}: {value}")
            print("="*50)
        
        elif args.check:
            print(f"🔍 执行质量检查: {args.project}")
            metrics = monitor_system.run_single_check()
            
            if metrics:
                print("\n" + "="*50)
                print("📈 质量检查结果")
                print("="*50)
                print(f"总体分数: {metrics.overall_score:.2f}/100")
                print(f"问题数量: {metrics.issues_count}")
                print(f"严重问题: {metrics.critical_issues}")
                print(f"安全分数: {metrics.security_score:.2f}")
                print(f"性能分数: {metrics.performance_score:.2f}")
                print(f"测试覆盖率: {metrics.coverage:.1f}%")
                print("="*50)
            else:
                print("❌ 质量检查失败")
                sys.exit(1)
        
        elif args.monitor:
            print(f"🔄 启动质量监控: {args.project}")
            monitor_system.start_monitoring()
            
            try:
                print("监控运行中... 按 Ctrl+C 停止")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n⏹️  停止监控...")
                monitor_system.stop_monitoring()
        
        elif args.dashboard:
            print(f"🚀 启动仪表板服务器: {args.project}")
            monitor_system.start_dashboard_server()
        
        elif args.report:
            print(f"📋 生成质量报告: {args.format}")
            monitor_system.generate_report(args.output, args.format)
            print(f"✅ 报告已生成: {args.output}")
        
        else:
            print("请指定操作: --monitor, --check, --dashboard, --report, 或 --status")
            parser.print_help()
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⏹️  操作被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"质量监控系统运行失败: {e}")
        print(f"❌ 质量监控系统运行失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()