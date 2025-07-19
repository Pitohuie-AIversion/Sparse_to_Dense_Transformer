#!/usr/bin/env python3
"""
代码质量仪表板 - 生成可视化的代码质量报告

这个脚本生成一个HTML仪表板，展示项目的代码质量指标，包括：
- 质量趋势图表
- 问题分布统计
- 覆盖率报告
- 复杂度分析
- 技术债务评估

作者: VIVTransformer Team
日期: 2024
"""

import argparse
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

from code_quality import CodeQualityChecker, QualityReport, Severity


class QualityDatabase:
    """质量数据库管理器"""
    
    def __init__(self, db_path: str = "quality_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建质量记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                commit_hash TEXT,
                branch TEXT,
                tool TEXT NOT NULL,
                critical_issues INTEGER DEFAULT 0,
                high_issues INTEGER DEFAULT 0,
                medium_issues INTEGER DEFAULT 0,
                low_issues INTEGER DEFAULT 0,
                total_issues INTEGER DEFAULT 0,
                test_coverage REAL DEFAULT 0.0,
                complexity_score REAL DEFAULT 0.0,
                technical_debt_minutes INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)
        
        # 创建问题详情表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS issue_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id INTEGER,
                tool TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_number INTEGER,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                rule_id TEXT,
                FOREIGN KEY (record_id) REFERENCES quality_records (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_quality_record(self, 
                           reports: Dict[str, QualityReport],
                           commit_hash: Optional[str] = None,
                           branch: Optional[str] = None) -> int:
        """保存质量记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # 计算总体统计
        total_stats = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'total': 0
        }
        
        for report in reports.values():
            for issue in report.issues:
                if issue.severity == Severity.CRITICAL:
                    total_stats['critical'] += 1
                elif issue.severity == Severity.HIGH:
                    total_stats['high'] += 1
                elif issue.severity == Severity.MEDIUM:
                    total_stats['medium'] += 1
                elif issue.severity == Severity.LOW:
                    total_stats['low'] += 1
                total_stats['total'] += 1
        
        # 插入主记录
        cursor.execute("""
            INSERT INTO quality_records 
            (timestamp, commit_hash, branch, tool, critical_issues, high_issues, 
             medium_issues, low_issues, total_issues, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, commit_hash, branch, 'aggregate',
            total_stats['critical'], total_stats['high'],
            total_stats['medium'], total_stats['low'],
            total_stats['total'], json.dumps({})
        ))
        
        record_id = cursor.lastrowid
        
        # 插入问题详情
        for tool_name, report in reports.items():
            for issue in report.issues:
                cursor.execute("""
                    INSERT INTO issue_details 
                    (record_id, tool, file_path, line_number, severity, message, rule_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    record_id, tool_name, issue.file, issue.line,
                    issue.severity.value, issue.message, issue.rule_id
                ))
        
        conn.commit()
        conn.close()
        
        return record_id
    
    def get_quality_history(self, days: int = 30) -> pd.DataFrame:
        """获取质量历史数据"""
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("需要安装pandas来获取历史数据")
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT timestamp, critical_issues, high_issues, medium_issues, 
                   low_issues, total_issues, test_coverage, complexity_score
            FROM quality_records 
            WHERE tool = 'aggregate' 
              AND datetime(timestamp) >= datetime('now', '-{} days')
            ORDER BY timestamp
        """.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df


class QualityDashboard:
    """质量仪表板生成器"""
    
    def __init__(self, 
                 target_dirs: List[str],
                 python_executable: str = "python",
                 db_path: str = "quality_history.db"):
        self.target_dirs = target_dirs
        self.python_executable = python_executable
        self.db = QualityDatabase(db_path)
        self.checker = CodeQualityChecker(target_dirs, python_executable)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def collect_current_metrics(self) -> Dict[str, QualityReport]:
        """收集当前质量指标"""
        self.logger.info("收集当前质量指标...")
        
        reports = self.checker.run_all_checks()
        
        # 保存到数据库
        self.db.save_quality_record(reports)
        
        return reports
    
    def generate_trend_charts(self, output_dir: Path) -> List[str]:
        """生成趋势图表"""
        if not VISUALIZATION_AVAILABLE:
            self.logger.warning("跳过图表生成 - 缺少可视化依赖")
            return []
        
        self.logger.info("生成趋势图表...")
        
        # 获取历史数据
        df = self.db.get_quality_history(30)
        
        if df.empty:
            self.logger.warning("没有历史数据，跳过趋势图表")
            return []
        
        charts = []
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. 问题趋势图
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['timestamp'], df['critical_issues'], label='严重', marker='o', linewidth=2)
        ax.plot(df['timestamp'], df['high_issues'], label='高级', marker='s', linewidth=2)
        ax.plot(df['timestamp'], df['medium_issues'], label='中级', marker='^', linewidth=2)
        ax.plot(df['timestamp'], df['low_issues'], label='低级', marker='d', linewidth=2)
        
        ax.set_title('代码质量问题趋势 (30天)', fontsize=16, fontweight='bold')
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('问题数量', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = output_dir / 'issues_trend.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(str(chart_path))
        
        # 2. 总问题数趋势
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['timestamp'], df['total_issues'], label='总问题数', 
                marker='o', linewidth=3, color='red')
        
        ax.set_title('总问题数趋势 (30天)', fontsize=16, fontweight='bold')
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('问题数量', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = output_dir / 'total_issues_trend.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(str(chart_path))
        
        # 3. 测试覆盖率趋势
        if 'test_coverage' in df.columns and df['test_coverage'].notna().any():
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['timestamp'], df['test_coverage'], label='测试覆盖率', 
                    marker='o', linewidth=3, color='green')
            
            ax.set_title('测试覆盖率趋势 (30天)', fontsize=16, fontweight='bold')
            ax.set_xlabel('日期', fontsize=12)
            ax.set_ylabel('覆盖率 (%)', fontsize=12)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            chart_path = output_dir / 'coverage_trend.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts.append(str(chart_path))
        
        return charts
    
    def generate_distribution_charts(self, 
                                   reports: Dict[str, QualityReport], 
                                   output_dir: Path) -> List[str]:
        """生成分布图表"""
        if not VISUALIZATION_AVAILABLE:
            return []
        
        self.logger.info("生成分布图表...")
        
        charts = []
        
        # 1. 问题严重性分布饼图
        severity_counts = {'严重': 0, '高级': 0, '中级': 0, '低级': 0}
        
        for report in reports.values():
            for issue in report.issues:
                if issue.severity == Severity.CRITICAL:
                    severity_counts['严重'] += 1
                elif issue.severity == Severity.HIGH:
                    severity_counts['高级'] += 1
                elif issue.severity == Severity.MEDIUM:
                    severity_counts['中级'] += 1
                elif issue.severity == Severity.LOW:
                    severity_counts['低级'] += 1
        
        if sum(severity_counts.values()) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = ['#ff4444', '#ff8800', '#ffcc00', '#4488ff']
            wedges, texts, autotexts = ax.pie(
                severity_counts.values(), 
                labels=severity_counts.keys(),
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            
            ax.set_title('问题严重性分布', fontsize=16, fontweight='bold')
            
            chart_path = output_dir / 'severity_distribution.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts.append(str(chart_path))
        
        # 2. 工具问题数对比
        tool_counts = {}
        for tool_name, report in reports.items():
            tool_counts[tool_name] = len(report.issues)
        
        if tool_counts:
            fig, ax = plt.subplots(figsize=(12, 6))
            tools = list(tool_counts.keys())
            counts = list(tool_counts.values())
            
            bars = ax.bar(tools, counts, color=sns.color_palette("husl", len(tools)))
            ax.set_title('各工具发现的问题数量', fontsize=16, fontweight='bold')
            ax.set_xlabel('工具', fontsize=12)
            ax.set_ylabel('问题数量', fontsize=12)
            
            # 在柱子上显示数值
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            chart_path = output_dir / 'tools_comparison.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts.append(str(chart_path))
        
        return charts
    
    def generate_html_dashboard(self, 
                              reports: Dict[str, QualityReport],
                              charts: List[str],
                              output_file: str) -> str:
        """生成HTML仪表板"""
        self.logger.info("生成HTML仪表板...")
        
        # 计算总体统计
        total_issues = sum(len(report.issues) for report in reports.values())
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for report in reports.values():
            for issue in report.issues:
                if issue.severity == Severity.CRITICAL:
                    severity_counts['critical'] += 1
                elif issue.severity == Severity.HIGH:
                    severity_counts['high'] += 1
                elif issue.severity == Severity.MEDIUM:
                    severity_counts['medium'] += 1
                elif issue.severity == Severity.LOW:
                    severity_counts['low'] += 1
        
        # 生成HTML内容
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VIVTransformer 代码质量仪表板</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
        }}
        .stat-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid;
        }}
        .stat-card.critical {{ border-left-color: #ff4444; }}
        .stat-card.high {{ border-left-color: #ff8800; }}
        .stat-card.medium {{ border-left-color: #ffcc00; }}
        .stat-card.low {{ border-left-color: #4488ff; }}
        .stat-card.total {{ border-left-color: #8e44ad; }}
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 0;
        }}
        .stat-label {{
            color: #666;
            margin: 5px 0 0 0;
            font-size: 0.9em;
        }}
        .section {{
            padding: 30px;
            border-top: 1px solid #eee;
        }}
        .section h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .chart-container {{
            text-align: center;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .tools-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .tool-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .tool-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .tool-name {{
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }}
        .tool-status {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .tool-status.success {{
            background-color: #d4edda;
            color: #155724;
        }}
        .tool-status.warning {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .tool-status.error {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .issue-list {{
            max-height: 200px;
            overflow-y: auto;
        }}
        .issue-item {{
            padding: 8px 0;
            border-bottom: 1px solid #eee;
            font-size: 0.9em;
        }}
        .issue-item:last-child {{
            border-bottom: none;
        }}
        .issue-severity {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .issue-severity.critical {{ background-color: #ff4444; }}
        .issue-severity.high {{ background-color: #ff8800; }}
        .issue-severity.medium {{ background-color: #ffcc00; }}
        .issue-severity.low {{ background-color: #4488ff; }}
        .footer {{
            background-color: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 代码质量仪表板</h1>
            <p>VIVTransformer 项目质量报告 - {datetime.now().strftime('%Y年%m月%d日 %H:%M')}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card critical">
                <div class="stat-number">{severity_counts['critical']}</div>
                <div class="stat-label">严重问题</div>
            </div>
            <div class="stat-card high">
                <div class="stat-number">{severity_counts['high']}</div>
                <div class="stat-label">高级问题</div>
            </div>
            <div class="stat-card medium">
                <div class="stat-number">{severity_counts['medium']}</div>
                <div class="stat-label">中级问题</div>
            </div>
            <div class="stat-card low">
                <div class="stat-number">{severity_counts['low']}</div>
                <div class="stat-label">低级问题</div>
            </div>
            <div class="stat-card total">
                <div class="stat-number">{total_issues}</div>
                <div class="stat-label">总问题数</div>
            </div>
        </div>
"""
        
        # 添加图表部分
        if charts:
            html_content += """
        <div class="section">
            <h2>📊 趋势分析</h2>
            <div class="charts-grid">
"""
            for chart in charts:
                chart_name = Path(chart).stem.replace('_', ' ').title()
                html_content += f"""
                <div class="chart-container">
                    <h3>{chart_name}</h3>
                    <img src="{Path(chart).name}" alt="{chart_name}">
                </div>
"""
            html_content += "            </div>\n        </div>\n"
        
        # 添加工具详情部分
        html_content += """
        <div class="section">
            <h2>🔧 工具检查详情</h2>
            <div class="tools-grid">
"""
        
        for tool_name, report in reports.items():
            issue_count = len(report.issues)
            if issue_count == 0:
                status_class = "success"
                status_text = "✅ 通过"
            elif issue_count <= 5:
                status_class = "warning"
                status_text = "⚠️ 警告"
            else:
                status_class = "error"
                status_text = "❌ 失败"
            
            html_content += f"""
                <div class="tool-card">
                    <div class="tool-header">
                        <div class="tool-name">{tool_name.title()}</div>
                        <div class="tool-status {status_class}">{status_text}</div>
                    </div>
                    <div class="issue-list">
"""
            
            if report.issues:
                for issue in report.issues[:10]:  # 只显示前10个问题
                    severity_class = issue.severity.value.lower()
                    html_content += f"""
                        <div class="issue-item">
                            <span class="issue-severity {severity_class}"></span>
                            <strong>{issue.file}:{issue.line}</strong> - {issue.message}
                        </div>
"""
                if len(report.issues) > 10:
                    html_content += f"                        <div class=\"issue-item\">... 还有 {len(report.issues) - 10} 个问题</div>\n"
            else:
                html_content += "                        <div class=\"issue-item\">🎉 没有发现问题！</div>\n"
            
            html_content += "                    </div>\n                </div>\n"
        
        html_content += """
            </div>
        </div>
        
        <div class="footer">
            <p>由 VIVTransformer 质量仪表板生成 | 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file
    
    def generate_dashboard(self, output_dir: str = "quality_dashboard") -> str:
        """生成完整的质量仪表板"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 收集当前指标
        reports = self.collect_current_metrics()
        
        # 生成图表
        charts = []
        if VISUALIZATION_AVAILABLE:
            trend_charts = self.generate_trend_charts(output_path)
            distribution_charts = self.generate_distribution_charts(reports, output_path)
            charts.extend(trend_charts)
            charts.extend(distribution_charts)
        
        # 生成HTML仪表板
        dashboard_file = output_path / "index.html"
        self.generate_html_dashboard(reports, charts, str(dashboard_file))
        
        self.logger.info(f"质量仪表板已生成: {dashboard_file}")
        return str(dashboard_file)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="生成代码质量仪表板",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--target-dirs',
        nargs='+',
        default=['modify_multi_attention'],
        help='要分析的目标目录'
    )
    
    parser.add_argument(
        '--output-dir',
        default='quality_dashboard',
        help='输出目录'
    )
    
    parser.add_argument(
        '--python',
        default='python',
        help='Python解释器路径'
    )
    
    parser.add_argument(
        '--db-path',
        default='quality_history.db',
        help='质量历史数据库路径'
    )
    
    args = parser.parse_args()
    
    if not VISUALIZATION_AVAILABLE:
        print("警告: 缺少可视化依赖 (matplotlib, seaborn, pandas)")
        print("运行: pip install matplotlib seaborn pandas")
        print("将生成简化版仪表板...\n")
    
    try:
        dashboard = QualityDashboard(
            target_dirs=args.target_dirs,
            python_executable=args.python,
            db_path=args.db_path
        )
        
        dashboard_file = dashboard.generate_dashboard(args.output_dir)
        
        print(f"\n✅ 质量仪表板生成成功!")
        print(f"📊 仪表板文件: {dashboard_file}")
        print(f"🌐 在浏览器中打开: file://{Path(dashboard_file).absolute()}")
        
    except Exception as e:
        print(f"生成仪表板时发生错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()