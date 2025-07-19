#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIVTransformer 质量监控系统使用示例

本脚本展示了如何使用质量监控系统的各种功能

作者: VIVTransformer Team
日期: 2024
"""

import os
import sys
import time
import json
from pathlib import Path

# 添加脚本目录到路径
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))


def example_basic_quality_check():
    """示例1: 基本质量检查"""
    print("\n" + "="*60)
    print("📊 示例1: 基本质量检查")
    print("="*60)
    
    try:
        from integrated_quality_system import IntegratedQualitySystem
        
        # 初始化质量分析系统
        project_path = "."
        quality_system = IntegratedQualitySystem(project_path)
        
        print(f"🔍 分析项目: {project_path}")
        
        # 执行质量分析
        results = quality_system.run_analysis()
        
        if results:
            print(f"✅ 分析完成")
            print(f"📈 总体质量分数: {results.get('overall_score', 0):.2f}/100")
            
            # 显示各维度分数
            dimensions = results.get('dimensions', [])
            for dim in dimensions:
                print(f"   - {dim['name']}: {dim['score']:.2f}/100")
            
            # 显示问题摘要
            summary = results.get('summary', {})
            print(f"🐛 问题统计:")
            print(f"   - 总问题数: {summary.get('total_issues', 0)}")
            print(f"   - 严重问题: {summary.get('critical_issues', 0)}")
            print(f"   - 高级问题: {summary.get('high_issues', 0)}")
            
        else:
            print("❌ 分析失败")
            
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")


def example_monitoring_setup():
    """示例2: 监控系统设置"""
    print("\n" + "="*60)
    print("🔄 示例2: 监控系统设置")
    print("="*60)
    
    try:
        from quality_monitor_system import QualityMonitorSystem
        
        # 初始化监控系统
        project_path = "."
        monitor_system = QualityMonitorSystem(project_path)
        
        print(f"🔧 设置监控系统: {project_path}")
        
        # 获取系统状态
        status = monitor_system.get_system_status()
        
        print("📊 系统状态:")
        for key, value in status.items():
            print(f"   - {key}: {value}")
        
        # 执行单次检查
        print("\n🔍 执行单次质量检查...")
        metrics = monitor_system.run_single_check()
        
        if metrics:
            print(f"✅ 检查完成")
            print(f"📈 质量指标:")
            print(f"   - 总体分数: {metrics.overall_score:.2f}")
            print(f"   - 问题数量: {metrics.issues_count}")
            print(f"   - 严重问题: {metrics.critical_issues}")
            print(f"   - 安全分数: {metrics.security_score:.2f}")
            print(f"   - 性能分数: {metrics.performance_score:.2f}")
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")


def example_dashboard_generation():
    """示例3: 仪表板生成"""
    print("\n" + "="*60)
    print("📊 示例3: 仪表板生成")
    print("="*60)
    
    try:
        from quality_dashboard import QualityDashboard
        
        # 初始化仪表板
        project_path = "."
        output_dir = "quality_output"
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        dashboard = QualityDashboard(project_path, output_dir)
        
        print(f"🎨 生成质量仪表板...")
        
        # 生成仪表板
        dashboard_file = dashboard.generate_dashboard()
        
        if dashboard_file and os.path.exists(dashboard_file):
            print(f"✅ 仪表板已生成: {dashboard_file}")
            print(f"🌐 可以在浏览器中打开查看")
        else:
            print("❌ 仪表板生成失败")
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")


def example_trend_analysis():
    """示例4: 趋势分析"""
    print("\n" + "="*60)
    print("📈 示例4: 趋势分析")
    print("="*60)
    
    try:
        from quality_trend_analyzer import QualityTrendAnalyzer
        
        # 初始化趋势分析器
        analyzer = QualityTrendAnalyzer()
        
        print(f"📊 收集质量快照...")
        
        # 收集当前快照
        snapshot = analyzer.collect_current_snapshot(".")
        
        if snapshot:
            print(f"✅ 快照收集完成")
            print(f"📈 快照信息:")
            print(f"   - 时间戳: {snapshot.timestamp}")
            print(f"   - 分支: {snapshot.branch}")
            print(f"   - 总体分数: {snapshot.overall_score:.2f}")
            print(f"   - 问题数量: {snapshot.issues_count}")
        
        print(f"\n📊 分析质量趋势...")
        
        # 分析趋势
        trend_report = analyzer.analyze_trends()
        
        if trend_report:
            print(f"✅ 趋势分析完成")
            print(f"📈 趋势摘要: {trend_report.summary}")
            
            if trend_report.recommendations:
                print(f"💡 改进建议:")
                for rec in trend_report.recommendations[:3]:
                    print(f"   - {rec}")
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")


def example_quality_gate():
    """示例5: 质量门禁"""
    print("\n" + "="*60)
    print("🚪 示例5: 质量门禁")
    print("="*60)
    
    try:
        from quality_gate import QualityGate, QualityThresholds
        
        # 设置质量阈值
        thresholds = QualityThresholds(
            min_overall_score=70.0,
            max_critical_issues=5,
            min_test_coverage=60.0,
            max_complexity=10,
            min_security_score=75.0
        )
        
        # 初始化质量门禁
        quality_gate = QualityGate(".", thresholds)
        
        print(f"🔍 执行质量门禁检查...")
        
        # 运行质量门禁
        passed, results = quality_gate.run_quality_gate()
        
        if passed:
            print(f"✅ 质量门禁通过")
        else:
            print(f"❌ 质量门禁失败")
        
        print(f"📊 检查结果:")
        for check, result in results.items():
            status = "✅" if result['passed'] else "❌"
            print(f"   {status} {check}: {result.get('message', '')}")
        
        # 显示失败原因
        failed_checks = [check for check, result in results.items() if not result['passed']]
        if failed_checks:
            print(f"\n🚨 失败的检查项:")
            for check in failed_checks:
                print(f"   - {check}")
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")


def example_security_analysis():
    """示例6: 安全分析"""
    print("\n" + "="*60)
    print("🔒 示例6: 安全分析")
    print("="*60)
    
    try:
        from security_analyzer import SecurityAnalyzer
        
        # 初始化安全分析器
        analyzer = SecurityAnalyzer(".")
        
        print(f"🔍 执行安全扫描...")
        
        # 运行安全分析
        results = analyzer.run_security_analysis()
        
        if results:
            print(f"✅ 安全分析完成")
            
            # 显示安全分数
            security_score = results.get('security_score', 0)
            print(f"🔒 安全分数: {security_score:.2f}/100")
            
            # 显示漏洞统计
            vulnerabilities = results.get('vulnerabilities', [])
            if vulnerabilities:
                print(f"🚨 发现 {len(vulnerabilities)} 个安全问题:")
                
                # 按严重性分组
                severity_counts = {}
                for vuln in vulnerabilities:
                    severity = vuln.get('severity', 'unknown')
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                for severity, count in severity_counts.items():
                    print(f"   - {severity}: {count}")
            else:
                print(f"✅ 未发现安全问题")
            
            # 显示改进建议
            recommendations = results.get('recommendations', [])
            if recommendations:
                print(f"💡 安全改进建议:")
                for rec in recommendations[:3]:
                    print(f"   - {rec}")
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")


def example_performance_analysis():
    """示例7: 性能分析"""
    print("\n" + "="*60)
    print("⚡ 示例7: 性能分析")
    print("="*60)
    
    try:
        from performance_monitor import PerformanceMonitor
        
        # 初始化性能监控器
        monitor = PerformanceMonitor(".")
        
        print(f"📊 执行性能分析...")
        
        # 运行性能分析
        results = monitor.run_performance_analysis()
        
        if results:
            print(f"✅ 性能分析完成")
            
            # 显示性能分数
            performance_score = results.get('performance_score', 0)
            print(f"⚡ 性能分数: {performance_score:.2f}/100")
            
            # 显示性能指标
            metrics = results.get('metrics', {})
            if metrics:
                print(f"📈 性能指标:")
                for metric, value in metrics.items():
                    print(f"   - {metric}: {value}")
            
            # 显示性能问题
            issues = results.get('issues', [])
            if issues:
                print(f"🚨 发现 {len(issues)} 个性能问题:")
                for issue in issues[:5]:  # 显示前5个
                    print(f"   - {issue.get('description', '')}")
            
            # 显示优化建议
            recommendations = results.get('recommendations', [])
            if recommendations:
                print(f"💡 性能优化建议:")
                for rec in recommendations[:3]:
                    print(f"   - {rec}")
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")


def example_team_collaboration():
    """示例8: 团队协作分析"""
    print("\n" + "="*60)
    print("👥 示例8: 团队协作分析")
    print("="*60)
    
    try:
        from team_collaboration_analyzer import TeamCollaborationAnalyzer
        
        # 初始化团队协作分析器
        analyzer = TeamCollaborationAnalyzer(".")
        
        print(f"👥 执行团队协作分析...")
        
        # 运行协作分析
        results = analyzer.run_collaboration_analysis()
        
        if results:
            print(f"✅ 协作分析完成")
            
            # 显示协作分数
            collaboration_score = results.get('collaboration_score', 0)
            print(f"👥 协作分数: {collaboration_score:.2f}/100")
            
            # 显示团队指标
            metrics = results.get('metrics', {})
            if metrics:
                print(f"📊 团队指标:")
                for metric, value in metrics.items():
                    print(f"   - {metric}: {value}")
            
            # 显示知识分布
            knowledge_distribution = results.get('knowledge_distribution', {})
            if knowledge_distribution:
                print(f"🧠 知识分布:")
                for area, experts in knowledge_distribution.items():
                    if experts:
                        print(f"   - {area}: {', '.join(experts[:3])}")
            
            # 显示改进建议
            recommendations = results.get('recommendations', [])
            if recommendations:
                print(f"💡 协作改进建议:")
                for rec in recommendations[:3]:
                    print(f"   - {rec}")
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")


def run_all_examples():
    """运行所有示例"""
    print("🚀 VIVTransformer 质量监控系统使用示例")
    print("="*80)
    
    examples = [
        example_basic_quality_check,
        example_monitoring_setup,
        example_dashboard_generation,
        example_trend_analysis,
        example_quality_gate,
        example_security_analysis,
        example_performance_analysis,
        example_team_collaboration
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            example()
            time.sleep(1)  # 短暂暂停
        except KeyboardInterrupt:
            print("\n⏹️  示例运行被用户中断")
            break
        except Exception as e:
            print(f"❌ 示例 {i} 运行失败: {e}")
            continue
    
    print("\n" + "="*80)
    print("✅ 所有示例运行完成")
    print("="*80)


def interactive_demo():
    """交互式演示"""
    examples_map = {
        '1': ('基本质量检查', example_basic_quality_check),
        '2': ('监控系统设置', example_monitoring_setup),
        '3': ('仪表板生成', example_dashboard_generation),
        '4': ('趋势分析', example_trend_analysis),
        '5': ('质量门禁', example_quality_gate),
        '6': ('安全分析', example_security_analysis),
        '7': ('性能分析', example_performance_analysis),
        '8': ('团队协作分析', example_team_collaboration),
        '9': ('运行所有示例', run_all_examples)
    }
    
    while True:
        print("\n" + "="*60)
        print("🎯 VIVTransformer 质量监控系统演示")
        print("="*60)
        print("请选择要运行的示例:")
        
        for key, (name, _) in examples_map.items():
            print(f"  {key}. {name}")
        
        print("  0. 退出")
        print("-"*60)
        
        choice = input("请输入选择 (0-9): ").strip()
        
        if choice == '0':
            print("👋 再见！")
            break
        elif choice in examples_map:
            name, func = examples_map[choice]
            print(f"\n🚀 运行示例: {name}")
            try:
                func()
            except KeyboardInterrupt:
                print("\n⏹️  示例被用户中断")
            except Exception as e:
                print(f"❌ 示例运行失败: {e}")
            
            input("\n按回车键继续...")
        else:
            print("❌ 无效选择，请重新输入")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VIVTransformer 质量监控系统使用示例",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--demo',
        choices=['1', '2', '3', '4', '5', '6', '7', '8', 'all'],
        help='运行特定示例 (1-8) 或所有示例 (all)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='交互式演示模式'
    )
    
    args = parser.parse_args()
    
    try:
        if args.demo:
            if args.demo == 'all':
                run_all_examples()
            else:
                demo_map = {
                    '1': example_basic_quality_check,
                    '2': example_monitoring_setup,
                    '3': example_dashboard_generation,
                    '4': example_trend_analysis,
                    '5': example_quality_gate,
                    '6': example_security_analysis,
                    '7': example_performance_analysis,
                    '8': example_team_collaboration
                }
                
                if args.demo in demo_map:
                    demo_map[args.demo]()
                else:
                    print(f"❌ 无效的示例编号: {args.demo}")
        
        elif args.interactive:
            interactive_demo()
        
        else:
            # 默认交互式模式
            interactive_demo()
    
    except KeyboardInterrupt:
        print("\n⏹️  程序被用户中断")
    except Exception as e:
        print(f"❌ 程序运行失败: {e}")


if __name__ == '__main__':
    main()