#!/usr/bin/env python3
"""测试运行脚本

提供便捷的测试运行接口，支持不同类型的测试和配置选项
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import time


def run_command(cmd, cwd=None):
    """运行命令并返回结果"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout, e.stderr


def check_dependencies():
    """检查测试依赖"""
    print("Checking test dependencies...")
    
    required_packages = ['pytest', 'torch', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements-dev.txt")
        return False
    
    return True


def run_unit_tests(args):
    """运行单元测试"""
    print("\n=== Running Unit Tests ===")
    
    cmd = ['python', '-m', 'pytest', 'tests/unit/']
    
    if args.verbose:
        cmd.append('-v')
    if args.coverage:
        cmd.extend(['--cov=modify_multi_attention', '--cov-report=html', '--cov-report=term-missing'])
    if args.parallel:
        cmd.extend(['-n', 'auto'])
    if not args.slow:
        cmd.extend(['-m', 'not slow'])
    
    return run_command(cmd)


def run_integration_tests(args):
    """运行集成测试"""
    print("\n=== Running Integration Tests ===")
    
    cmd = ['python', '-m', 'pytest', 'tests/integration/']
    
    if args.verbose:
        cmd.append('-v')
    if args.coverage:
        cmd.extend(['--cov=modify_multi_attention', '--cov-report=html', '--cov-report=term-missing'])
    if not args.slow:
        cmd.extend(['-m', 'not slow'])
    
    return run_command(cmd)


def run_performance_tests(args):
    """运行性能测试"""
    print("\n=== Running Performance Tests ===")
    
    cmd = ['python', '-m', 'pytest', '-m', 'performance']
    
    if args.verbose:
        cmd.append('-v')
    
    return run_command(cmd)


def run_smoke_tests(args):
    """运行冒烟测试"""
    print("\n=== Running Smoke Tests ===")
    
    cmd = ['python', '-m', 'pytest', '-m', 'smoke']
    
    if args.verbose:
        cmd.append('-v')
    
    return run_command(cmd)


def run_all_tests(args):
    """运行所有测试"""
    print("\n=== Running All Tests ===")
    
    cmd = ['python', '-m', 'pytest', 'tests/']
    
    if args.verbose:
        cmd.append('-v')
    if args.coverage:
        cmd.extend(['--cov=modify_multi_attention', '--cov-report=html', '--cov-report=term-missing'])
    if args.parallel:
        cmd.extend(['-n', 'auto'])
    if not args.slow:
        cmd.extend(['-m', 'not slow'])
    
    return run_command(cmd)


def run_specific_test(args):
    """运行特定测试"""
    print(f"\n=== Running Specific Test: {args.test_path} ===")
    
    cmd = ['python', '-m', 'pytest', args.test_path]
    
    if args.verbose:
        cmd.append('-v')
    if args.coverage:
        cmd.extend(['--cov=modify_multi_attention', '--cov-report=html', '--cov-report=term-missing'])
    
    return run_command(cmd)


def run_failed_tests(args):
    """重新运行失败的测试"""
    print("\n=== Re-running Failed Tests ===")
    
    cmd = ['python', '-m', 'pytest', '--lf']
    
    if args.verbose:
        cmd.append('-v')
    
    return run_command(cmd)


def generate_test_report(args):
    """生成测试报告"""
    print("\n=== Generating Test Report ===")
    
    cmd = [
        'python', '-m', 'pytest', 'tests/',
        '--html=test_report.html',
        '--self-contained-html',
        '--cov=modify_multi_attention',
        '--cov-report=html',
        '--cov-report=xml'
    ]
    
    if not args.slow:
        cmd.extend(['-m', 'not slow'])
    
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode == 0:
        print("\nTest report generated:")
        print("- HTML report: test_report.html")
        print("- Coverage report: htmlcov/index.html")
        print("- Coverage XML: coverage.xml")
    
    return returncode, stdout, stderr


def setup_test_environment():
    """设置测试环境"""
    print("Setting up test environment...")
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = str(Path.cwd())
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一个GPU（如果可用）
    
    # 创建必要的目录
    directories = ['logs', 'results', 'htmlcov']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("Test environment setup complete.")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="VIVTransformer Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py --unit                    # Run unit tests
  python scripts/run_tests.py --integration             # Run integration tests
  python scripts/run_tests.py --all --coverage          # Run all tests with coverage
  python scripts/run_tests.py --performance             # Run performance tests
  python scripts/run_tests.py --specific tests/unit/test_attention.py
  python scripts/run_tests.py --failed                  # Re-run failed tests
  python scripts/run_tests.py --report                  # Generate test report
        """
    )
    
    # 测试类型选项
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument('--unit', action='store_true', help='Run unit tests')
    test_group.add_argument('--integration', action='store_true', help='Run integration tests')
    test_group.add_argument('--performance', action='store_true', help='Run performance tests')
    test_group.add_argument('--smoke', action='store_true', help='Run smoke tests')
    test_group.add_argument('--all', action='store_true', help='Run all tests')
    test_group.add_argument('--specific', metavar='TEST_PATH', help='Run specific test file or directory')
    test_group.add_argument('--failed', action='store_true', help='Re-run failed tests')
    test_group.add_argument('--report', action='store_true', help='Generate comprehensive test report')
    
    # 配置选项
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-c', '--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('-p', '--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--slow', action='store_true', help='Include slow tests')
    parser.add_argument('--no-setup', action='store_true', help='Skip environment setup')
    parser.add_argument('--check-deps', action='store_true', help='Only check dependencies')
    
    args = parser.parse_args()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    if args.check_deps:
        print("All dependencies are available.")
        sys.exit(0)
    
    # 设置测试环境
    if not args.no_setup:
        setup_test_environment()
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行测试
    returncode = 0
    stdout = ""
    stderr = ""
    
    try:
        if args.unit:
            returncode, stdout, stderr = run_unit_tests(args)
        elif args.integration:
            returncode, stdout, stderr = run_integration_tests(args)
        elif args.performance:
            returncode, stdout, stderr = run_performance_tests(args)
        elif args.smoke:
            returncode, stdout, stderr = run_smoke_tests(args)
        elif args.all:
            returncode, stdout, stderr = run_all_tests(args)
        elif args.specific:
            args.test_path = args.specific
            returncode, stdout, stderr = run_specific_test(args)
        elif args.failed:
            returncode, stdout, stderr = run_failed_tests(args)
        elif args.report:
            returncode, stdout, stderr = generate_test_report(args)
        
        # 输出结果
        if stdout:
            print("\nSTDOUT:")
            print(stdout)
        
        if stderr:
            print("\nSTDERR:")
            print(stderr)
        
        # 计算运行时间
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nTest execution completed in {duration:.2f} seconds")
        
        if returncode == 0:
            print("✓ All tests passed!")
        else:
            print(f"✗ Tests failed with return code {returncode}")
        
        sys.exit(returncode)
        
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during test execution: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()