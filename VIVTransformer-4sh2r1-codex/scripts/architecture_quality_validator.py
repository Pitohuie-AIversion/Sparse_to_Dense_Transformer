#!/usr/bin/env python3
"""
VIVTransformer 代码架构质量验证器

这个脚本提供代码架构质量验证功能，包括：
- 架构一致性检查
- 设计模式检测和验证
- 依赖关系分析
- 模块化质量评估
- 架构违规检测
- 代码组织结构分析

使用方法:
    python scripts/architecture_quality_validator.py --check-architecture src/
    python scripts/architecture_quality_validator.py --validate-patterns src/models/
    python scripts/architecture_quality_validator.py --analyze-dependencies
"""

import argparse
import ast
import json
import logging
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Union

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ArchitectureViolation:
    """架构违规数据类"""
    id: str
    type: str
    severity: str
    file_path: str
    line_number: int
    description: str
    rule_violated: str
    suggestion: str
    impact: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DependencyInfo:
    """依赖信息数据类"""
    source_module: str
    target_module: str
    dependency_type: str
    line_number: int
    import_statement: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DesignPattern:
    """设计模式数据类"""
    pattern_name: str
    pattern_type: str
    file_path: str
    class_name: str
    confidence: float
    description: str
    implementation_quality: str
    suggestions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModuleMetrics:
    """模块指标数据类"""
    module_path: str
    lines_of_code: int
    class_count: int
    function_count: int
    import_count: int
    complexity_score: float
    cohesion_score: float
    coupling_score: float
    maintainability_index: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DependencyAnalyzer:
    """依赖关系分析器"""
    
    def __init__(self):
        self.dependencies = []
        self.module_graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)
    
    def analyze_file_dependencies(self, file_path: str) -> List[DependencyInfo]:
        """分析文件的依赖关系"""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            source_module = self._get_module_name(file_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dep = DependencyInfo(
                            source_module=source_module,
                            target_module=alias.name,
                            dependency_type="import",
                            line_number=node.lineno,
                            import_statement=f"import {alias.name}"
                        )
                        dependencies.append(dep)
                        self.module_graph[source_module].add(alias.name)
                        self.reverse_graph[alias.name].add(source_module)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dep = DependencyInfo(
                            source_module=source_module,
                            target_module=node.module,
                            dependency_type="from_import",
                            line_number=node.lineno,
                            import_statement=f"from {node.module} import {', '.join(alias.name for alias in node.names)}"
                        )
                        dependencies.append(dep)
                        self.module_graph[source_module].add(node.module)
                        self.reverse_graph[node.module].add(source_module)
            
            return dependencies
            
        except Exception as e:
            logger.error(f"Failed to analyze dependencies in {file_path}: {e}")
            return []
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """检测循环依赖"""
        def dfs(node, path, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.module_graph[node]:
                if neighbor not in visited:
                    cycle = dfs(neighbor, path.copy(), visited, rec_stack)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # 找到循环
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
            
            rec_stack.remove(node)
            return None
        
        visited = set()
        cycles = []
        
        for node in self.module_graph:
            if node not in visited:
                cycle = dfs(node, [], visited, set())
                if cycle:
                    cycles.append(cycle)
        
        return cycles
    
    def calculate_coupling_metrics(self) -> Dict[str, float]:
        """计算耦合度指标"""
        metrics = {}
        
        for module in self.module_graph:
            # 出度耦合 (Ce - Efferent Coupling)
            efferent_coupling = len(self.module_graph[module])
            
            # 入度耦合 (Ca - Afferent Coupling)
            afferent_coupling = len(self.reverse_graph[module])
            
            # 不稳定性 (I = Ce / (Ca + Ce))
            total_coupling = afferent_coupling + efferent_coupling
            instability = efferent_coupling / max(1, total_coupling)
            
            metrics[module] = {
                'efferent_coupling': efferent_coupling,
                'afferent_coupling': afferent_coupling,
                'instability': instability
            }
        
        return metrics
    
    def _get_module_name(self, file_path: str) -> str:
        """从文件路径获取模块名"""
        path = Path(file_path)
        parts = path.with_suffix('').parts
        
        # 移除常见的根目录
        if parts and parts[0] in ['src', 'lib', 'app']:
            parts = parts[1:]
        
        return '.'.join(parts)


class DesignPatternDetector:
    """设计模式检测器"""
    
    def __init__(self):
        self.patterns = []
    
    def detect_patterns_in_file(self, file_path: str) -> List[DesignPattern]:
        """检测文件中的设计模式"""
        patterns = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # 检测各种设计模式
            patterns.extend(self._detect_singleton_pattern(tree, file_path))
            patterns.extend(self._detect_factory_pattern(tree, file_path))
            patterns.extend(self._detect_observer_pattern(tree, file_path))
            patterns.extend(self._detect_strategy_pattern(tree, file_path))
            patterns.extend(self._detect_decorator_pattern(tree, file_path))
            patterns.extend(self._detect_adapter_pattern(tree, file_path))
            patterns.extend(self._detect_builder_pattern(tree, file_path))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect patterns in {file_path}: {e}")
            return []
    
    def _detect_singleton_pattern(self, tree: ast.AST, file_path: str) -> List[DesignPattern]:
        """检测单例模式"""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 检查是否有 __new__ 方法实现单例
                has_new_method = False
                has_instance_check = False
                
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == '__new__':
                        has_new_method = True
                        
                        # 检查是否有实例检查逻辑
                        for stmt in ast.walk(method):
                            if isinstance(stmt, ast.If):
                                has_instance_check = True
                                break
                
                if has_new_method and has_instance_check:
                    patterns.append(DesignPattern(
                        pattern_name="Singleton",
                        pattern_type="Creational",
                        file_path=file_path,
                        class_name=node.name,
                        confidence=0.8,
                        description=f"Class '{node.name}' implements Singleton pattern using __new__ method",
                        implementation_quality="good",
                        suggestions=[
                            "Consider thread safety for multi-threaded applications",
                            "Ensure proper handling of subclassing"
                        ]
                    ))
        
        return patterns
    
    def _detect_factory_pattern(self, tree: ast.AST, file_path: str) -> List[DesignPattern]:
        """检测工厂模式"""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 检查是否有工厂方法
                factory_methods = []
                
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        # 检查方法名是否包含 create, make, build 等
                        if any(keyword in method.name.lower() for keyword in ['create', 'make', 'build', 'factory']):
                            # 检查是否返回类实例
                            for stmt in ast.walk(method):
                                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                                    factory_methods.append(method.name)
                                    break
                
                if factory_methods:
                    confidence = min(0.9, len(factory_methods) * 0.3)
                    patterns.append(DesignPattern(
                        pattern_name="Factory",
                        pattern_type="Creational",
                        file_path=file_path,
                        class_name=node.name,
                        confidence=confidence,
                        description=f"Class '{node.name}' implements Factory pattern with methods: {', '.join(factory_methods)}",
                        implementation_quality="good" if len(factory_methods) > 1 else "basic",
                        suggestions=[
                            "Consider using Abstract Factory for multiple product families",
                            "Ensure proper error handling in factory methods"
                        ]
                    ))
        
        return patterns
    
    def _detect_observer_pattern(self, tree: ast.AST, file_path: str) -> List[DesignPattern]:
        """检测观察者模式"""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 检查是否有观察者相关方法
                observer_methods = []
                has_observers_list = False
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_name = item.name.lower()
                        if any(keyword in method_name for keyword in ['attach', 'detach', 'notify', 'subscribe', 'unsubscribe', 'add_observer', 'remove_observer']):
                            observer_methods.append(item.name)
                    
                    elif isinstance(item, ast.Assign):
                        # 检查是否有观察者列表
                        for target in item.targets:
                            if isinstance(target, ast.Name) and 'observer' in target.id.lower():
                                has_observers_list = True
                
                if len(observer_methods) >= 2 and has_observers_list:
                    patterns.append(DesignPattern(
                        pattern_name="Observer",
                        pattern_type="Behavioral",
                        file_path=file_path,
                        class_name=node.name,
                        confidence=0.8,
                        description=f"Class '{node.name}' implements Observer pattern with methods: {', '.join(observer_methods)}",
                        implementation_quality="good",
                        suggestions=[
                            "Consider using weak references to avoid memory leaks",
                            "Implement proper exception handling in notification"
                        ]
                    ))
        
        return patterns
    
    def _detect_strategy_pattern(self, tree: ast.AST, file_path: str) -> List[DesignPattern]:
        """检测策略模式"""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 检查是否有策略相关的属性和方法
                has_strategy_attribute = False
                has_execute_method = False
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name in ['execute', 'run', 'apply', 'perform']:
                            has_execute_method = True
                        elif 'strategy' in item.name.lower():
                            has_strategy_attribute = True
                    
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and 'strategy' in target.id.lower():
                                has_strategy_attribute = True
                
                if has_strategy_attribute and has_execute_method:
                    patterns.append(DesignPattern(
                        pattern_name="Strategy",
                        pattern_type="Behavioral",
                        file_path=file_path,
                        class_name=node.name,
                        confidence=0.7,
                        description=f"Class '{node.name}' implements Strategy pattern",
                        implementation_quality="good",
                        suggestions=[
                            "Ensure all strategies implement the same interface",
                            "Consider using dependency injection for strategy selection"
                        ]
                    ))
        
        return patterns
    
    def _detect_decorator_pattern(self, tree: ast.AST, file_path: str) -> List[DesignPattern]:
        """检测装饰器模式"""
        patterns = []
        
        # 检查函数装饰器
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.decorator_list:
                if len(node.decorator_list) > 0:
                    patterns.append(DesignPattern(
                        pattern_name="Decorator",
                        pattern_type="Structural",
                        file_path=file_path,
                        class_name=node.name,
                        confidence=0.9,
                        description=f"Function '{node.name}' uses decorator pattern",
                        implementation_quality="good",
                        suggestions=[
                            "Ensure decorators preserve function metadata",
                            "Consider using functools.wraps for better debugging"
                        ]
                    ))
        
        return patterns
    
    def _detect_adapter_pattern(self, tree: ast.AST, file_path: str) -> List[DesignPattern]:
        """检测适配器模式"""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 检查是否有适配器相关的特征
                has_adaptee = False
                has_adapter_methods = False
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        # 检查构造函数是否接受被适配对象
                        if len(item.args.args) > 1:  # 除了self
                            has_adaptee = True
                    
                    elif isinstance(item, ast.FunctionDef):
                        # 检查是否有方法调用其他对象的方法
                        for stmt in ast.walk(item):
                            if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
                                has_adapter_methods = True
                
                if has_adaptee and has_adapter_methods and 'adapter' in node.name.lower():
                    patterns.append(DesignPattern(
                        pattern_name="Adapter",
                        pattern_type="Structural",
                        file_path=file_path,
                        class_name=node.name,
                        confidence=0.7,
                        description=f"Class '{node.name}' implements Adapter pattern",
                        implementation_quality="good",
                        suggestions=[
                            "Ensure the adapter implements the target interface completely",
                            "Consider using composition over inheritance"
                        ]
                    ))
        
        return patterns
    
    def _detect_builder_pattern(self, tree: ast.AST, file_path: str) -> List[DesignPattern]:
        """检测建造者模式"""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 检查是否有建造者模式的特征
                builder_methods = []
                has_build_method = False
                
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        if method.name == 'build':
                            has_build_method = True
                        elif method.name.startswith('set_') or method.name.startswith('with_'):
                            # 检查是否返回self
                            for stmt in ast.walk(method):
                                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name) and stmt.value.id == 'self':
                                    builder_methods.append(method.name)
                                    break
                
                if has_build_method and len(builder_methods) >= 2:
                    patterns.append(DesignPattern(
                        pattern_name="Builder",
                        pattern_type="Creational",
                        file_path=file_path,
                        class_name=node.name,
                        confidence=0.8,
                        description=f"Class '{node.name}' implements Builder pattern with methods: {', '.join(builder_methods)}",
                        implementation_quality="good",
                        suggestions=[
                            "Consider implementing a Director class for complex building logic",
                            "Ensure the builder can create different representations"
                        ]
                    ))
        
        return patterns


class ArchitectureRuleEngine:
    """架构规则引擎"""
    
    def __init__(self):
        self.rules = self._load_default_rules()
        self.violations = []
    
    def _load_default_rules(self) -> Dict[str, Any]:
        """加载默认架构规则"""
        return {
            'layered_architecture': {
                'description': 'Enforce layered architecture constraints',
                'rules': {
                    'no_skip_layer': True,
                    'no_reverse_dependency': True,
                    'layer_order': ['presentation', 'business', 'data']
                }
            },
            'dependency_rules': {
                'description': 'Dependency management rules',
                'rules': {
                    'no_circular_dependencies': True,
                    'max_coupling_threshold': 10,
                    'max_dependency_depth': 5
                }
            },
            'naming_conventions': {
                'description': 'Naming convention rules',
                'rules': {
                    'class_naming': r'^[A-Z][a-zA-Z0-9]*$',
                    'function_naming': r'^[a-z_][a-z0-9_]*$',
                    'constant_naming': r'^[A-Z_][A-Z0-9_]*$',
                    'module_naming': r'^[a-z_][a-z0-9_]*$'
                }
            },
            'code_organization': {
                'description': 'Code organization rules',
                'rules': {
                    'max_file_length': 500,
                    'max_class_methods': 20,
                    'max_function_parameters': 5,
                    'max_nesting_depth': 4
                }
            }
        }
    
    def validate_architecture(self, file_path: str, dependencies: List[DependencyInfo]) -> List[ArchitectureViolation]:
        """验证架构规则"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # 验证各种规则
            violations.extend(self._validate_naming_conventions(tree, file_path))
            violations.extend(self._validate_code_organization(tree, file_path, content))
            violations.extend(self._validate_dependency_rules(dependencies, file_path))
            
            return violations
            
        except Exception as e:
            logger.error(f"Failed to validate architecture in {file_path}: {e}")
            return []
    
    def _validate_naming_conventions(self, tree: ast.AST, file_path: str) -> List[ArchitectureViolation]:
        """验证命名约定"""
        violations = []
        rules = self.rules['naming_conventions']['rules']
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not re.match(rules['class_naming'], node.name):
                    violations.append(ArchitectureViolation(
                        id=f"naming_class_{node.name}_{node.lineno}",
                        type="naming_convention",
                        severity="medium",
                        file_path=file_path,
                        line_number=node.lineno,
                        description=f"Class name '{node.name}' doesn't follow naming convention",
                        rule_violated="class_naming",
                        suggestion="Use PascalCase for class names (e.g., MyClass)",
                        impact="Reduces code readability and consistency"
                    ))
            
            elif isinstance(node, ast.FunctionDef):
                if not re.match(rules['function_naming'], node.name) and not node.name.startswith('__'):
                    violations.append(ArchitectureViolation(
                        id=f"naming_function_{node.name}_{node.lineno}",
                        type="naming_convention",
                        severity="low",
                        file_path=file_path,
                        line_number=node.lineno,
                        description=f"Function name '{node.name}' doesn't follow naming convention",
                        rule_violated="function_naming",
                        suggestion="Use snake_case for function names (e.g., my_function)",
                        impact="Reduces code readability and consistency"
                    ))
            
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # 检查常量命名（全大写变量）
                        if target.id.isupper() and not re.match(rules['constant_naming'], target.id):
                            violations.append(ArchitectureViolation(
                                id=f"naming_constant_{target.id}_{node.lineno}",
                                type="naming_convention",
                                severity="low",
                                file_path=file_path,
                                line_number=node.lineno,
                                description=f"Constant name '{target.id}' doesn't follow naming convention",
                                rule_violated="constant_naming",
                                suggestion="Use UPPER_SNAKE_CASE for constants (e.g., MY_CONSTANT)",
                                impact="Reduces code readability and consistency"
                            ))
        
        return violations
    
    def _validate_code_organization(self, tree: ast.AST, file_path: str, content: str) -> List[ArchitectureViolation]:
        """验证代码组织规则"""
        violations = []
        rules = self.rules['code_organization']['rules']
        
        # 检查文件长度
        lines = content.split('\n')
        if len(lines) > rules['max_file_length']:
            violations.append(ArchitectureViolation(
                id=f"file_length_{len(lines)}",
                type="code_organization",
                severity="medium",
                file_path=file_path,
                line_number=1,
                description=f"File is too long ({len(lines)} lines, max: {rules['max_file_length']})",
                rule_violated="max_file_length",
                suggestion="Split large files into smaller, more focused modules",
                impact="Reduces maintainability and readability"
            ))
        
        # 检查类和函数的复杂度
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > rules['max_class_methods']:
                    violations.append(ArchitectureViolation(
                        id=f"class_methods_{node.name}_{len(methods)}",
                        type="code_organization",
                        severity="high",
                        file_path=file_path,
                        line_number=node.lineno,
                        description=f"Class '{node.name}' has too many methods ({len(methods)}, max: {rules['max_class_methods']})",
                        rule_violated="max_class_methods",
                        suggestion="Split large classes following Single Responsibility Principle",
                        impact="Violates SRP and reduces maintainability"
                    ))
            
            elif isinstance(node, ast.FunctionDef):
                # 检查参数数量
                param_count = len(node.args.args)
                if param_count > rules['max_function_parameters']:
                    violations.append(ArchitectureViolation(
                        id=f"function_params_{node.name}_{param_count}",
                        type="code_organization",
                        severity="medium",
                        file_path=file_path,
                        line_number=node.lineno,
                        description=f"Function '{node.name}' has too many parameters ({param_count}, max: {rules['max_function_parameters']})",
                        rule_violated="max_function_parameters",
                        suggestion="Use parameter objects or reduce function responsibilities",
                        impact="Reduces function usability and testability"
                    ))
                
                # 检查嵌套深度
                max_depth = self._calculate_nesting_depth(node)
                if max_depth > rules['max_nesting_depth']:
                    violations.append(ArchitectureViolation(
                        id=f"nesting_depth_{node.name}_{max_depth}",
                        type="code_organization",
                        severity="medium",
                        file_path=file_path,
                        line_number=node.lineno,
                        description=f"Function '{node.name}' has excessive nesting depth ({max_depth}, max: {rules['max_nesting_depth']})",
                        rule_violated="max_nesting_depth",
                        suggestion="Extract nested logic into separate functions or use guard clauses",
                        impact="Reduces readability and increases complexity"
                    ))
        
        return violations
    
    def _validate_dependency_rules(self, dependencies: List[DependencyInfo], file_path: str) -> List[ArchitectureViolation]:
        """验证依赖规则"""
        violations = []
        rules = self.rules['dependency_rules']['rules']
        
        # 检查依赖数量
        if len(dependencies) > rules['max_coupling_threshold']:
            violations.append(ArchitectureViolation(
                id=f"high_coupling_{len(dependencies)}",
                type="dependency_rule",
                severity="high",
                file_path=file_path,
                line_number=1,
                description=f"Module has high coupling ({len(dependencies)} dependencies, max: {rules['max_coupling_threshold']})",
                rule_violated="max_coupling_threshold",
                suggestion="Reduce dependencies by applying dependency inversion and interface segregation",
                impact="Increases coupling and reduces maintainability"
            ))
        
        return violations
    
    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """计算嵌套深度"""
        max_depth = 0
        
        def visit_node(n, depth):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.Try)):
                depth += 1
            
            for child in ast.iter_child_nodes(n):
                visit_node(child, depth)
        
        visit_node(node, 0)
        return max_depth


class ModuleQualityAnalyzer:
    """模块质量分析器"""
    
    def __init__(self):
        self.dependency_analyzer = DependencyAnalyzer()
    
    def analyze_module(self, file_path: str) -> ModuleMetrics:
        """分析模块质量指标"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            lines = content.split('\n')
            
            # 计算基本指标
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            
            # 计算复杂度
            complexity_score = self._calculate_complexity_score(tree)
            
            # 计算内聚度
            cohesion_score = self._calculate_cohesion_score(tree)
            
            # 计算耦合度
            dependencies = self.dependency_analyzer.analyze_file_dependencies(file_path)
            coupling_score = len(dependencies) / 10.0  # 简化的耦合度计算
            
            # 计算可维护性指数
            maintainability_index = self._calculate_maintainability_index(
                len(lines), complexity_score, len(functions)
            )
            
            return ModuleMetrics(
                module_path=file_path,
                lines_of_code=len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
                class_count=len(classes),
                function_count=len(functions),
                import_count=len(imports),
                complexity_score=complexity_score,
                cohesion_score=cohesion_score,
                coupling_score=min(1.0, coupling_score),
                maintainability_index=maintainability_index
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze module {file_path}: {e}")
            return ModuleMetrics(
                module_path=file_path,
                lines_of_code=0,
                class_count=0,
                function_count=0,
                import_count=0,
                complexity_score=0.0,
                cohesion_score=0.0,
                coupling_score=0.0,
                maintainability_index=0.0
            )
    
    def _calculate_complexity_score(self, tree: ast.AST) -> float:
        """计算复杂度分数"""
        total_complexity = 0
        function_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
                # 计算圈复杂度
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                        complexity += 1
                    elif isinstance(child, ast.ExceptHandler):
                        complexity += 1
                    elif isinstance(child, (ast.And, ast.Or)):
                        complexity += 1
                
                total_complexity += complexity
        
        return total_complexity / max(1, function_count)
    
    def _calculate_cohesion_score(self, tree: ast.AST) -> float:
        """计算内聚度分数"""
        # 简化的内聚度计算：基于类中方法对实例变量的使用
        total_cohesion = 0.0
        class_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_count += 1
                
                # 收集实例变量
                instance_vars = set()
                methods = []
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item)
                        
                        # 查找实例变量使用
                        for stmt in ast.walk(item):
                            if isinstance(stmt, ast.Attribute) and isinstance(stmt.value, ast.Name) and stmt.value.id == 'self':
                                instance_vars.add(stmt.attr)
                
                if methods and instance_vars:
                    # 计算方法对实例变量的使用率
                    usage_matrix = []
                    for method in methods:
                        used_vars = set()
                        for stmt in ast.walk(method):
                            if isinstance(stmt, ast.Attribute) and isinstance(stmt.value, ast.Name) and stmt.value.id == 'self':
                                if stmt.attr in instance_vars:
                                    used_vars.add(stmt.attr)
                        usage_matrix.append(len(used_vars) / len(instance_vars))
                    
                    total_cohesion += sum(usage_matrix) / len(usage_matrix)
                else:
                    total_cohesion += 1.0  # 没有实例变量的类认为是高内聚的
        
        return total_cohesion / max(1, class_count)
    
    def _calculate_maintainability_index(self, loc: int, complexity: float, function_count: int) -> float:
        """计算可维护性指数"""
        # 简化的可维护性指数计算
        # 基于代码行数、复杂度和函数数量
        
        if loc == 0:
            return 100.0
        
        # 标准化各个指标
        loc_factor = max(0, 100 - (loc / 10))  # 每10行代码减1分
        complexity_factor = max(0, 100 - (complexity * 10))  # 复杂度越高分数越低
        function_factor = min(100, function_count * 5)  # 函数数量适中加分
        
        maintainability = (loc_factor + complexity_factor + function_factor) / 3
        return min(100.0, max(0.0, maintainability))


class ArchitectureQualityValidator:
    """架构质量验证器主类"""
    
    def __init__(self):
        self.dependency_analyzer = DependencyAnalyzer()
        self.pattern_detector = DesignPatternDetector()
        self.rule_engine = ArchitectureRuleEngine()
        self.module_analyzer = ModuleQualityAnalyzer()
    
    def validate_project_architecture(self, project_path: str) -> Dict[str, Any]:
        """验证项目架构质量"""
        project_path = Path(project_path)
        python_files = list(project_path.rglob('*.py'))
        
        results = {
            'summary': {
                'total_files': len(python_files),
                'analysis_timestamp': datetime.now().isoformat(),
                'project_path': str(project_path)
            },
            'violations': [],
            'dependencies': [],
            'patterns': [],
            'module_metrics': [],
            'architecture_health': {}
        }
        
        logger.info(f"Analyzing {len(python_files)} Python files...")
        
        # 分析每个文件
        all_dependencies = []
        for file_path in python_files:
            try:
                # 依赖分析
                file_dependencies = self.dependency_analyzer.analyze_file_dependencies(str(file_path))
                all_dependencies.extend(file_dependencies)
                results['dependencies'].extend([dep.to_dict() for dep in file_dependencies])
                
                # 架构规则验证
                violations = self.rule_engine.validate_architecture(str(file_path), file_dependencies)
                results['violations'].extend([v.to_dict() for v in violations])
                
                # 设计模式检测
                patterns = self.pattern_detector.detect_patterns_in_file(str(file_path))
                results['patterns'].extend([p.to_dict() for p in patterns])
                
                # 模块质量分析
                module_metrics = self.module_analyzer.analyze_module(str(file_path))
                results['module_metrics'].append(module_metrics.to_dict())
                
            except Exception as e:
                logger.error(f"Failed to analyze {file_path}: {e}")
        
        # 全局架构分析
        results['architecture_health'] = self._calculate_architecture_health(results)
        
        # 检测循环依赖
        circular_deps = self.dependency_analyzer.detect_circular_dependencies()
        if circular_deps:
            for cycle in circular_deps:
                results['violations'].append({
                    'id': f"circular_dependency_{hash(tuple(cycle))}",
                    'type': 'circular_dependency',
                    'severity': 'high',
                    'file_path': 'multiple',
                    'line_number': 0,
                    'description': f"Circular dependency detected: {' -> '.join(cycle)}",
                    'rule_violated': 'no_circular_dependencies',
                    'suggestion': 'Refactor to remove circular dependencies using dependency inversion',
                    'impact': 'Makes code harder to test and maintain'
                })
        
        return results
    
    def _calculate_architecture_health(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算架构健康度"""
        violations = results['violations']
        module_metrics = results['module_metrics']
        patterns = results['patterns']
        
        # 违规统计
        violation_counts = {'high': 0, 'medium': 0, 'low': 0}
        for violation in violations:
            severity = violation.get('severity', 'low')
            violation_counts[severity] += 1
        
        # 模块质量统计
        if module_metrics:
            avg_complexity = sum(m['complexity_score'] for m in module_metrics) / len(module_metrics)
            avg_cohesion = sum(m['cohesion_score'] for m in module_metrics) / len(module_metrics)
            avg_coupling = sum(m['coupling_score'] for m in module_metrics) / len(module_metrics)
            avg_maintainability = sum(m['maintainability_index'] for m in module_metrics) / len(module_metrics)
        else:
            avg_complexity = avg_cohesion = avg_coupling = avg_maintainability = 0.0
        
        # 设计模式统计
        pattern_counts = {}
        for pattern in patterns:
            pattern_type = pattern.get('pattern_type', 'Unknown')
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        # 计算总体健康分数
        health_score = 100.0
        
        # 违规扣分
        health_score -= violation_counts['high'] * 10
        health_score -= violation_counts['medium'] * 5
        health_score -= violation_counts['low'] * 1
        
        # 复杂度扣分
        if avg_complexity > 5:
            health_score -= (avg_complexity - 5) * 5
        
        # 耦合度扣分
        if avg_coupling > 0.5:
            health_score -= (avg_coupling - 0.5) * 20
        
        # 内聚度加分
        health_score += avg_cohesion * 10
        
        # 可维护性加分
        health_score += (avg_maintainability - 50) / 5
        
        health_score = max(0.0, min(100.0, health_score))
        
        return {
            'overall_score': round(health_score, 2),
            'violation_summary': violation_counts,
            'quality_metrics': {
                'average_complexity': round(avg_complexity, 2),
                'average_cohesion': round(avg_cohesion, 2),
                'average_coupling': round(avg_coupling, 2),
                'average_maintainability': round(avg_maintainability, 2)
            },
            'pattern_distribution': pattern_counts,
            'recommendations': self._generate_recommendations(health_score, violation_counts, avg_complexity, avg_coupling)
        }
    
    def _generate_recommendations(self, health_score: float, violations: Dict[str, int], 
                                complexity: float, coupling: float) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if health_score < 50:
            recommendations.append("架构健康度较低，需要重点关注代码质量改进")
        
        if violations['high'] > 0:
            recommendations.append(f"优先解决 {violations['high']} 个高严重性架构违规")
        
        if complexity > 5:
            recommendations.append("代码复杂度较高，考虑重构复杂函数和类")
        
        if coupling > 0.7:
            recommendations.append("模块耦合度过高，应用依赖倒置原则降低耦合")
        
        if not recommendations:
            recommendations.append("架构质量良好，继续保持良好的编码实践")
        
        return recommendations
    
    def generate_architecture_report(self, results: Dict[str, Any]) -> str:
        """生成架构质量报告"""
        report_lines = []
        
        # 报告标题
        report_lines.append("# 代码架构质量验证报告")
        report_lines.append(f"\n**生成时间**: {results['summary']['analysis_timestamp']}")
        report_lines.append(f"**项目路径**: {results['summary']['project_path']}")
        report_lines.append(f"**分析文件数**: {results['summary']['total_files']}")
        
        # 架构健康度
        health = results['architecture_health']
        report_lines.append("\n## 🏥 架构健康度")
        report_lines.append(f"\n**总体评分**: {health['overall_score']}/100")
        
        score = health['overall_score']
        if score >= 80:
            status = "🟢 优秀"
        elif score >= 60:
            status = "🟡 良好"
        elif score >= 40:
            status = "🟠 需要改进"
        else:
            status = "🔴 需要重构"
        
        report_lines.append(f"**健康状态**: {status}")
        
        # 质量指标
        metrics = health['quality_metrics']
        report_lines.append("\n### 📊 质量指标")
        report_lines.append(f"- **平均复杂度**: {metrics['average_complexity']}")
        report_lines.append(f"- **平均内聚度**: {metrics['average_cohesion']}")
        report_lines.append(f"- **平均耦合度**: {metrics['average_coupling']}")
        report_lines.append(f"- **平均可维护性**: {metrics['average_maintainability']}")
        
        # 违规摘要
        violations = results['violations']
        violation_summary = health['violation_summary']
        
        report_lines.append("\n## ⚠️ 架构违规摘要")
        report_lines.append(f"\n**总违规数**: {len(violations)}")
        report_lines.append(f"- 高严重性: {violation_summary['high']}")
        report_lines.append(f"- 中严重性: {violation_summary['medium']}")
        report_lines.append(f"- 低严重性: {violation_summary['low']}")
        
        # 违规详情
        if violations:
            report_lines.append("\n### 🔍 违规详情")
            
            # 按类型分组
            violations_by_type = {}
            for violation in violations:
                vtype = violation['type']
                if vtype not in violations_by_type:
                    violations_by_type[vtype] = []
                violations_by_type[vtype].append(violation)
            
            for vtype, vlist in violations_by_type.items():
                report_lines.append(f"\n#### {vtype.replace('_', ' ').title()}")
                
                for violation in sorted(vlist, key=lambda x: x['severity'], reverse=True)[:5]:  # 只显示前5个
                    report_lines.append(f"\n- **{violation['severity'].upper()}**: {violation['description']}")
                    report_lines.append(f"  - 文件: {violation['file_path']}:{violation['line_number']}")
                    report_lines.append(f"  - 建议: {violation['suggestion']}")
        
        # 设计模式
        patterns = results['patterns']
        if patterns:
            report_lines.append("\n## 🎨 设计模式分析")
            
            pattern_counts = health['pattern_distribution']
            report_lines.append(f"\n**检测到的模式**: {len(patterns)}")
            
            for pattern_type, count in pattern_counts.items():
                report_lines.append(f"- {pattern_type}: {count}")
            
            # 模式详情
            report_lines.append("\n### 模式详情")
            for pattern in patterns[:10]:  # 只显示前10个
                report_lines.append(f"\n#### {pattern['pattern_name']} ({pattern['pattern_type']})")
                report_lines.append(f"- **文件**: {pattern['file_path']}")
                report_lines.append(f"- **类**: {pattern['class_name']}")
                report_lines.append(f"- **置信度**: {pattern['confidence']:.2f}")
                report_lines.append(f"- **描述**: {pattern['description']}")
                report_lines.append(f"- **实现质量**: {pattern['implementation_quality']}")
        
        # 依赖分析
        dependencies = results['dependencies']
        if dependencies:
            report_lines.append("\n## 🔗 依赖关系分析")
            report_lines.append(f"\n**总依赖数**: {len(dependencies)}")
            
            # 依赖类型统计
            dep_types = {}
            for dep in dependencies:
                dtype = dep['dependency_type']
                dep_types[dtype] = dep_types.get(dtype, 0) + 1
            
            for dtype, count in dep_types.items():
                report_lines.append(f"- {dtype}: {count}")
        
        # 模块质量
        module_metrics = results['module_metrics']
        if module_metrics:
            report_lines.append("\n## 📦 模块质量分析")
            
            # 找出质量最好和最差的模块
            sorted_modules = sorted(module_metrics, key=lambda x: x['maintainability_index'], reverse=True)
            
            if sorted_modules:
                report_lines.append("\n### 🏆 质量最佳模块")
                best_module = sorted_modules[0]
                report_lines.append(f"- **{best_module['module_path']}**")
                report_lines.append(f"  - 可维护性指数: {best_module['maintainability_index']:.2f}")
                report_lines.append(f"  - 复杂度: {best_module['complexity_score']:.2f}")
                report_lines.append(f"  - 内聚度: {best_module['cohesion_score']:.2f}")
                
                if len(sorted_modules) > 1:
                    report_lines.append("\n### ⚠️ 需要改进的模块")
                    worst_module = sorted_modules[-1]
                    report_lines.append(f"- **{worst_module['module_path']}**")
                    report_lines.append(f"  - 可维护性指数: {worst_module['maintainability_index']:.2f}")
                    report_lines.append(f"  - 复杂度: {worst_module['complexity_score']:.2f}")
                    report_lines.append(f"  - 耦合度: {worst_module['coupling_score']:.2f}")
        
        # 改进建议
        recommendations = health['recommendations']
        report_lines.append("\n## 💡 改进建议")
        
        for i, rec in enumerate(recommendations, 1):
            report_lines.append(f"\n{i}. {rec}")
        
        # 下一步行动
        report_lines.append("\n## 🎯 下一步行动")
        
        if violation_summary['high'] > 0:
            report_lines.append("\n1. **立即行动**: 解决高严重性架构违规")
            report_lines.append("   - 这些问题可能影响系统稳定性和安全性")
        
        if metrics['average_complexity'] > 5:
            report_lines.append("\n2. **重构计划**: 降低代码复杂度")
            report_lines.append("   - 识别复杂函数和类")
            report_lines.append("   - 应用提取方法等重构技术")
        
        if metrics['average_coupling'] > 0.7:
            report_lines.append("\n3. **架构优化**: 降低模块耦合")
            report_lines.append("   - 应用依赖倒置原则")
            report_lines.append("   - 引入接口和抽象层")
        
        report_lines.append("\n4. **持续监控**: 建立架构质量监控")
        report_lines.append("   - 在CI/CD中集成架构验证")
        report_lines.append("   - 定期生成架构质量报告")
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="VIVTransformer 代码架构质量验证器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --check-architecture src/
  %(prog)s --validate-patterns src/models/
  %(prog)s --analyze-dependencies
        """
    )
    
    parser.add_argument(
        '--check-architecture',
        metavar='PATH',
        help='检查指定目录的架构质量'
    )
    
    parser.add_argument(
        '--validate-patterns',
        metavar='PATH',
        help='验证指定目录的设计模式'
    )
    
    parser.add_argument(
        '--analyze-dependencies',
        action='store_true',
        help='分析项目依赖关系'
    )
    
    parser.add_argument(
        '--output',
        default='reports/architecture_quality_report.md',
        help='输出报告文件路径'
    )
    
    parser.add_argument(
        '--format',
        choices=['markdown', 'json'],
        default='markdown',
        help='输出格式'
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
    
    # 初始化验证器
    validator = ArchitectureQualityValidator()
    
    try:
        target_path = None
        
        if args.check_architecture:
            target_path = args.check_architecture
        elif args.validate_patterns:
            target_path = args.validate_patterns
        elif args.analyze_dependencies:
            target_path = '.'
        else:
            # 默认检查当前目录
            target_path = 'modify_multi_attention/'
        
        if not os.path.exists(target_path):
            print(f"❌ 路径不存在: {target_path}")
            sys.exit(1)
        
        print(f"🔍 分析架构质量: {target_path}")
        
        # 执行架构验证
        results = validator.validate_project_architecture(target_path)
        
        # 显示结果摘要
        health = results['architecture_health']
        print("\n" + "="*60)
        print("📊 架构质量分析结果")
        print(f"🏥 总体评分: {health['overall_score']}/100")
        
        violations = results['violations']
        print(f"⚠️  架构违规: {len(violations)}")
        
        violation_summary = health['violation_summary']
        if violation_summary['high'] > 0:
            print(f"   🔴 高严重性: {violation_summary['high']}")
        if violation_summary['medium'] > 0:
            print(f"   🟡 中严重性: {violation_summary['medium']}")
        if violation_summary['low'] > 0:
            print(f"   🟢 低严重性: {violation_summary['low']}")
        
        patterns = results['patterns']
        print(f"🎨 设计模式: {len(patterns)}")
        
        dependencies = results['dependencies']
        print(f"🔗 依赖关系: {len(dependencies)}")
        
        # 生成报告
        if args.format == 'json':
            output_content = json.dumps(results, indent=2, ensure_ascii=False)
        else:
            output_content = validator.generate_architecture_report(results)
        
        # 保存报告
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        print(f"\n📋 详细报告已保存: {args.output}")
        
        # 显示改进建议
        recommendations = health['recommendations']
        if recommendations:
            print("\n💡 改进建议:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⏹️  操作被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"架构验证失败: {e}")
        print(f"❌ 架构验证失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()