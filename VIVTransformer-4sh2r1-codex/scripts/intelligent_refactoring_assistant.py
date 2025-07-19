#!/usr/bin/env python3
"""
VIVTransformer 智能代码重构助手

这个脚本提供智能化的代码重构功能，包括：
- 自动检测重构机会
- 生成重构建议
- 自动应用安全的重构
- 重构影响分析
- 重构效果验证

使用方法:
    python scripts/intelligent_refactoring_assistant.py --analyze src/
    python scripts/intelligent_refactoring_assistant.py --refactor src/models/attention.py
    python scripts/intelligent_refactoring_assistant.py --auto-refactor --safe-only
"""

import argparse
import ast
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

import astor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RefactoringOpportunity:
    """重构机会数据类"""
    id: str
    type: str
    file_path: str
    line_start: int
    line_end: int
    severity: str
    confidence: float
    description: str
    suggestion: str
    estimated_effort: str
    benefits: List[str]
    risks: List[str]
    auto_applicable: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RefactoringResult:
    """重构结果数据类"""
    opportunity_id: str
    status: str
    original_code: str
    refactored_code: str
    changes_made: List[str]
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    improvement_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CodeComplexityAnalyzer:
    """代码复杂度分析器"""
    
    def __init__(self):
        self.metrics = {}
    
    def analyze_function(self, func_node: ast.FunctionDef, source_lines: List[str]) -> Dict[str, float]:
        """分析函数复杂度"""
        metrics = {
            'cyclomatic_complexity': self._calculate_cyclomatic_complexity(func_node),
            'cognitive_complexity': self._calculate_cognitive_complexity(func_node),
            'lines_of_code': self._count_lines_of_code(func_node, source_lines),
            'nesting_depth': self._calculate_nesting_depth(func_node),
            'parameter_count': len(func_node.args.args),
            'return_statements': self._count_return_statements(func_node)
        }
        
        return metrics
    
    def analyze_class(self, class_node: ast.ClassDef, source_lines: List[str]) -> Dict[str, float]:
        """分析类复杂度"""
        methods = [node for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        metrics = {
            'method_count': len(methods),
            'lines_of_code': self._count_lines_of_code(class_node, source_lines),
            'public_methods': len([m for m in methods if not m.name.startswith('_')]),
            'private_methods': len([m for m in methods if m.name.startswith('_')]),
            'inheritance_depth': len(class_node.bases),
            'avg_method_complexity': self._calculate_avg_method_complexity(methods)
        }
        
        return metrics
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """计算圈复杂度"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def _calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """计算认知复杂度"""
        complexity = 0
        nesting_level = 0
        
        def visit_node(n, level):
            nonlocal complexity
            
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1 + level
                level += 1
            elif isinstance(n, ast.ExceptHandler):
                complexity += 1 + level
                level += 1
            elif isinstance(n, (ast.And, ast.Or)):
                complexity += 1
            
            for child in ast.iter_child_nodes(n):
                visit_node(child, level)
        
        visit_node(node, nesting_level)
        return complexity
    
    def _count_lines_of_code(self, node: ast.AST, source_lines: List[str]) -> int:
        """计算代码行数"""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno - node.lineno + 1
        
        # 简化计算
        return len([n for n in ast.walk(node) if hasattr(n, 'lineno')])
    
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
    
    def _count_return_statements(self, node: ast.AST) -> int:
        """计算返回语句数量"""
        return len([n for n in ast.walk(node) if isinstance(n, ast.Return)])
    
    def _calculate_avg_method_complexity(self, methods: List[ast.FunctionDef]) -> float:
        """计算平均方法复杂度"""
        if not methods:
            return 0.0
        
        total_complexity = sum(self._calculate_cyclomatic_complexity(method) for method in methods)
        return total_complexity / len(methods)


class RefactoringOpportunityDetector:
    """重构机会检测器"""
    
    def __init__(self):
        self.complexity_analyzer = CodeComplexityAnalyzer()
        self.opportunities = []
    
    def detect_opportunities(self, file_path: str) -> List[RefactoringOpportunity]:
        """检测重构机会"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            source_lines = source_code.split('\n')
            
            opportunities = []
            
            # 检测各种重构机会
            opportunities.extend(self._detect_long_methods(tree, source_lines, file_path))
            opportunities.extend(self._detect_large_classes(tree, source_lines, file_path))
            opportunities.extend(self._detect_complex_conditionals(tree, source_lines, file_path))
            opportunities.extend(self._detect_duplicate_code(tree, source_lines, file_path))
            opportunities.extend(self._detect_god_objects(tree, source_lines, file_path))
            opportunities.extend(self._detect_feature_envy(tree, source_lines, file_path))
            opportunities.extend(self._detect_data_clumps(tree, source_lines, file_path))
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Failed to detect opportunities in {file_path}: {e}")
            return []
    
    def _detect_long_methods(self, tree: ast.AST, source_lines: List[str], file_path: str) -> List[RefactoringOpportunity]:
        """检测长方法"""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics = self.complexity_analyzer.analyze_function(node, source_lines)
                
                if metrics['lines_of_code'] > 50 or metrics['cyclomatic_complexity'] > 10:
                    severity = 'high' if metrics['lines_of_code'] > 100 else 'medium'
                    confidence = min(0.9, metrics['lines_of_code'] / 100)
                    
                    opportunities.append(RefactoringOpportunity(
                        id=f"long_method_{node.name}_{node.lineno}",
                        type="extract_method",
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno + 10),
                        severity=severity,
                        confidence=confidence,
                        description=f"Method '{node.name}' is too long ({metrics['lines_of_code']} lines)",
                        suggestion="Extract smaller methods to improve readability and maintainability",
                        estimated_effort="medium",
                        benefits=[
                            "Improved readability",
                            "Better testability",
                            "Easier maintenance",
                            "Reduced complexity"
                        ],
                        risks=[
                            "May introduce coupling between extracted methods",
                            "Requires careful parameter passing"
                        ],
                        auto_applicable=metrics['lines_of_code'] < 80
                    ))
        
        return opportunities
    
    def _detect_large_classes(self, tree: ast.AST, source_lines: List[str], file_path: str) -> List[RefactoringOpportunity]:
        """检测大类"""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                metrics = self.complexity_analyzer.analyze_class(node, source_lines)
                
                if metrics['method_count'] > 20 or metrics['lines_of_code'] > 500:
                    severity = 'high' if metrics['method_count'] > 30 else 'medium'
                    confidence = min(0.8, metrics['method_count'] / 30)
                    
                    opportunities.append(RefactoringOpportunity(
                        id=f"large_class_{node.name}_{node.lineno}",
                        type="extract_class",
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno + 50),
                        severity=severity,
                        confidence=confidence,
                        description=f"Class '{node.name}' is too large ({metrics['method_count']} methods)",
                        suggestion="Split class into smaller, more focused classes",
                        estimated_effort="high",
                        benefits=[
                            "Single responsibility principle",
                            "Better modularity",
                            "Easier testing",
                            "Improved maintainability"
                        ],
                        risks=[
                            "May break existing dependencies",
                            "Requires careful interface design",
                            "High refactoring effort"
                        ],
                        auto_applicable=False
                    ))
        
        return opportunities
    
    def _detect_complex_conditionals(self, tree: ast.AST, source_lines: List[str], file_path: str) -> List[RefactoringOpportunity]:
        """检测复杂条件语句"""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                complexity = self._count_conditional_complexity(node.test)
                nesting_depth = self._get_nesting_depth(node, tree)
                
                if complexity > 5 or nesting_depth > 3:
                    severity = 'high' if complexity > 8 else 'medium'
                    confidence = min(0.7, complexity / 10)
                    
                    opportunities.append(RefactoringOpportunity(
                        id=f"complex_conditional_{node.lineno}",
                        type="simplify_conditional",
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno + 5),
                        severity=severity,
                        confidence=confidence,
                        description=f"Complex conditional statement (complexity: {complexity})",
                        suggestion="Extract condition to a well-named method or use guard clauses",
                        estimated_effort="low",
                        benefits=[
                            "Improved readability",
                            "Better testability",
                            "Easier debugging"
                        ],
                        risks=[
                            "May change execution flow",
                            "Requires careful testing"
                        ],
                        auto_applicable=complexity < 7
                    ))
        
        return opportunities
    
    def _detect_duplicate_code(self, tree: ast.AST, source_lines: List[str], file_path: str) -> List[RefactoringOpportunity]:
        """检测重复代码"""
        opportunities = []
        
        # 简化的重复代码检测
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        for i, func1 in enumerate(functions):
            for func2 in functions[i+1:]:
                similarity = self._calculate_function_similarity(func1, func2)
                
                if similarity > 0.7:
                    opportunities.append(RefactoringOpportunity(
                        id=f"duplicate_code_{func1.name}_{func2.name}_{func1.lineno}",
                        type="extract_common_method",
                        file_path=file_path,
                        line_start=min(func1.lineno, func2.lineno),
                        line_end=max(getattr(func1, 'end_lineno', func1.lineno + 10),
                                   getattr(func2, 'end_lineno', func2.lineno + 10)),
                        severity="medium",
                        confidence=similarity,
                        description=f"Similar code found in '{func1.name}' and '{func2.name}' (similarity: {similarity:.2f})",
                        suggestion="Extract common functionality into a shared method",
                        estimated_effort="medium",
                        benefits=[
                            "Reduced code duplication",
                            "Easier maintenance",
                            "Single source of truth"
                        ],
                        risks=[
                            "May introduce coupling",
                            "Requires parameter generalization"
                        ],
                        auto_applicable=similarity > 0.8
                    ))
        
        return opportunities
    
    def _detect_god_objects(self, tree: ast.AST, source_lines: List[str], file_path: str) -> List[RefactoringOpportunity]:
        """检测上帝对象"""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                metrics = self.complexity_analyzer.analyze_class(node, source_lines)
                responsibilities = self._count_class_responsibilities(node)
                
                if responsibilities > 5 and metrics['method_count'] > 15:
                    opportunities.append(RefactoringOpportunity(
                        id=f"god_object_{node.name}_{node.lineno}",
                        type="decompose_class",
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno + 100),
                        severity="high",
                        confidence=0.8,
                        description=f"Class '{node.name}' has too many responsibilities ({responsibilities})",
                        suggestion="Decompose into smaller, focused classes following SRP",
                        estimated_effort="high",
                        benefits=[
                            "Single Responsibility Principle",
                            "Better testability",
                            "Improved maintainability",
                            "Reduced coupling"
                        ],
                        risks=[
                            "Major architectural change",
                            "High refactoring effort",
                            "May break existing code"
                        ],
                        auto_applicable=False
                    ))
        
        return opportunities
    
    def _detect_feature_envy(self, tree: ast.AST, source_lines: List[str], file_path: str) -> List[RefactoringOpportunity]:
        """检测特性依恋"""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                external_calls = self._count_external_method_calls(node)
                
                if external_calls > 5:
                    opportunities.append(RefactoringOpportunity(
                        id=f"feature_envy_{node.name}_{node.lineno}",
                        type="move_method",
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno + 20),
                        severity="medium",
                        confidence=0.6,
                        description=f"Method '{node.name}' makes many external calls ({external_calls})",
                        suggestion="Consider moving method closer to the data it uses",
                        estimated_effort="medium",
                        benefits=[
                            "Better encapsulation",
                            "Reduced coupling",
                            "Improved cohesion"
                        ],
                        risks=[
                            "May change class interface",
                            "Requires dependency analysis"
                        ],
                        auto_applicable=False
                    ))
        
        return opportunities
    
    def _detect_data_clumps(self, tree: ast.AST, source_lines: List[str], file_path: str) -> List[RefactoringOpportunity]:
        """检测数据泥团"""
        opportunities = []
        
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        # 分析参数模式
        parameter_patterns = {}
        for func in functions:
            params = [arg.arg for arg in func.args.args if arg.arg != 'self']
            if len(params) >= 3:
                param_signature = tuple(sorted(params))
                if param_signature not in parameter_patterns:
                    parameter_patterns[param_signature] = []
                parameter_patterns[param_signature].append(func)
        
        # 检测重复的参数模式
        for params, funcs in parameter_patterns.items():
            if len(funcs) >= 2 and len(params) >= 3:
                opportunities.append(RefactoringOpportunity(
                    id=f"data_clump_{hash(params)}",
                    type="introduce_parameter_object",
                    file_path=file_path,
                    line_start=min(f.lineno for f in funcs),
                    line_end=max(getattr(f, 'end_lineno', f.lineno + 10) for f in funcs),
                    severity="medium",
                    confidence=0.7,
                    description=f"Data clump detected: {len(funcs)} functions share {len(params)} parameters",
                    suggestion="Introduce a parameter object to group related data",
                    estimated_effort="medium",
                    benefits=[
                        "Reduced parameter lists",
                        "Better data organization",
                        "Improved maintainability"
                    ],
                    risks=[
                        "Changes function signatures",
                        "May introduce new dependencies"
                    ],
                    auto_applicable=False
                ))
        
        return opportunities
    
    def _count_conditional_complexity(self, node: ast.AST) -> int:
        """计算条件复杂度"""
        complexity = 0
        
        for child in ast.walk(node):
            if isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.Compare):
                complexity += len(child.ops)
        
        return max(1, complexity)
    
    def _get_nesting_depth(self, target_node: ast.AST, tree: ast.AST) -> int:
        """获取节点的嵌套深度"""
        depth = 0
        
        def find_depth(node, current_depth):
            nonlocal depth
            
            if node is target_node:
                depth = current_depth
                return
            
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                current_depth += 1
            
            for child in ast.iter_child_nodes(node):
                find_depth(child, current_depth)
        
        find_depth(tree, 0)
        return depth
    
    def _calculate_function_similarity(self, func1: ast.FunctionDef, func2: ast.FunctionDef) -> float:
        """计算函数相似度"""
        try:
            # 简化的相似度计算
            code1 = astor.to_source(func1)
            code2 = astor.to_source(func2)
            
            # 移除函数名和参数名的影响
            normalized_code1 = re.sub(r'\b\w+\b', 'VAR', code1)
            normalized_code2 = re.sub(r'\b\w+\b', 'VAR', code2)
            
            # 计算编辑距离的简化版本
            lines1 = normalized_code1.split('\n')
            lines2 = normalized_code2.split('\n')
            
            common_lines = len(set(lines1) & set(lines2))
            total_lines = len(set(lines1) | set(lines2))
            
            return common_lines / max(1, total_lines)
            
        except Exception:
            return 0.0
    
    def _count_class_responsibilities(self, class_node: ast.ClassDef) -> int:
        """计算类的职责数量"""
        # 简化的职责计算：基于方法名的动词分析
        verbs = set()
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                # 提取方法名中的动词
                method_parts = re.findall(r'[A-Z][a-z]*|[a-z]+', node.name)
                if method_parts:
                    verbs.add(method_parts[0].lower())
        
        return len(verbs)
    
    def _count_external_method_calls(self, func_node: ast.FunctionDef) -> int:
        """计算外部方法调用数量"""
        external_calls = 0
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Attribute):
                external_calls += 1
        
        return external_calls


class AutomaticRefactorer:
    """自动重构器"""
    
    def __init__(self):
        self.complexity_analyzer = CodeComplexityAnalyzer()
    
    def apply_refactoring(self, opportunity: RefactoringOpportunity) -> RefactoringResult:
        """应用重构"""
        try:
            with open(opportunity.file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
            
            tree = ast.parse(original_code)
            
            # 计算重构前的指标
            metrics_before = self._calculate_file_metrics(tree, original_code.split('\n'))
            
            # 应用具体的重构
            refactored_tree = self._apply_specific_refactoring(tree, opportunity)
            
            if refactored_tree is None:
                return RefactoringResult(
                    opportunity_id=opportunity.id,
                    status="failed",
                    original_code=original_code,
                    refactored_code=original_code,
                    changes_made=[],
                    metrics_before=metrics_before,
                    metrics_after=metrics_before,
                    improvement_score=0.0
                )
            
            # 生成重构后的代码
            refactored_code = astor.to_source(refactored_tree)
            
            # 计算重构后的指标
            metrics_after = self._calculate_file_metrics(refactored_tree, refactored_code.split('\n'))
            
            # 计算改进分数
            improvement_score = self._calculate_improvement_score(metrics_before, metrics_after)
            
            return RefactoringResult(
                opportunity_id=opportunity.id,
                status="success",
                original_code=original_code,
                refactored_code=refactored_code,
                changes_made=self._identify_changes(original_code, refactored_code),
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                improvement_score=improvement_score
            )
            
        except Exception as e:
            logger.error(f"Refactoring failed for {opportunity.id}: {e}")
            return RefactoringResult(
                opportunity_id=opportunity.id,
                status="error",
                original_code="",
                refactored_code="",
                changes_made=[],
                metrics_before={},
                metrics_after={},
                improvement_score=0.0
            )
    
    def _apply_specific_refactoring(self, tree: ast.AST, opportunity: RefactoringOpportunity) -> Optional[ast.AST]:
        """应用具体的重构类型"""
        if opportunity.type == "extract_method":
            return self._extract_method(tree, opportunity)
        elif opportunity.type == "simplify_conditional":
            return self._simplify_conditional(tree, opportunity)
        elif opportunity.type == "extract_common_method":
            return self._extract_common_method(tree, opportunity)
        else:
            logger.warning(f"Unsupported refactoring type: {opportunity.type}")
            return None
    
    def _extract_method(self, tree: ast.AST, opportunity: RefactoringOpportunity) -> Optional[ast.AST]:
        """提取方法重构"""
        # 找到目标函数
        target_function = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef) and 
                node.lineno <= opportunity.line_start <= 
                getattr(node, 'end_lineno', node.lineno + 100)):
                target_function = node
                break
        
        if not target_function:
            return None
        
        # 简化的方法提取：将函数分成两部分
        if len(target_function.body) > 6:
            # 提取后半部分为新方法
            split_point = len(target_function.body) // 2
            
            # 创建新方法
            extracted_body = target_function.body[split_point:]
            new_method_name = f"{target_function.name}_extracted"
            
            new_method = ast.FunctionDef(
                name=new_method_name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg='self', annotation=None)] if target_function.args.args and target_function.args.args[0].arg == 'self' else [],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[]
                ),
                body=extracted_body,
                decorator_list=[],
                returns=None
            )
            
            # 修改原方法
            target_function.body = target_function.body[:split_point]
            
            # 添加对新方法的调用
            call_node = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='self', ctx=ast.Load()) if target_function.args.args and target_function.args.args[0].arg == 'self' else ast.Name(id=new_method_name, ctx=ast.Load()),
                        attr=new_method_name,
                        ctx=ast.Load()
                    ) if target_function.args.args and target_function.args.args[0].arg == 'self' else ast.Name(id=new_method_name, ctx=ast.Load()),
                    args=[],
                    keywords=[]
                )
            )
            target_function.body.append(call_node)
            
            # 将新方法添加到类中
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if child is target_function:
                            node.body.append(new_method)
                            break
                    break
        
        return tree
    
    def _simplify_conditional(self, tree: ast.AST, opportunity: RefactoringOpportunity) -> Optional[ast.AST]:
        """简化条件语句"""
        # 找到目标条件语句
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and node.lineno == opportunity.line_start:
                # 简化复杂的布尔表达式
                if isinstance(node.test, ast.BoolOp):
                    # 如果是 And 操作，尝试使用早期返回
                    if isinstance(node.test, ast.And):
                        # 转换为多个 if 语句
                        new_ifs = []
                        for condition in node.test.values:
                            new_if = ast.If(
                                test=ast.UnaryOp(op=ast.Not(), operand=condition),
                                body=[ast.Return(value=None)],
                                orelse=[]
                            )
                            new_ifs.append(new_if)
                        
                        # 替换原始 if 语句
                        parent = self._find_parent(tree, node)
                        if parent and hasattr(parent, 'body'):
                            index = parent.body.index(node)
                            parent.body[index:index+1] = new_ifs + node.body
                
                break
        
        return tree
    
    def _extract_common_method(self, tree: ast.AST, opportunity: RefactoringOpportunity) -> Optional[ast.AST]:
        """提取公共方法"""
        # 这是一个复杂的重构，需要更详细的实现
        # 这里提供一个简化版本
        return tree
    
    def _find_parent(self, tree: ast.AST, target: ast.AST) -> Optional[ast.AST]:
        """查找节点的父节点"""
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                if child is target:
                    return node
        return None
    
    def _calculate_file_metrics(self, tree: ast.AST, source_lines: List[str]) -> Dict[str, float]:
        """计算文件指标"""
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        total_complexity = 0
        total_lines = 0
        
        for func in functions:
            func_metrics = self.complexity_analyzer.analyze_function(func, source_lines)
            total_complexity += func_metrics['cyclomatic_complexity']
            total_lines += func_metrics['lines_of_code']
        
        return {
            'total_functions': len(functions),
            'total_classes': len(classes),
            'average_complexity': total_complexity / max(1, len(functions)),
            'total_lines': total_lines,
            'average_function_length': total_lines / max(1, len(functions))
        }
    
    def _calculate_improvement_score(self, before: Dict[str, float], after: Dict[str, float]) -> float:
        """计算改进分数"""
        improvements = []
        
        # 复杂度改进
        if 'average_complexity' in before and 'average_complexity' in after:
            complexity_improvement = (before['average_complexity'] - after['average_complexity']) / max(1, before['average_complexity'])
            improvements.append(complexity_improvement)
        
        # 函数长度改进
        if 'average_function_length' in before and 'average_function_length' in after:
            length_improvement = (before['average_function_length'] - after['average_function_length']) / max(1, before['average_function_length'])
            improvements.append(length_improvement)
        
        return sum(improvements) / max(1, len(improvements)) * 100
    
    def _identify_changes(self, original: str, refactored: str) -> List[str]:
        """识别变更"""
        changes = []
        
        original_lines = original.split('\n')
        refactored_lines = refactored.split('\n')
        
        if len(refactored_lines) != len(original_lines):
            changes.append(f"Line count changed from {len(original_lines)} to {len(refactored_lines)}")
        
        # 简化的变更检测
        if original != refactored:
            changes.append("Code structure modified")
        
        return changes


class IntelligentRefactoringAssistant:
    """智能重构助手主类"""
    
    def __init__(self):
        self.detector = RefactoringOpportunityDetector()
        self.refactorer = AutomaticRefactorer()
        self.results_history = []
    
    def analyze_file(self, file_path: str) -> List[RefactoringOpportunity]:
        """分析文件的重构机会"""
        logger.info(f"Analyzing refactoring opportunities in {file_path}")
        return self.detector.detect_opportunities(file_path)
    
    def analyze_directory(self, directory: str) -> Dict[str, List[RefactoringOpportunity]]:
        """分析目录中所有文件的重构机会"""
        directory_path = Path(directory)
        python_files = list(directory_path.rglob('*.py'))
        
        results = {}
        
        for file_path in python_files:
            opportunities = self.analyze_file(str(file_path))
            if opportunities:
                results[str(file_path)] = opportunities
        
        return results
    
    def apply_safe_refactorings(self, opportunities: List[RefactoringOpportunity]) -> List[RefactoringResult]:
        """应用安全的重构"""
        results = []
        
        # 只应用标记为自动适用的重构
        safe_opportunities = [op for op in opportunities if op.auto_applicable and op.confidence > 0.7]
        
        for opportunity in safe_opportunities:
            logger.info(f"Applying refactoring: {opportunity.id}")
            result = self.refactorer.apply_refactoring(opportunity)
            results.append(result)
            
            # 如果重构成功且有改进，保存文件
            if result.status == "success" and result.improvement_score > 0:
                self._backup_and_save(opportunity.file_path, result.refactored_code)
        
        return results
    
    def _backup_and_save(self, file_path: str, refactored_code: str):
        """备份原文件并保存重构后的代码"""
        # 创建备份
        backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(file_path, backup_path)
        logger.info(f"Backup created: {backup_path}")
        
        # 保存重构后的代码
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(refactored_code)
        
        logger.info(f"Refactored code saved to {file_path}")
    
    def generate_refactoring_report(self, opportunities: List[RefactoringOpportunity], 
                                  results: List[RefactoringResult] = None) -> str:
        """生成重构报告"""
        report_lines = []
        
        # 报告标题
        report_lines.append("# 智能重构分析报告")
        report_lines.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**发现的重构机会**: {len(opportunities)}")
        
        # 按类型分组
        opportunities_by_type = {}
        for op in opportunities:
            if op.type not in opportunities_by_type:
                opportunities_by_type[op.type] = []
            opportunities_by_type[op.type].append(op)
        
        # 重构机会摘要
        report_lines.append("\n## 📊 重构机会摘要")
        for refactor_type, ops in opportunities_by_type.items():
            high_priority = len([op for op in ops if op.severity == 'high'])
            medium_priority = len([op for op in ops if op.severity == 'medium'])
            low_priority = len([op for op in ops if op.severity == 'low'])
            
            report_lines.append(f"\n### {refactor_type.replace('_', ' ').title()}")
            report_lines.append(f"- 总数: {len(ops)}")
            report_lines.append(f"- 高优先级: {high_priority}")
            report_lines.append(f"- 中优先级: {medium_priority}")
            report_lines.append(f"- 低优先级: {low_priority}")
        
        # 详细机会列表
        report_lines.append("\n## 🎯 详细重构机会")
        
        for refactor_type, ops in opportunities_by_type.items():
            report_lines.append(f"\n### {refactor_type.replace('_', ' ').title()}")
            
            for op in sorted(ops, key=lambda x: x.confidence, reverse=True):
                report_lines.append(f"\n#### {op.id}")
                report_lines.append(f"- **文件**: {op.file_path}")
                report_lines.append(f"- **位置**: 第 {op.line_start}-{op.line_end} 行")
                report_lines.append(f"- **严重程度**: {op.severity}")
                report_lines.append(f"- **置信度**: {op.confidence:.2f}")
                report_lines.append(f"- **描述**: {op.description}")
                report_lines.append(f"- **建议**: {op.suggestion}")
                report_lines.append(f"- **预估工作量**: {op.estimated_effort}")
                report_lines.append(f"- **自动适用**: {'是' if op.auto_applicable else '否'}")
                
                if op.benefits:
                    report_lines.append("- **收益**:")
                    for benefit in op.benefits:
                        report_lines.append(f"  - {benefit}")
                
                if op.risks:
                    report_lines.append("- **风险**:")
                    for risk in op.risks:
                        report_lines.append(f"  - {risk}")
        
        # 重构结果
        if results:
            report_lines.append("\n## 🔧 重构执行结果")
            
            successful_refactorings = [r for r in results if r.status == "success"]
            failed_refactorings = [r for r in results if r.status == "failed"]
            
            report_lines.append(f"\n- **执行的重构**: {len(results)}")
            report_lines.append(f"- **成功**: {len(successful_refactorings)}")
            report_lines.append(f"- **失败**: {len(failed_refactorings)}")
            
            if successful_refactorings:
                avg_improvement = sum(r.improvement_score for r in successful_refactorings) / len(successful_refactorings)
                report_lines.append(f"- **平均改进分数**: {avg_improvement:.2f}%")
                
                report_lines.append("\n### 成功的重构")
                for result in successful_refactorings:
                    report_lines.append(f"\n#### {result.opportunity_id}")
                    report_lines.append(f"- **改进分数**: {result.improvement_score:.2f}%")
                    report_lines.append(f"- **变更**: {', '.join(result.changes_made)}")
        
        # 建议和下一步
        report_lines.append("\n## 💡 建议和下一步")
        
        high_priority_ops = [op for op in opportunities if op.severity == 'high']
        auto_applicable_ops = [op for op in opportunities if op.auto_applicable]
        
        if high_priority_ops:
            report_lines.append(f"\n1. **优先处理高优先级重构** ({len(high_priority_ops)} 个)")
            for op in high_priority_ops[:3]:  # 只显示前3个
                report_lines.append(f"   - {op.description}")
        
        if auto_applicable_ops:
            report_lines.append(f"\n2. **考虑自动应用安全重构** ({len(auto_applicable_ops)} 个)")
            report_lines.append("   - 这些重构风险较低，可以自动应用")
        
        report_lines.append("\n3. **建立重构计划**")
        report_lines.append("   - 按优先级和工作量制定重构计划")
        report_lines.append("   - 确保充分的测试覆盖")
        report_lines.append("   - 逐步进行，避免大规模变更")
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="VIVTransformer 智能重构助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --analyze src/
  %(prog)s --refactor src/models/attention.py
  %(prog)s --auto-refactor --safe-only
        """
    )
    
    parser.add_argument(
        '--analyze',
        metavar='PATH',
        help='分析指定文件或目录的重构机会'
    )
    
    parser.add_argument(
        '--refactor',
        metavar='FILE',
        help='对指定文件进行重构分析和建议'
    )
    
    parser.add_argument(
        '--auto-refactor',
        action='store_true',
        help='自动应用安全的重构'
    )
    
    parser.add_argument(
        '--safe-only',
        action='store_true',
        help='只应用标记为安全的重构'
    )
    
    parser.add_argument(
        '--output',
        default='reports/refactoring_report.md',
        help='输出报告文件路径'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.7,
        help='重构置信度阈值 (0.0-1.0)'
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
    
    # 初始化重构助手
    assistant = IntelligentRefactoringAssistant()
    
    try:
        opportunities = []
        results = []
        
        # 分析模式
        if args.analyze:
            if os.path.isfile(args.analyze):
                opportunities = assistant.analyze_file(args.analyze)
                print(f"📁 分析文件: {args.analyze}")
            elif os.path.isdir(args.analyze):
                file_opportunities = assistant.analyze_directory(args.analyze)
                opportunities = []
                for file_path, ops in file_opportunities.items():
                    opportunities.extend(ops)
                print(f"📁 分析目录: {args.analyze}")
            else:
                print(f"❌ 路径不存在: {args.analyze}")
                sys.exit(1)
        
        # 重构模式
        elif args.refactor:
            if os.path.isfile(args.refactor):
                opportunities = assistant.analyze_file(args.refactor)
                print(f"🔧 重构分析: {args.refactor}")
            else:
                print(f"❌ 文件不存在: {args.refactor}")
                sys.exit(1)
        
        # 自动重构模式
        elif args.auto_refactor:
            # 分析当前目录
            file_opportunities = assistant.analyze_directory('.')
            opportunities = []
            for file_path, ops in file_opportunities.items():
                opportunities.extend(ops)
            
            # 过滤置信度
            high_confidence_ops = [op for op in opportunities if op.confidence >= args.confidence_threshold]
            
            if args.safe_only:
                high_confidence_ops = [op for op in high_confidence_ops if op.auto_applicable]
            
            if high_confidence_ops:
                print(f"🤖 自动应用 {len(high_confidence_ops)} 个重构...")
                results = assistant.apply_safe_refactorings(high_confidence_ops)
            else:
                print("ℹ️  没有找到符合条件的安全重构")
        
        else:
            # 默认分析当前目录
            file_opportunities = assistant.analyze_directory('modify_multi_attention/')
            opportunities = []
            for file_path, ops in file_opportunities.items():
                opportunities.extend(ops)
            print("📁 分析默认目录: modify_multi_attention/")
        
        # 显示结果摘要
        if opportunities:
            print("\n" + "="*60)
            print("📊 重构分析结果")
            print(f"🎯 发现重构机会: {len(opportunities)}")
            
            # 按严重程度统计
            severity_counts = {}
            for op in opportunities:
                severity_counts[op.severity] = severity_counts.get(op.severity, 0) + 1
            
            for severity, count in severity_counts.items():
                print(f"   {severity}: {count}")
            
            # 按类型统计
            type_counts = {}
            for op in opportunities:
                type_counts[op.type] = type_counts.get(op.type, 0) + 1
            
            print("\n🔧 重构类型分布:")
            for refactor_type, count in type_counts.items():
                print(f"   {refactor_type.replace('_', ' ').title()}: {count}")
            
            # 自动适用统计
            auto_applicable = len([op for op in opportunities if op.auto_applicable])
            print(f"\n🤖 可自动应用: {auto_applicable}/{len(opportunities)}")
        
        # 显示重构结果
        if results:
            successful = len([r for r in results if r.status == "success"])
            print(f"\n✅ 重构执行结果: {successful}/{len(results)} 成功")
            
            if successful > 0:
                avg_improvement = sum(r.improvement_score for r in results if r.status == "success") / successful
                print(f"📈 平均改进分数: {avg_improvement:.2f}%")
        
        # 生成报告
        if opportunities or results:
            report = assistant.generate_refactoring_report(opportunities, results)
            
            # 保存报告
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"\n📋 详细报告已保存: {args.output}")
        
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⏹️  操作被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"重构分析失败: {e}")
        print(f"❌ 重构分析失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()