import os
import re
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import argparse
import ast
from datetime import datetime
import statistics

try:
    import yaml  # Optional
except Exception:
    yaml = None

try:
    import numpy as np
except ImportError:
    np = None


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        try:
            # scientific notation like 7.9e-05
            return float(x.strip())
        except Exception:
            return None


def safe_json_load(file_path: Path) -> Optional[Dict[str, Any]]:
    """Safely load JSON file with comprehensive error handling"""
    try:
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return None
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Warning: JSON decode error in {file_path}: {e}")
        return None
    except UnicodeDecodeError as e:
        print(f"Warning: Unicode decode error in {file_path}: {e}")
        try:
            # Try with different encoding
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        except Exception:
            pass
        return None
    except Exception as e:
        print(f"Warning: Unexpected error reading {file_path}: {e}")
        return None


def safe_file_read(file_path: Path, encoding: str = 'utf-8') -> Optional[str]:
    """Safely read text file with fallback encodings"""
    encodings = [encoding, 'utf-8', 'gbk', 'latin-1']
    
    for enc in encodings:
        try:
            if not file_path.exists():
                return None
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Warning: Error reading {file_path} with {enc}: {e}")
            continue
    
    print(f"Error: Could not read {file_path} with any encoding")
    return None


def validate_experiment_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean experiment data"""
    validated = data.copy()
    
    # Validate numeric fields
    numeric_fields = [
        'final_train_loss', 'final_valid_loss', 'final_test_loss',
        'best_valid_loss', 'avg_test_loss', 'min_test_loss', 'max_test_loss',
        'total_duration_seconds', 'learning_rate_initial', 'learning_rate_final'
    ]
    
    for field in numeric_fields:
        if field in validated and validated[field] is not None:
            try:
                validated[field] = float(validated[field])
                # Check for reasonable ranges
                if field.endswith('_loss') and (validated[field] < 0 or validated[field] > 1000):
                    print(f"Warning: Unusual loss value {validated[field]} for {field}")
                elif field == 'total_duration_seconds' and (validated[field] < 0 or validated[field] > 86400 * 7):  # 7 days
                    print(f"Warning: Unusual duration {validated[field]} seconds")
            except (ValueError, TypeError):
                print(f"Warning: Invalid numeric value for {field}: {validated[field]}")
                validated[field] = None
    
    # Validate string fields
    string_fields = ['attention', 'source_group', 'status']
    for field in string_fields:
        if field in validated and validated[field] is not None:
            validated[field] = str(validated[field]).strip()
            if not validated[field]:  # Empty after strip
                validated[field] = None
    
    # Ensure status is valid
    if validated.get('status') not in ['ok', 'failed', None]:
        print(f"Warning: Unknown status '{validated.get('status')}', setting to None")
        validated['status'] = None
    
    return validated


def parse_loss_log(loss_log_path: Path) -> Dict[str, Any]:
    """Parse loss_logs/loss_log.txt with header: Epoch, Train Loss, Valid Loss, Test Loss"""
    result = {
        "epochs": 0,
        "final_train_loss": None,
        "final_valid_loss": None,
        "final_test_loss": None,
        "best_valid_loss": None,
        "best_valid_epoch": None,
    }
    if not loss_log_path.exists():
        return result

    lines = [line.strip() for line in loss_log_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines or len(lines) <= 1:
        return result

    header = [h.strip().lower() for h in lines[0].split(',')]
    # Build column indices safely
    idx = {name: header.index(name) if name in header else None for name in ["epoch", "train loss", "valid loss", "test loss"]}

    best_valid: Tuple[int, float] = (-1, float('inf'))

    for row in lines[1:]:
        parts = [p.strip() for p in row.split(',')]
        if idx["epoch"] is not None and idx["epoch"] < len(parts):
            epoch = int(parts[idx["epoch"]])
            result["epochs"] = max(result["epochs"], epoch)
        else:
            epoch = result["epochs"] + 1
            result["epochs"] = epoch

        if idx["train loss"] is not None and idx["train loss"] < len(parts):
            result["final_train_loss"] = safe_float(parts[idx["train loss"]])
        if idx["valid loss"] is not None and idx["valid loss"] < len(parts):
            v = safe_float(parts[idx["valid loss"]])
            result["final_valid_loss"] = v
            if v is not None and v < best_valid[1]:
                best_valid = (epoch, v)
        if idx["test loss"] is not None and idx["test loss"] < len(parts):
            result["final_test_loss"] = safe_float(parts[idx["test loss"]])

    if best_valid[0] != -1:
        result["best_valid_epoch"] = best_valid[0]
        result["best_valid_loss"] = best_valid[1]
    return result


def parse_test_result_file(txt_path: Path) -> Optional[float]:
    """Parse 'Test Loss for xxx: value'"""
    if not txt_path.exists():
        return None
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Test\s+Loss\s+for\s+[^:]+:\s*([0-9eE\.+\-]+)", text)
    if m:
        return safe_float(m.group(1))
    return None


def parse_test_loss_log(txt_path: Path) -> Dict[str, Optional[float]]:
    """Parse test_results/test_loss_log.txt with lines and 'Average Test Loss: X'"""
    result = {"avg_test_loss": None, "min_test_loss": None, "max_test_loss": None}
    if not txt_path.exists():
        return result
    values: List[float] = []
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("batch"):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                v = safe_float(parts[1])
                if v is not None:
                    values.append(v)
        elif line.lower().startswith("average test loss"):
            m = re.search(r":\s*([0-9eE\.+\-]+)", line)
            if m:
                result["avg_test_loss"] = safe_float(m.group(1))
    if values:
        result["min_test_loss"] = min(values)
        result["max_test_loss"] = max(values)
        if result["avg_test_loss"] is None:
            result["avg_test_loss"] = sum(values) / len(values)
    return result


def read_hardware_summary(research_dir: Path) -> Optional[Dict[str, Any]]:
    """Read hardware monitoring summary data"""
    hardware_file = research_dir / "hardware_summary.json"
    data = safe_json_load(hardware_file)
    if data:
        return {
            "avg_gpu_utilization": data.get("avg_gpu_utilization"),
            "max_gpu_memory_mb": data.get("max_gpu_memory_mb"),
            "avg_cpu_percent": data.get("avg_cpu_percent"),
            "max_ram_mb": data.get("max_ram_mb"),
            "gpu_temperature_max": data.get("gpu_temperature_max")
        }
    return None


def read_learning_rate_data(research_dir: Path) -> Optional[Dict[str, Any]]:
    """Read learning rate schedule data"""
    lr_file = research_dir / "learning_rate_log.json"
    data = safe_json_load(lr_file)
    if data:
        lr_values = data.get("learning_rates", [])
        if lr_values:
            return {
                "initial_lr": lr_values[0] if lr_values else None,
                "final_lr": lr_values[-1] if lr_values else None,
                "min_lr": min(lr_values),
                "max_lr": max(lr_values),
                "lr_schedule_type": data.get("schedule_type")
            }
    return None


def read_optimizer_data(research_dir: Path) -> Optional[Dict[str, Any]]:
    """Read optimizer configuration and state"""
    opt_file = research_dir / "optimizer_info.json"
    data = safe_json_load(opt_file)
    if data is None:
        return None
    
    return {
        "optimizer_type": data.get("optimizer_type"),
        "weight_decay": data.get("weight_decay"),
        "momentum": data.get("momentum"),
        "beta1": data.get("beta1"),
        "beta2": data.get("beta2"),
        "eps": data.get("eps")
    }


def analyze_training_convergence(loss_log_path: Path) -> Optional[Dict[str, Any]]:
    """Analyze training convergence from loss log"""
    content = safe_file_read(loss_log_path)
    if content is None:
        return None
    
    try:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) <= 1:
            return None
        
        header = [h.strip().lower() for h in lines[0].split(',')]
        train_losses, valid_losses = [], []
        
        # Extract loss values
        for row in lines[1:]:
            parts = [p.strip() for p in row.split(',')]
            if len(parts) >= len(header):
                try:
                    if "train loss" in header:
                        train_loss = safe_float(parts[header.index("train loss")])
                        if train_loss is not None:
                            train_losses.append(train_loss)
                    if "valid loss" in header:
                        valid_loss = safe_float(parts[header.index("valid loss")])
                        if valid_loss is not None:
                            valid_losses.append(valid_loss)
                except (IndexError, ValueError):
                    continue
        
        if not train_losses or not valid_losses:
            return None
        
        # Calculate convergence metrics
        metrics = {}
        
        # Training stability (variance in last 20% of epochs)
        if len(train_losses) >= 10:
            last_20_percent = int(len(train_losses) * 0.2)
            recent_train = train_losses[-last_20_percent:]
            recent_valid = valid_losses[-last_20_percent:]
            
            if np and len(recent_train) > 1:
                metrics["train_loss_variance_late"] = float(np.var(recent_train))
                metrics["valid_loss_variance_late"] = float(np.var(recent_valid))
            else:
                metrics["train_loss_variance_late"] = statistics.variance(recent_train) if len(recent_train) > 1 else 0
                metrics["valid_loss_variance_late"] = statistics.variance(recent_valid) if len(recent_valid) > 1 else 0
        
        # Overfitting detection (validation loss trend)
        if len(valid_losses) >= 5:
            best_valid_idx = valid_losses.index(min(valid_losses))
            epochs_after_best = len(valid_losses) - best_valid_idx - 1
            metrics["epochs_after_best_valid"] = epochs_after_best
            
            # Check if validation loss increased significantly after best
            if epochs_after_best > 0:
                final_valid = valid_losses[-1]
                best_valid = min(valid_losses)
                metrics["overfitting_ratio"] = final_valid / best_valid if best_valid > 0 else 1.0
        
        # Learning rate (if training loss decreases consistently)
        if len(train_losses) >= 3:
            decreasing_epochs = 0
            for i in range(1, len(train_losses)):
                if train_losses[i] < train_losses[i-1]:
                    decreasing_epochs += 1
            metrics["loss_decreasing_ratio"] = decreasing_epochs / (len(train_losses) - 1)
        
        # Convergence speed (epochs to reach 90% of final improvement)
        if len(train_losses) >= 5:
            initial_loss = train_losses[0]
            final_loss = train_losses[-1]
            target_loss = initial_loss - 0.9 * (initial_loss - final_loss)
            
            convergence_epoch = None
            for i, loss in enumerate(train_losses):
                if loss <= target_loss:
                    convergence_epoch = i + 1
                    break
            metrics["convergence_epoch_90pct"] = convergence_epoch
        
        return metrics
        
    except Exception:
        return None


def read_convergence_metrics(research_dir: Path) -> Optional[Dict[str, Any]]:
    """Read convergence analysis metrics"""
    conv_file = research_dir / "convergence_metrics.json"
    data = safe_json_load(conv_file)
    if data is None:
        return None
    
    return {
        "loss_variance": data.get("loss_variance"),
        "gradient_norm_avg": data.get("gradient_norm_avg"),
        "gradient_norm_max": data.get("gradient_norm_max"),
        "early_stopping_epoch": data.get("early_stopping_epoch"),
        "plateau_epochs": data.get("plateau_epochs")
    }


def read_system_info(research_dir: Path) -> Optional[Dict[str, Any]]:
    """Read system information"""
    sys_file = research_dir / "system_info.json"
    data = safe_json_load(sys_file)
    if data is None:
        return None
    
    return {
        "python_version": data.get("python_version"),
        "torch_version": data.get("torch_version"),
        "cuda_version": data.get("cuda_version"),
        "gpu_name": data.get("gpu_name"),
        "cpu_count": data.get("cpu_count"),
        "total_ram_gb": data.get("total_ram_gb")
    }


def read_research_metrics(research_dir: Path, attn: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "total_duration_seconds": None,
        "final_test_loss_collector": None,
        "model_parameters": None,
        "flops_human": None,
        "params_profile": None,
        "hardware_summary": None,
        "learning_rate_schedule": None,
        "optimizer_info": None,
        "convergence_metrics": None,
        "system_info": None,
    }
    metrics_path = research_dir / f"comprehensive_metrics_{attn}.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8", errors="ignore"))
            exp = data.get("experiment_summary", {})
            out["total_duration_seconds"] = exp.get("total_duration_seconds")
            out["final_test_loss_collector"] = exp.get("final_test_loss")
            mc = data.get("model_complexity", {})
            out["model_parameters"] = mc.get("trainable_parameters") or mc.get("total_parameters")
            out["flops_human"] = mc.get("flops_human")
            out["params_profile"] = mc.get("params_profile")
        except Exception:
            pass
    
    # Read additional monitoring data
    out["hardware_summary"] = read_hardware_summary(research_dir)
    out["learning_rate_schedule"] = read_learning_rate_data(research_dir)
    out["optimizer_info"] = read_optimizer_data(research_dir)
    out["convergence_metrics"] = read_convergence_metrics(research_dir)
    out["system_info"] = read_system_info(research_dir)
    
    return out


# ============== New: Load loss config mapping and failure logs ==============

def load_loss_config_mapping(config_file: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Load loss_configs from a YAML config, map to ids like loss_config_0 -> config dict.
    If yaml is unavailable or file missing/invalid, return empty mapping.
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    if not config_file or not config_file.exists() or yaml is None:
        return mapping
    try:
        data = yaml.safe_load(config_file.read_text(encoding="utf-8", errors="ignore")) or {}
        loss_cfgs = data.get("loss_configs") or []
        if isinstance(loss_cfgs, list):
            for i, cfg in enumerate(loss_cfgs):
                if isinstance(cfg, dict):
                    mapping[f"loss_config_{i}"] = cfg
    except Exception:
        return {}
    return mapping


def annotate_with_loss_config(entries: List[Dict[str, Any]], mapping: Dict[str, Dict[str, Any]]) -> None:
    """Annotate each entry with loss config details if available."""
    if not mapping:
        return
    for e in entries:
        loss_id = e.get("loss_config_id") or e.get("source_group")
        cfg = mapping.get(loss_id)
        if not cfg:
            continue
        e["config_base_weight"] = cfg.get("base_weight")
        e["config_svd_weights"] = cfg.get("svd_weights")
        e["config_topk"] = cfg.get("topk")


def parse_failed_attention_log(loss_root: Path) -> List[Dict[str, Any]]:
    """Parse failed_attention_log.txt at results root produced by main.py.
    Each line expected like: ("attention_type", min_valid_loss)
    """
    out: List[Dict[str, Any]] = []
    log_path = loss_root / "failed_attention_log.txt"
    if not log_path.exists():
        return out
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        attn = None
        min_v = None
        try:
            val = ast.literal_eval(line)
            if isinstance(val, (list, tuple)) and len(val) >= 1:
                attn = val[0]
                if len(val) >= 2:
                    min_v = safe_float(str(val[1]))
        except Exception:
            # fallback: simple split
            m = re.match(r"\(\s*'([^']+)'\s*,\s*([^\)]+)\)", line)
            if m:
                attn = m.group(1)
                min_v = safe_float(m.group(2))
        if attn:
            out.append({
                "attention": attn,
                "source_group": "failed_log",
                "path": str(loss_root),
                "epochs": None,
                "final_train_loss": None,
                "final_valid_loss": None,
                "final_test_loss": None,
                "best_valid_loss": None,
                "best_valid_epoch": None,
                "test_result_file_loss": None,
                "avg_test_loss": None,
                "min_test_loss": None,
                "max_test_loss": None,
                "total_duration_seconds": None,
                "model_parameters": None,
                "flops_human": None,
                "params_profile": None,
                "status": "failed",
                "failed_min_valid_loss": min_v,
            })
    return out

# ===========================================================================


def scan_runs(base_root: Path, dated_root_override: Optional[Path] = None, loss_cfg_root_override: Optional[Path] = None) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    # Candidate roots
    codex_root = base_root / "VIVTransformer-4sh2r1-codex"
    dated_root = Path(dated_root_override) if dated_root_override else codex_root / "modify_multi_attention" / "attention_results"
    loss_cfg_root = Path(loss_cfg_root_override) if loss_cfg_root_override else codex_root / "attention_results"

    # 1) Dated runs under modify_multi_attention/attention_results
    # Support both nested layout: <dated_root>/<datetime>/<attn>
    # and flat layout: <dated_root>/<attn>
    if dated_root.exists():
        children = [p for p in dated_root.iterdir() if p.is_dir()]
        for child in sorted(children):
            # flat layout detection: the child itself looks like an attn_dir
            has_logs = (child / "loss_logs").exists() or (child / "test_results").exists() or any(child.glob("test_result_*.txt"))
            if has_logs:
                attn = child.name
                entry = collect_single_run(child, attn, source_group=f"dated:{dated_root.name}")
                entries.append(entry)
            else:
                # nested layout: iterate its subdirs as attention dirs
                for attn_dir in sorted([p for p in child.iterdir() if p.is_dir()]):
                    attn = attn_dir.name
                    entry = collect_single_run(attn_dir, attn, source_group=f"dated:{child.name}")
                    entries.append(entry)

    # 2) loss_config_* runs under codex/attention_results/loss_config_X/<attn>
    if loss_cfg_root.exists():
        for cfg_dir in sorted([p for p in loss_cfg_root.iterdir() if p.is_dir() and p.name.startswith("loss_config_")]):
            for attn_dir in sorted([p for p in cfg_dir.iterdir() if p.is_dir()]):
                attn = attn_dir.name
                entry = collect_single_run(attn_dir, attn, source_group=cfg_dir.name)
                entry["loss_config_id"] = cfg_dir.name
                entries.append(entry)

    return entries


def collect_single_run(attn_dir: Path, attn: str, source_group: str) -> Dict[str, Any]:
    loss_log = attn_dir / "loss_logs" / "loss_log.txt"
    test_result = None
    # e.g. test_result_cbam.txt
    for f in attn_dir.glob("test_result_*.txt"):
        test_result = parse_test_result_file(f)
        break
    test_loss_log = attn_dir / "test_results" / "test_loss_log.txt"

    loss_stats = parse_loss_log(loss_log)
    test_stats = parse_test_loss_log(test_loss_log)
    
    # Analyze training convergence from loss log
    convergence_analysis = analyze_training_convergence(loss_log)

    # research_data might be under attn_dir/research_data
    research_dir = attn_dir / "research_data"
    research_stats = read_research_metrics(research_dir, attn)
    
    # Flatten nested dictionaries for CSV output
    flattened_data = {}
    
    # Hardware summary
    if research_stats.get("hardware_summary"):
        hw = research_stats["hardware_summary"]
        flattened_data.update({
            "avg_gpu_utilization": hw.get("avg_gpu_utilization"),
            "max_gpu_memory_mb": hw.get("max_gpu_memory_mb"),
            "avg_cpu_percent": hw.get("avg_cpu_percent"),
            "max_ram_mb": hw.get("max_ram_mb"),
            "gpu_temperature_max": hw.get("gpu_temperature_max")
        })
    
    # Learning rate schedule
    if research_stats.get("learning_rate_schedule"):
        lr = research_stats["learning_rate_schedule"]
        flattened_data.update({
            "initial_lr": lr.get("initial_lr"),
            "final_lr": lr.get("final_lr"),
            "min_lr": lr.get("min_lr"),
            "max_lr": lr.get("max_lr"),
            "lr_schedule_type": lr.get("lr_schedule_type")
        })
    
    # Optimizer info
    if research_stats.get("optimizer_info"):
        opt = research_stats["optimizer_info"]
        flattened_data.update({
            "optimizer_type": opt.get("optimizer_type"),
            "weight_decay": opt.get("weight_decay"),
            "momentum": opt.get("momentum"),
            "beta1": opt.get("beta1"),
            "beta2": opt.get("beta2"),
            "eps": opt.get("eps")
        })
    
    # Convergence metrics
    if research_stats.get("convergence_metrics"):
        conv = research_stats["convergence_metrics"]
        flattened_data.update({
            "loss_variance": conv.get("loss_variance"),
            "gradient_norm_avg": conv.get("gradient_norm_avg"),
            "gradient_norm_max": conv.get("gradient_norm_max"),
            "early_stopping_epoch": conv.get("early_stopping_epoch"),
            "plateau_epochs": conv.get("plateau_epochs")
        })
    
    # Training convergence analysis
    if convergence_analysis:
        flattened_data.update({
            "train_loss_variance_late": convergence_analysis.get("train_loss_variance_late"),
            "valid_loss_variance_late": convergence_analysis.get("valid_loss_variance_late"),
            "epochs_after_best_valid": convergence_analysis.get("epochs_after_best_valid"),
            "overfitting_ratio": convergence_analysis.get("overfitting_ratio"),
            "loss_decreasing_ratio": convergence_analysis.get("loss_decreasing_ratio"),
            "convergence_epoch_90pct": convergence_analysis.get("convergence_epoch_90pct")
        })
    
    # System info
    if research_stats.get("system_info"):
        sys_info = research_stats["system_info"]
        flattened_data.update({
            "python_version": sys_info.get("python_version"),
            "torch_version": sys_info.get("torch_version"),
            "cuda_version": sys_info.get("cuda_version"),
            "gpu_name": sys_info.get("gpu_name"),
            "cpu_count": sys_info.get("cpu_count"),
            "total_ram_gb": sys_info.get("total_ram_gb")
        })

    # Combine all collected data
    collected_data = {
        "attention": attn,
        "source_group": source_group,
        "path": str(attn_dir),
        **loss_stats,
        "test_result_file_loss": test_result,
        **test_stats,
        **research_stats,
        **flattened_data,
        "status": "ok",
    }
    
    # Validate and clean the collected data
    validated_data = validate_experiment_data(collected_data)
    
    return validated_data


def write_csv_summary(entries: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "attention", "source_group", "loss_config_id", "status", "path", "epochs",
        "final_train_loss", "final_valid_loss", "final_test_loss",
        "best_valid_loss", "best_valid_epoch",
        "test_result_file_loss", "avg_test_loss", "min_test_loss", "max_test_loss",
        "total_duration_seconds", "model_parameters", "flops_human", "params_profile",
        # Hardware monitoring fields
        "avg_gpu_utilization", "max_gpu_memory_mb", "avg_cpu_percent", "max_ram_mb", "gpu_temperature_max",
        # Learning rate fields
        "initial_lr", "final_lr", "min_lr", "max_lr", "lr_schedule_type",
        # Optimizer fields
        "optimizer_type", "weight_decay", "momentum", "beta1", "beta2", "eps",
        # Convergence fields
        "loss_variance", "gradient_norm_avg", "gradient_norm_max", "early_stopping_epoch", "plateau_epochs",
        # Training stability fields
        "train_loss_variance_late", "valid_loss_variance_late", "epochs_after_best_valid", "overfitting_ratio",
        "loss_decreasing_ratio", "convergence_epoch_90pct",
        # System info fields
        "python_version", "torch_version", "cuda_version", "gpu_name", "cpu_count", "total_ram_gb",
        "config_base_weight", "config_svd_weights", "config_topk", "failed_min_valid_loss",
    ]
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in entries:
            writer.writerow({k: e.get(k) for k in fields})


def best_by_attention(entries: List[Dict[str, Any]], key_order: List[str]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        grouped.setdefault(e["attention"], []).append(e)

    best: Dict[str, Dict[str, Any]] = {}
    for attn, arr in grouped.items():
        def score(x: Dict[str, Any]) -> Tuple:
            vals = []
            for k in key_order:
                v = x.get(k)
                if v is None:
                    vals.append(float('inf'))
                else:
                    vals.append(float(v))
            # Also prefer more epochs when scores tie
            vals.append(-int(x.get("epochs") or 0))
            return tuple(vals)
        best[attn] = sorted(arr, key=score)[0]
    return best


def calculate_performance_statistics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive performance statistics"""
    stats = {
        "total_experiments": len(entries),
        "successful_experiments": len([e for e in entries if e.get("status") == "ok"]),
        "failed_experiments": len([e for e in entries if e.get("status") == "failed"]),
        "attention_types": len(set(e.get("attention", "unknown") for e in entries)),
        "avg_training_time": None,
        "avg_model_parameters": None,
        "convergence_success_rate": None,
        "overfitting_rate": None
    }
    
    valid_entries = [e for e in entries if e.get("status") == "ok"]
    if not valid_entries:
        return stats
    
    # Training time statistics
    training_times = [e.get("total_duration_seconds") for e in valid_entries if e.get("total_duration_seconds") is not None]
    if training_times:
        stats["avg_training_time"] = sum(training_times) / len(training_times)
        stats["min_training_time"] = min(training_times)
        stats["max_training_time"] = max(training_times)
    
    # Model complexity statistics
    model_params = [e.get("model_parameters") for e in valid_entries if e.get("model_parameters") is not None]
    if model_params:
        # Convert string representations to numbers if needed
        numeric_params = []
        for p in model_params:
            if isinstance(p, str):
                try:
                    # Handle formats like "1.2M" or "1,234,567"
                    if 'M' in p:
                        numeric_params.append(float(p.replace('M', '').replace(',', '')) * 1e6)
                    elif 'K' in p:
                        numeric_params.append(float(p.replace('K', '').replace(',', '')) * 1e3)
                    else:
                        numeric_params.append(float(p.replace(',', '')))
                except:
                    continue
            elif isinstance(p, (int, float)):
                numeric_params.append(float(p))
        
        if numeric_params:
            stats["avg_model_parameters"] = sum(numeric_params) / len(numeric_params)
            stats["min_model_parameters"] = min(numeric_params)
            stats["max_model_parameters"] = max(numeric_params)
    
    # Convergence analysis
    convergence_ratios = [e.get("loss_decreasing_ratio") for e in valid_entries if e.get("loss_decreasing_ratio") is not None]
    if convergence_ratios:
        # Consider convergence successful if >80% of epochs showed decreasing loss
        successful_convergence = len([r for r in convergence_ratios if r > 0.8])
        stats["convergence_success_rate"] = successful_convergence / len(convergence_ratios)
    
    # Overfitting analysis
    overfitting_ratios = [e.get("overfitting_ratio") for e in valid_entries if e.get("overfitting_ratio") is not None]
    if overfitting_ratios:
        # Consider overfitting if validation loss increased by >10% from best
        overfitting_cases = len([r for r in overfitting_ratios if r > 1.1])
        stats["overfitting_rate"] = overfitting_cases / len(overfitting_ratios)
    
    return stats


def verify_reproducibility(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify experiment reproducibility by analyzing consistency across runs"""
    reproducibility = {
        "attention_consistency": {},
        "config_consistency": {},
        "environment_consistency": {},
        "seed_consistency": {},
        "dataset_consistency": {},
        "overall_score": 0.0,
        "issues": [],
        "recommendations": []
    }
    
    # Group by attention type and configuration
    attention_groups = {}
    for e in entries:
        if e.get("status") != "ok":
            continue
        
        attn = e.get("attention", "unknown")
        config_id = e.get("loss_config_id", "default")
        key = f"{attn}_{config_id}"
        
        if key not in attention_groups:
            attention_groups[key] = []
        attention_groups[key].append(e)
    
    # Analyze consistency within each group
    total_groups = 0
    consistent_groups = 0
    
    for group_key, runs in attention_groups.items():
        if len(runs) < 2:
            continue  # Need at least 2 runs for comparison
        
        total_groups += 1
        attn_name = group_key.split('_')[0]
        
        # Performance consistency
        test_losses = [r.get("avg_test_loss") for r in runs if r.get("avg_test_loss") is not None]
        if len(test_losses) >= 2:
            cv = statistics.stdev(test_losses) / (sum(test_losses) / len(test_losses))  # Coefficient of variation
            
            reproducibility["attention_consistency"][attn_name] = {
                "runs": len(test_losses),
                "mean_loss": sum(test_losses) / len(test_losses),
                "std_loss": statistics.stdev(test_losses),
                "cv": cv,
                "is_consistent": cv < 0.1  # CV < 10% considered consistent
            }
            
            if cv < 0.1:
                consistent_groups += 1
        
        # Configuration consistency
        configs = {}
        for run in runs:
            for key in ['learning_rate_initial', 'optimizer_type', 'model_parameters']:
                if run.get(key) is not None:
                    if key not in configs:
                        configs[key] = []
                    configs[key].append(run.get(key))
        
        config_consistent = True
        for key, values in configs.items():
            if len(set(str(v) for v in values)) > 1:  # More than one unique value
                config_consistent = False
                reproducibility["issues"].append(f"{attn_name}: 配置不一致 - {key}")
        
        reproducibility["config_consistency"][attn_name] = config_consistent
        
        # Environment consistency
        env_keys = ['pytorch_version', 'cuda_version', 'gpu_name']
        env_consistent = True
        for key in env_keys:
            values = [r.get(key) for r in runs if r.get(key) is not None]
            if len(set(values)) > 1:
                env_consistent = False
                reproducibility["issues"].append(f"{attn_name}: 环境不一致 - {key}")
        
        reproducibility["environment_consistency"][attn_name] = env_consistent
        
        # Random seed consistency check
        seed_values = [r.get("random_seed") for r in runs if r.get("random_seed") is not None]
        seed_consistent = True
        if len(seed_values) > 1:
            unique_seeds = set(seed_values)
            if len(unique_seeds) == 1:
                # All runs use the same seed - good for reproducibility verification
                reproducibility["seed_consistency"][attn_name] = {
                    "status": "identical",
                    "seed": list(unique_seeds)[0],
                    "runs": len(seed_values)
                }
            elif len(unique_seeds) == len(seed_values):
                # All runs use different seeds - good for robustness testing
                reproducibility["seed_consistency"][attn_name] = {
                    "status": "diverse",
                    "unique_seeds": len(unique_seeds),
                    "runs": len(seed_values)
                }
            else:
                # Mixed seed usage - potentially problematic
                seed_consistent = False
                reproducibility["issues"].append(f"{attn_name}: 随机种子使用不一致")
                reproducibility["seed_consistency"][attn_name] = {
                    "status": "inconsistent",
                    "unique_seeds": len(unique_seeds),
                    "runs": len(seed_values)
                }
        
        # Dataset consistency check
        dataset_keys = ['dataset_version', 'data_split_seed', 'preprocessing_version']
        dataset_consistent = True
        dataset_info = {}
        for key in dataset_keys:
            values = [r.get(key) for r in runs if r.get(key) is not None]
            if values:
                unique_values = set(str(v) for v in values)
                dataset_info[key] = {
                    "unique_count": len(unique_values),
                    "total_runs": len(values)
                }
                if len(unique_values) > 1:
                    dataset_consistent = False
                    reproducibility["issues"].append(f"{attn_name}: 数据集不一致 - {key}")
        
        reproducibility["dataset_consistency"][attn_name] = {
            "consistent": dataset_consistent,
            "details": dataset_info
        }
    
    # Calculate overall reproducibility score
    if total_groups > 0:
        # Weight different aspects of reproducibility
        performance_score = consistent_groups / total_groups
        config_score = sum(1 for v in reproducibility["config_consistency"].values() if v) / max(len(reproducibility["config_consistency"]), 1)
        env_score = sum(1 for v in reproducibility["environment_consistency"].values() if v) / max(len(reproducibility["environment_consistency"]), 1)
        
        # Overall score is weighted average
        reproducibility["overall_score"] = (performance_score * 0.5 + config_score * 0.3 + env_score * 0.2)
    
    # Generate recommendations based on analysis
    _generate_reproducibility_recommendations(reproducibility)
    
    # Add general issues
    if reproducibility["overall_score"] < 0.7:
        reproducibility["issues"].append("整体可复现性较低，建议检查实验设置")
    
    return reproducibility


def _generate_reproducibility_recommendations(reproducibility: Dict[str, Any]) -> None:
    """Generate specific recommendations based on reproducibility analysis"""
    recommendations = reproducibility["recommendations"]
    
    # Performance consistency recommendations
    inconsistent_attentions = []
    for attn, consistency in reproducibility.get("attention_consistency", {}).items():
        if not consistency.get("is_consistent", True):
            inconsistent_attentions.append(attn)
    
    if inconsistent_attentions:
        recommendations.append(f"性能不一致的注意力机制 ({', '.join(inconsistent_attentions)}): 增加实验重复次数或检查超参数设置")
    
    # Configuration consistency recommendations
    config_issues = [attn for attn, consistent in reproducibility.get("config_consistency", {}).items() if not consistent]
    if config_issues:
        recommendations.append(f"配置不一致的注意力机制 ({', '.join(config_issues)}): 统一实验配置文件和超参数")
    
    # Environment consistency recommendations
    env_issues = [attn for attn, consistent in reproducibility.get("environment_consistency", {}).items() if not consistent]
    if env_issues:
        recommendations.append(f"环境不一致的注意力机制 ({', '.join(env_issues)}): 使用容器化环境或统一软件版本")
    
    # Seed consistency recommendations
    for attn, seed_info in reproducibility.get("seed_consistency", {}).items():
        if seed_info.get("status") == "inconsistent":
            recommendations.append(f"{attn}: 随机种子使用不规范，建议统一使用相同种子或完全不同种子")
        elif seed_info.get("status") == "identical" and seed_info.get("runs", 0) < 3:
            recommendations.append(f"{attn}: 建议增加更多重复实验以验证可复现性")
    
    # Dataset consistency recommendations
    for attn, dataset_info in reproducibility.get("dataset_consistency", {}).items():
        if not dataset_info.get("consistent", True):
            recommendations.append(f"{attn}: 数据集版本不一致，建议固定数据集版本和预处理流程")
    
    # Overall recommendations
    if reproducibility["overall_score"] >= 0.8:
        recommendations.append("✓ 实验具有良好的可复现性，建议继续保持当前实验规范")
    elif reproducibility["overall_score"] >= 0.6:
        recommendations.append("⚠ 实验可复现性中等，重点关注配置和环境一致性")
    else:
        recommendations.append("✗ 实验可复现性较差，建议全面检查实验流程和环境设置")
        recommendations.append("建议使用实验管理工具(如MLflow, Weights&Biases)来跟踪实验")
        recommendations.append("建议建立标准化的实验执行流程和检查清单")


def perform_statistical_analysis(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform statistical significance tests between attention mechanisms"""
    analysis = {"comparisons": [], "best_attention": None, "statistical_notes": []}
    
    # Group by attention type
    attention_groups = {}
    for e in entries:
        if e.get("status") != "ok":
            continue
        attn = e.get("attention", "unknown")
        if attn not in attention_groups:
            attention_groups[attn] = []
        attention_groups[attn].append(e)
    
    # Calculate mean performance for each attention type
    attention_means = {}
    for attn, runs in attention_groups.items():
        test_losses = [r.get("avg_test_loss") for r in runs if r.get("avg_test_loss") is not None]
        if test_losses:
            attention_means[attn] = {
                "mean_test_loss": sum(test_losses) / len(test_losses),
                "std_test_loss": statistics.stdev(test_losses) if len(test_losses) > 1 else 0,
                "sample_size": len(test_losses),
                "min_test_loss": min(test_losses),
                "max_test_loss": max(test_losses)
            }
    
    # Find best performing attention mechanism
    if attention_means:
        best_attn = min(attention_means.keys(), key=lambda k: attention_means[k]["mean_test_loss"])
        analysis["best_attention"] = {
            "name": best_attn,
            "mean_test_loss": attention_means[best_attn]["mean_test_loss"],
            "std_test_loss": attention_means[best_attn]["std_test_loss"],
            "sample_size": attention_means[best_attn]["sample_size"]
        }
    
    # Pairwise comparisons (simplified without scipy)
    attention_list = list(attention_means.keys())
    for i, attn1 in enumerate(attention_list):
        for attn2 in attention_list[i+1:]:
            mean1 = attention_means[attn1]["mean_test_loss"]
            mean2 = attention_means[attn2]["mean_test_loss"]
            std1 = attention_means[attn1]["std_test_loss"]
            std2 = attention_means[attn2]["std_test_loss"]
            n1 = attention_means[attn1]["sample_size"]
            n2 = attention_means[attn2]["sample_size"]
            
            # Simple effect size calculation (Cohen's d approximation)
            if std1 > 0 or std2 > 0:
                pooled_std = ((std1**2 + std2**2) / 2) ** 0.5
                effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            else:
                effect_size = 0
            
            comparison = {
                "attention1": attn1,
                "attention2": attn2,
                "mean_diff": mean1 - mean2,
                "effect_size": effect_size,
                "better": attn1 if mean1 < mean2 else attn2,
                "confidence": "high" if effect_size > 0.8 else "medium" if effect_size > 0.5 else "low"
            }
            analysis["comparisons"].append(comparison)
    
    # Add statistical notes
    if len(attention_groups) < 2:
        analysis["statistical_notes"].append("需要至少2种注意力机制进行统计比较")
    
    total_samples = sum(len(runs) for runs in attention_groups.values())
    if total_samples < 10:
        analysis["statistical_notes"].append("样本量较小，统计结果可能不够可靠")
    
    return analysis


def generate_markdown(entries: List[Dict[str, Any]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate comprehensive statistics
    perf_stats = calculate_performance_statistics(entries)
    stat_analysis = perform_statistical_analysis(entries)
    reproducibility = verify_reproducibility(entries)

    # Prefer ranking by avg_test_loss, then final_test_loss, then best_valid_loss
    # Exclude failed entries from ranking
    valid_entries = [e for e in entries if e.get("status") != "failed"]
    if not valid_entries:
        valid_entries = entries
    best_map = best_by_attention(valid_entries, key_order=["avg_test_loss", "final_test_loss", "best_valid_loss"])

    # Build a sorted list of best entries
    best_list = list(best_map.values())
    def sort_key(e: Dict[str, Any]):
        return (
            float(e.get("avg_test_loss") or e.get("final_test_loss") or 1e9),
            float(e.get("final_test_loss") or 1e9),
        )
    best_list.sort(key=sort_key)

    lines: List[str] = []
    lines.append("# 注意力机制对比分析报告\n")
    lines.append("本报告聚合了批量实验的训练与测试日志，按注意力机制汇总最佳结果并进行横向对比。\n")
    
    # Performance overview section
    lines.append("## 实验概览\n")
    lines.append(f"- **总实验数**: {perf_stats['total_experiments']}\n")
    lines.append(f"- **成功实验**: {perf_stats['successful_experiments']}\n")
    lines.append(f"- **失败实验**: {perf_stats['failed_experiments']}\n")
    lines.append(f"- **注意力机制类型**: {perf_stats['attention_types']}\n")
    
    if perf_stats.get('avg_training_time'):
        lines.append(f"- **平均训练时间**: {perf_stats['avg_training_time']:.1f}秒\n")
    if perf_stats.get('convergence_success_rate'):
        lines.append(f"- **收敛成功率**: {perf_stats['convergence_success_rate']:.1%}\n")
    if perf_stats.get('overfitting_rate'):
        lines.append(f"- **过拟合率**: {perf_stats['overfitting_rate']:.1%}\n")
    
    # Statistical analysis section
    if stat_analysis.get('best_attention'):
        best_attn = stat_analysis['best_attention']
        lines.append("\n## 统计分析结果\n")
        lines.append(f"**最佳注意力机制**: {best_attn['name']}\n")
        lines.append(f"- 平均测试损失: {fmt(best_attn['mean_test_loss'])}\n")
        lines.append(f"- 标准差: {fmt(best_attn['std_test_loss'])}\n")
        lines.append(f"- 样本数: {best_attn['sample_size']}\n")
        
        # Top comparisons
        if stat_analysis.get('comparisons'):
            lines.append("\n### 主要对比结果\n")
            lines.append("| 注意力1 | 注意力2 | 性能差异 | 效应大小 | 更优者 | 置信度 |\n")
            lines.append("|:--|:--|--:|--:|:--|:--|\n")
            
            # Sort by effect size and show top 5
            top_comparisons = sorted(stat_analysis['comparisons'], 
                                   key=lambda x: x['effect_size'], reverse=True)[:5]
            for comp in top_comparisons:
                lines.append(f"| {comp['attention1']} | {comp['attention2']} | {fmt(comp['mean_diff'])} | {fmt(comp['effect_size'])} | {comp['better']} | {comp['confidence']} |\n")
    
    if stat_analysis.get('statistical_notes'):
        lines.append("\n### 统计说明\n")
        for note in stat_analysis['statistical_notes']:
            lines.append(f"- {note}\n")
    
    # Reproducibility analysis section
    lines.append("\n## 可复现性分析\n")
    lines.append(f"**整体可复现性评分**: {reproducibility['overall_score']:.1%}\n")
    
    if reproducibility.get('attention_consistency'):
        lines.append("\n### 性能一致性分析\n")
        lines.append("| 注意力机制 | 实验次数 | 平均损失 | 标准差 | 变异系数 | 一致性 |\n")
        lines.append("|:--|--:|--:|--:|--:|:--|\n")
        
        for attn, consistency in reproducibility['attention_consistency'].items():
            status = "✓ 一致" if consistency['is_consistent'] else "✗ 不一致"
            lines.append(f"| {attn} | {consistency['runs']} | {fmt(consistency['mean_loss'])} | {fmt(consistency['std_loss'])} | {consistency['cv']:.3f} | {status} |\n")
    
    # Random seed consistency analysis
    if reproducibility.get('seed_consistency'):
        lines.append("\n### 随机种子一致性\n")
        lines.append("| 注意力机制 | 种子策略 | 实验次数 | 唯一种子数 | 状态 |\n")
        lines.append("|:--|:--|--:|--:|:--|\n")
        
        for attn, seed_info in reproducibility['seed_consistency'].items():
            status_map = {
                "identical": "✓ 相同种子",
                "diverse": "✓ 不同种子", 
                "inconsistent": "✗ 不规范"
            }
            status = status_map.get(seed_info.get('status'), '未知')
            unique_seeds = seed_info.get('unique_seeds', seed_info.get('seed', 'N/A'))
            lines.append(f"| {attn} | {seed_info.get('status', 'unknown')} | {seed_info.get('runs', 0)} | {unique_seeds} | {status} |\n")
    
    # Dataset consistency analysis
    if reproducibility.get('dataset_consistency'):
        lines.append("\n### 数据集一致性\n")
        dataset_consistent_count = sum(1 for info in reproducibility['dataset_consistency'].values() if info.get('consistent', True))
        dataset_total = len(reproducibility['dataset_consistency'])
        lines.append(f"**数据集一致性**: {dataset_consistent_count}/{dataset_total} 注意力机制通过检查\n")
        
        inconsistent_datasets = [attn for attn, info in reproducibility['dataset_consistency'].items() if not info.get('consistent', True)]
        if inconsistent_datasets:
            lines.append(f"**数据集不一致的注意力机制**: {', '.join(inconsistent_datasets)}\n")
    
    # Configuration and environment summary
    if reproducibility.get('config_consistency') or reproducibility.get('environment_consistency'):
        lines.append("\n### 配置与环境一致性\n")
        
        if reproducibility.get('config_consistency'):
            config_consistent_count = sum(1 for consistent in reproducibility['config_consistency'].values() if consistent)
            config_total = len(reproducibility['config_consistency'])
            lines.append(f"- **配置一致性**: {config_consistent_count}/{config_total} 注意力机制\n")
        
        if reproducibility.get('environment_consistency'):
            env_consistent_count = sum(1 for consistent in reproducibility['environment_consistency'].values() if consistent)
            env_total = len(reproducibility['environment_consistency'])
            lines.append(f"- **环境一致性**: {env_consistent_count}/{env_total} 注意力机制\n")
    
    if reproducibility.get('issues'):
        lines.append("\n### 发现的问题\n")
        for issue in reproducibility['issues']:
            lines.append(f"- {issue}\n")
    
    # Enhanced recommendations
    if reproducibility.get('recommendations'):
        lines.append("\n### 改进建议\n")
        for recommendation in reproducibility['recommendations']:
            lines.append(f"- {recommendation}\n")

    # Top-5 summary
    lines.append("## 总体排名（Top-5，按测试损失）\n")
    lines.append("| 排名 | 注意力 | 平均测试损失 | 最终测试损失 | 最佳验证损失 | 收敛指标 | 轮数 | 源目录 | Loss配置 |\n")
    lines.append("|---:|:--|--:|--:|--:|:--|--:|:--|:--|\n")
    for i, e in enumerate(best_list[:5], 1):
        convergence_info = ""
        if e.get('loss_decreasing_ratio') is not None:
            convergence_info += f"收敛率:{e.get('loss_decreasing_ratio'):.2f}"
        if e.get('overfitting_ratio') is not None:
            if convergence_info:
                convergence_info += ", "
            convergence_info += f"过拟合:{e.get('overfitting_ratio'):.2f}"
        if not convergence_info:
            convergence_info = "N/A"
        lines.append(
            f"| {i} | {e['attention']} | {fmt(e.get('avg_test_loss'))} | {fmt(e.get('final_test_loss'))} | {fmt(e.get('best_valid_loss'))} | {convergence_info} | {e.get('epochs') or '-'} | {e.get('source_group')} | {e.get('loss_config_id') or '-'} |\n"
        )

    # Detailed table
    lines.append("\n## 详细对比（每类注意力的最佳一次）\n")
    lines.append("| 注意力 | 平均测试损失 | 最终测试损失 | 最佳验证损失 | 轮数 | 训练时长(s) | 参数量 | FLOPs | 硬件利用率 | 源目录 | Loss配置 | base_weight | topk |\n")
    lines.append("|:--|--:|--:|--:|--:|--:|--:|:--|:--|:--|:--|--:|--:|\n")
    for e in best_list:
        # Hardware utilization summary
        hw_info = ""
        if e.get('avg_gpu_utilization') is not None:
            hw_info += f"GPU:{e.get('avg_gpu_utilization'):.1f}%"
        if e.get('max_gpu_memory_mb') is not None:
            if hw_info:
                hw_info += ", "
            hw_info += f"Mem:{e.get('max_gpu_memory_mb'):.0f}MB"
        if not hw_info:
            hw_info = "N/A"
        
        lines.append(
            f"| {e['attention']} | {fmt(e.get('avg_test_loss'))} | {fmt(e.get('final_test_loss'))} | {fmt(e.get('best_valid_loss'))} | {e.get('epochs') or '-'} | {fmt(e.get('total_duration_seconds'))} | {e.get('model_parameters') or '-'} | {e.get('flops_human') or '-'} | {hw_info} | {e.get('source_group')} | {e.get('loss_config_id') or '-'} | {fmt(e.get('config_base_weight'))} | {e.get('config_topk') or '-'} |\n"
        )

    # Failures
    failures = [e for e in entries if e.get("status") == "failed"]
    if failures:
        lines.append("\n## 失败实验（来自 failed_attention_log.txt）\n")
        lines.append("| 注意力 | 最小验证损失 | 失败原因分析 | 来源 |\n")
        lines.append("|:--|--:|:--|:--|\n")
        for e in failures:
            # Try to infer failure reason from available data
            failure_reason = "未知"
            if e.get('max_gpu_memory_mb') and e.get('max_gpu_memory_mb') > 10000:  # >10GB
                failure_reason = "GPU内存不足"
            elif e.get('total_duration_seconds') and e.get('total_duration_seconds') < 60:
                failure_reason = "快速失败(可能配置错误)"
            elif e.get('epochs_after_best_valid') and e.get('epochs_after_best_valid') > 50:
                failure_reason = "训练发散"
            
            lines.append(f"| {e['attention']} | {fmt(e.get('failed_min_valid_loss'))} | {failure_reason} | {e.get('source_group')} |\n")

    # Performance insights section
    lines.append("\n## 性能洞察\n")
    
    # Training efficiency analysis
    if perf_stats.get('avg_training_time') and perf_stats.get('avg_model_parameters'):
        lines.append("### 训练效率分析\n")
        lines.append(f"- 平均训练时间: {perf_stats['avg_training_time']:.1f}秒\n")
        if perf_stats.get('min_training_time') and perf_stats.get('max_training_time'):
            lines.append(f"- 训练时间范围: {perf_stats['min_training_time']:.1f}s - {perf_stats['max_training_time']:.1f}s\n")
        lines.append(f"- 平均模型参数量: {fmt(perf_stats['avg_model_parameters'])}\n")
    
    # Convergence analysis
    if perf_stats.get('convergence_success_rate') is not None:
        lines.append("\n### 收敛性分析\n")
        lines.append(f"- 收敛成功率: {perf_stats['convergence_success_rate']:.1%}\n")
        if perf_stats.get('overfitting_rate') is not None:
            lines.append(f"- 过拟合发生率: {perf_stats['overfitting_rate']:.1%}\n")
    
    # Hardware utilization insights
    hw_entries = [e for e in valid_entries if e.get('avg_gpu_utilization') is not None]
    if hw_entries:
        avg_gpu_util = sum(e.get('avg_gpu_utilization', 0) for e in hw_entries) / len(hw_entries)
        lines.append("\n### 硬件利用率分析\n")
        lines.append(f"- 平均GPU利用率: {avg_gpu_util:.1f}%\n")
        
        high_util_count = len([e for e in hw_entries if e.get('avg_gpu_utilization', 0) > 80])
        lines.append(f"- 高GPU利用率实验(>80%): {high_util_count}/{len(hw_entries)}\n")
        
        memory_entries = [e for e in hw_entries if e.get('max_gpu_memory_mb') is not None]
        if memory_entries:
            avg_memory = sum(e.get('max_gpu_memory_mb', 0) for e in memory_entries) / len(memory_entries)
            lines.append(f"- 平均GPU内存使用: {avg_memory:.0f}MB\n")
    
    # Notes on data sources
    lines.append("\n## 数据来源与解析说明\n")
    lines.append("- loss_logs/loss_log.txt：解析最终/最佳损失与轮数。\n")
    lines.append("- test_results/test_loss_log.txt：解析批次级测试损失与平均值。\n")
    lines.append("- test_result_*.txt：解析一次性汇报的测试损失。\n")
    lines.append("- research_data/comprehensive_metrics_*.json：若存在，补充训练时长、模型复杂度与FLOPs。\n")
    lines.append("- failed_attention_log.txt：若存在，列出训练失败的注意力机制（来自训练脚本 main.py）。\n")
    lines.append("\n### 统计方法说明\n")
    lines.append("- **效应大小**: 使用Cohen's d近似计算，>0.8为大效应，>0.5为中等效应\n")
    lines.append("- **收敛成功**: 定义为>80%的训练轮次显示损失下降\n")
    lines.append("- **过拟合检测**: 验证损失相比最佳值增长>10%\n")
    lines.append("- **硬件利用率**: 基于训练过程中的GPU和内存监控数据\n")
    lines.append("- **失败原因推断**: 基于训练时长、内存使用和收敛模式的启发式分析\n")

    out_md.write_text("".join(lines), encoding="utf-8")


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    try:
        x = float(v)
        if x >= 0.01:
            return f"{x:.6f}"
        else:
            return f"{x:.2e}"
    except Exception:
        return str(v)


def main():
    # repo root: .../Sparse_to_Dense_Transformer
    script_path = Path(__file__).resolve()
    default_base_root = script_path.parents[2]

    parser = argparse.ArgumentParser(description="Analyze attention experiment results and generate reports")
    parser.add_argument("--base-root", type=str, default=str(default_base_root), help="Repository root directory (default: inferred)")
    parser.add_argument("--dated-root", type=str, default=None, help="Override dated results root (e.g., modify_multi_attention/attention_results1)")
    parser.add_argument("--loss-root", type=str, default=None, help="Override loss_config_* results root")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to write outputs; default: <base-root>/report")
    # New options to integrate outputs with config and failures
    parser.add_argument("--config-file", type=str, default=None, help="Path to YAML config (e.g., config_server.yaml) to annotate loss_config details")
    parser.add_argument("--include-failures", action="store_true", help="Include failed attention entries from failed_attention_log.txt if present under loss-root")
    args = parser.parse_args()

    base_root = Path(args.base_root).resolve()
    dated_root_override = Path(args.dated_root).resolve() if args.dated_root else None
    loss_cfg_root_override = Path(args.loss_root).resolve() if args.loss_root else None

    entries = scan_runs(base_root, dated_root_override=dated_root_override, loss_cfg_root_override=loss_cfg_root_override)

    # Optionally include failures from failed_attention_log.txt
    if args.include_failures and loss_cfg_root_override and (loss_cfg_root_override / "failed_attention_log.txt").exists():
        failures = parse_failed_attention_log(loss_cfg_root_override)
        entries.extend(failures)

    # Optionally annotate entries with loss config details
    cfg_mapping = load_loss_config_mapping(Path(args.config_file).resolve()) if args.config_file else {}
    if cfg_mapping:
        annotate_with_loss_config(entries, cfg_mapping)

    report_dir = Path(args.output_dir).resolve() if args.output_dir else base_root / "report"
    out_csv = report_dir / "attention_summary.csv"
    out_md = report_dir / "attention_report.md"

    write_csv_summary(entries, out_csv)
    generate_markdown(entries, out_md)

    print(f"Scanning base_root: {base_root}")
    if dated_root_override:
        print(f"Using dated_root override: {dated_root_override}")
    if loss_cfg_root_override:
        print(f"Using loss_cfg_root override: {loss_cfg_root_override}")
    if args.config_file:
        print(f"Using config_file: {Path(args.config_file).resolve()}")
    print(f"Summary CSV: {out_csv}")
    print(f"Markdown Report: {out_md}")
    print(f"Total entries: {len(entries)}")


if __name__ == "__main__":
    main()