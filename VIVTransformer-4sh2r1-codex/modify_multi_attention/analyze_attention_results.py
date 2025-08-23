import os
import re
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import argparse
import ast

try:
    import yaml  # Optional
except Exception:
    yaml = None


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        try:
            # scientific notation like 7.9e-05
            return float(x.strip())
        except Exception:
            return None


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


def read_research_metrics(research_dir: Path, attn: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "total_duration_seconds": None,
        "final_test_loss_collector": None,
        "model_parameters": None,
        "flops_human": None,
        "params_profile": None,
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


# ============== New: Parse per-run main_config.yaml and gather artifacts ==============

def _dig(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def parse_main_config(main_cfg_path: Path) -> Dict[str, Any]:
    """Parse research_data/configs/main_config.yaml if exists (best-effort, schema-agnostic)."""
    out: Dict[str, Any] = {
        "run_config_path": None,
        "run_dataset": None,
        "run_epochs": None,
        "run_batch_size": None,
        "run_lr": None,
        "run_optimizer": None,
        "run_scheduler": None,
        "run_model": None,
        "run_device": None,
        "run_seed": None,
        "run_loss_base_weight": None,
        "run_loss_svd_weights": None,
        "run_loss_topk": None,
    }
    if yaml is None or not main_cfg_path.exists():
        return out
    try:
        data = yaml.safe_load(main_cfg_path.read_text(encoding="utf-8", errors="ignore")) or {}
        out["run_config_path"] = str(main_cfg_path)
        # dataset
        out["run_dataset"] = (
            _dig(data, ["dataset"]) or _dig(data, ["data", "dataset"]) or _dig(data, ["data", "name"]) or _dig(data, ["dataset_name"]) or _dig(data, ["data", "dataset_name"])
        )
        # epochs
        out["run_epochs"] = _dig(data, ["epochs"]) or _dig(data, ["train", "epochs"]) or _dig(data, ["training", "epochs"]) or _dig(data, ["trainer", "epochs"])
        # batch_size
        out["run_batch_size"] = (
            _dig(data, ["batch_size"]) or _dig(data, ["data", "batch_size"]) or _dig(data, ["train", "batch_size"]) or _dig(data, ["training", "batch_size"]) or _dig(data, ["loader", "batch_size"])
        )
        # learning rate
        out["run_lr"] = (
            _dig(data, ["lr"]) or _dig(data, ["learning_rate"]) or _dig(data, ["optimizer", "lr"]) or _dig(data, ["optim", "lr"]) or _dig(data, ["train", "lr"]) or _dig(data, ["training", "lr"]) or _dig(data, ["trainer", "lr"]) 
        )
        # optimizer
        out["run_optimizer"] = _dig(data, ["optimizer", "name"]) or _dig(data, ["optim", "name"]) or _dig(data, ["optimizer"]) or _dig(data, ["optim"]) or _dig(data, ["trainer", "optimizer"])
        # scheduler
        out["run_scheduler"] = _dig(data, ["scheduler", "name"]) or _dig(data, ["lr_scheduler", "name"]) or _dig(data, ["scheduler"]) or _dig(data, ["lr_scheduler"]) 
        # model name/backbone
        out["run_model"] = _dig(data, ["model", "name"]) or _dig(data, ["model_name"]) or _dig(data, ["architecture", "model"]) or _dig(data, ["backbone"]) or _dig(data, ["model"]) 
        # device & seed
        out["run_device"] = _dig(data, ["device"]) or _dig(data, ["train", "device"]) or _dig(data, ["training", "device"]) or _dig(data, ["trainer", "device"]) 
        out["run_seed"] = _dig(data, ["seed"]) or _dig(data, ["train", "seed"]) or _dig(data, ["training", "seed"]) or _dig(data, ["reproducibility", "seed"]) 
        # loss settings
        out["run_loss_base_weight"] = (
            _dig(data, ["loss", "base_weight"]) or _dig(data, ["loss", "weights", "base"]) or _dig(data, ["loss_base_weight"]) or _dig(data, ["losses", "base_weight"]) 
        )
        out["run_loss_svd_weights"] = _dig(data, ["loss", "svd_weights"]) or _dig(data, ["loss", "weights", "svd"]) or _dig(data, ["loss_svd_weights"]) or _dig(data, ["losses", "svd_weights"]) 
        out["run_loss_topk"] = _dig(data, ["loss", "topk"]) or _dig(data, ["loss_topk"]) or _dig(data, ["losses", "topk"]) 
    except Exception:
        pass
    return out


def _file_size_mb(p: Path) -> Optional[float]:
    try:
        if p.exists() and p.is_file():
            return round(p.stat().st_size / (1024 * 1024), 3)
    except Exception:
        return None
    return None


def gather_artifacts_info(attn_dir: Path, attn: str) -> Dict[str, Any]:
    """Collect presence/size of key artifacts under a single attention run directory."""
    info: Dict[str, Any] = {
        "artifact_best_model_path": None,
        "artifact_best_model_mb": None,
        "artifact_checkpoint_path": None,
        "artifact_checkpoint_mb": None,
        "artifact_loss_plot_path": None,
        "artifact_loss_plot_exists": False,
        "artifact_tb_events": 0,
        "artifact_hardware_log_files": 0,
    }
    # best model (*.pt)
    best = None
    for patt in [f"best_model_{attn}.pt", "best_model*.pt", "*.pt"]:
        cand = next((c for c in attn_dir.glob(patt) if c.is_file()), None)
        if cand:
            best = cand
            break
    if best:
        info["artifact_best_model_path"] = str(best)
        info["artifact_best_model_mb"] = _file_size_mb(best)
    # checkpoint (*.pth)
    ckpt = None
    for patt in [f"checkpoint_{attn}.pth", "checkpoint*.pth", "*.pth"]:
        cand = next((c for c in attn_dir.glob(patt) if c.is_file()), None)
        if cand:
            ckpt = cand
            break
    if ckpt:
        info["artifact_checkpoint_path"] = str(ckpt)
        info["artifact_checkpoint_mb"] = _file_size_mb(ckpt)
    # loss curve png
    plot = next((c for c in attn_dir.glob(f"loss_curve_{attn}.png") if c.is_file()), None) or \
           next((c for c in attn_dir.glob("loss_curve*.png") if c.is_file()), None)
    if plot:
        info["artifact_loss_plot_path"] = str(plot)
        info["artifact_loss_plot_exists"] = True
    # TensorBoard runs
    runs_dir = attn_dir / "runs"
    if runs_dir.exists():
        try:
            info["artifact_tb_events"] = sum(1 for p in runs_dir.rglob("*") if p.is_file() and ("tfevents" in p.name.lower()))
        except Exception:
            pass
    # hardware logs
    hw_dir = attn_dir / "hardware_logs"
    if hw_dir.exists():
        try:
            info["artifact_hardware_log_files"] = sum(1 for p in hw_dir.rglob("*") if p.is_file())
        except Exception:
            pass
    return info

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

    # research_data might be under attn_dir/research_data
    research_dir = attn_dir / "research_data"
    research_stats = read_research_metrics(research_dir, attn)

    # New: parse per-run main_config.yaml (best-effort)
    main_cfg_path = research_dir / "configs" / "main_config.yaml"
    run_cfg = parse_main_config(main_cfg_path)

    # New: collect artifact presence and size
    artifacts = gather_artifacts_info(attn_dir, attn)

    return {
        "attention": attn,
        "source_group": source_group,
        "path": str(attn_dir),
        **loss_stats,
        "test_result_file_loss": test_result,
        **test_stats,
        **research_stats,
        **run_cfg,
        **artifacts,
        "status": "ok",
    }


def write_csv_summary(entries: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "attention", "source_group", "loss_config_id", "status", "path", "epochs",
        "final_train_loss", "final_valid_loss", "final_test_loss",
        "best_valid_loss", "best_valid_epoch",
        "test_result_file_loss", "avg_test_loss", "min_test_loss", "max_test_loss",
        "total_duration_seconds", "model_parameters", "flops_human", "params_profile",
        "config_base_weight", "config_svd_weights", "config_topk", "failed_min_valid_loss",
        # New: per-run config fields
        "run_config_path", "run_dataset", "run_epochs", "run_batch_size", "run_lr", "run_optimizer", "run_scheduler", "run_model", "run_device", "run_seed",
        "run_loss_base_weight", "run_loss_svd_weights", "run_loss_topk",
        # New: artifact fields
        "artifact_best_model_path", "artifact_best_model_mb", "artifact_checkpoint_path", "artifact_checkpoint_mb",
        "artifact_loss_plot_path", "artifact_loss_plot_exists", "artifact_tb_events", "artifact_hardware_log_files",
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


def generate_markdown(entries: List[Dict[str, Any]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)

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

    # Top-5 summary
    lines.append("## 总体排名（Top-5，按测试损失）\n")
    lines.append("| 排名 | 注意力 | 平均测试损失 | 最终测试损失 | 最佳验证损失 | 轮数 | 源目录 | Loss配置 |\n")
    lines.append("|---:|:--|--:|--:|--:|--:|:--|:--|\n")
    for i, e in enumerate(best_list[:5], 1):
        lines.append(
            f"| {i} | {e['attention']} | {fmt(e.get('avg_test_loss'))} | {fmt(e.get('final_test_loss'))} | {fmt(e.get('best_valid_loss'))} | {e.get('epochs') or '-'} | {e.get('source_group')} | {e.get('loss_config_id') or '-'} |\n"
        )

    # Detailed table (add run config columns)
    lines.append("\n## 详细对比（每类注意力的最佳一次）\n")
    lines.append("| 注意力 | 平均测试损失 | 最终测试损失 | 最佳验证损失 | 轮数 | 训练时长(s) | 参数量 | FLOPs | 源目录 | Loss配置 | base_weight | topk | 数据集 | 批量 | 学习率 | 优化器 |\n")
    lines.append("|:--|--:|--:|--:|--:|--:|--:|:--|:--|:--|--:|--:|:--|--:|--:|:--|\n")
    for e in best_list:
        lines.append(
            f"| {e['attention']} | {fmt(e.get('avg_test_loss'))} | {fmt(e.get('final_test_loss'))} | {fmt(e.get('best_valid_loss'))} | {e.get('epochs') or '-'} | {fmt(e.get('total_duration_seconds'))} | {e.get('model_parameters') or '-'} | {e.get('flops_human') or '-'} | {e.get('source_group')} | {e.get('loss_config_id') or '-'} | {fmt(e.get('config_base_weight'))} | {e.get('config_topk') or '-'} | {e.get('run_dataset') or '-'} | {fmt(e.get('run_batch_size'))} | {fmt(e.get('run_lr'))} | {e.get('run_optimizer') or '-'} |\n"
        )

    # Failures
    failures = [e for e in entries if e.get("status") == "failed"]
    if failures:
        lines.append("\n## 失败实验（来自 failed_attention_log.txt）\n")
        lines.append("| 注意力 | 最小验证损失 | 来源 |\n")
        lines.append("|:--|--:|:--|\n")
        for e in failures:
            lines.append(f"| {e['attention']} | {fmt(e.get('failed_min_valid_loss'))} | {e.get('source_group')} |\n")

    # Notes on data sources
    lines.append("\n## 数据来源与解析说明\n")
    lines.append("- loss_logs/loss_log.txt：解析最终/最佳损失与轮数。\n")
    lines.append("- test_results/test_loss_log.txt：解析批次级测试损失与平均值。\n")
    lines.append("- test_result_*.txt：解析一次性汇报的测试损失。\n")
    lines.append("- research_data/comprehensive_metrics_*.json：若存在，补充训练时长、模型复杂度与FLOPs。\n")
    lines.append("- research_data/configs/main_config.yaml：若存在，抽取数据集、批量、学习率、优化器等运行配置。\n")
    lines.append("- 产物统计：记录 best_model/ckpt 文件大小、是否存在 loss 曲线图、TensorBoard 事件文件数量与硬件日志文件数。\n")
    lines.append("- failed_attention_log.txt：若存在，列出训练失败的注意力机制（来自训练脚本 main.py）。\n")

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