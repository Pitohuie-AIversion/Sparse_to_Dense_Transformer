import os
import re
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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


def scan_runs(base_root: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    # Candidate roots
    codex_root = base_root / "VIVTransformer-4sh2r1-codex"
    dated_root = codex_root / "modify_multi_attention" / "attention_results"
    loss_cfg_root = codex_root / "attention_results"

    # 1) Dated runs under modify_multi_attention/attention_results/<datetime>/<attn>
    if dated_root.exists():
        for run_dir in sorted([p for p in dated_root.iterdir() if p.is_dir()]):
            for attn_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
                attn = attn_dir.name
                entry = collect_single_run(attn_dir, attn, source_group=f"dated:{run_dir.name}")
                entries.append(entry)

    # 2) loss_config_* runs under codex/attention_results/loss_config_X/<attn>
    if loss_cfg_root.exists():
        for cfg_dir in sorted([p for p in loss_cfg_root.iterdir() if p.is_dir() and p.name.startswith("loss_config_")]):
            for attn_dir in sorted([p for p in cfg_dir.iterdir() if p.is_dir()]):
                attn = attn_dir.name
                entry = collect_single_run(attn_dir, attn, source_group=cfg_dir.name)
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

    return {
        "attention": attn,
        "source_group": source_group,
        "path": str(attn_dir),
        **loss_stats,
        "test_result_file_loss": test_result,
        **test_stats,
        **research_stats,
    }


def write_csv_summary(entries: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "attention", "source_group", "path", "epochs",
        "final_train_loss", "final_valid_loss", "final_test_loss",
        "best_valid_loss", "best_valid_epoch",
        "test_result_file_loss", "avg_test_loss", "min_test_loss", "max_test_loss",
        "total_duration_seconds", "model_parameters", "flops_human", "params_profile",
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
    best_map = best_by_attention(entries, key_order=["avg_test_loss", "final_test_loss", "best_valid_loss"])

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
    lines.append("| 排名 | 注意力 | 平均测试损失 | 最终测试损失 | 最佳验证损失 | 轮数 | 源目录 |\n")
    lines.append("|---:|:--|--:|--:|--:|--:|:--|\n")
    for i, e in enumerate(best_list[:5], 1):
        lines.append(
            f"| {i} | {e['attention']} | {fmt(e.get('avg_test_loss'))} | {fmt(e.get('final_test_loss'))} | {fmt(e.get('best_valid_loss'))} | {e.get('epochs') or '-'} | {e.get('source_group')} |\n"
        )

    # Detailed table
    lines.append("\n## 详细对比（每类注意力的最佳一次）\n")
    lines.append("| 注意力 | 平均测试损失 | 最终测试损失 | 最佳验证损失 | 轮数 | 训练时长(s) | 参数量 | FLOPs | 源目录 |\n")
    lines.append("|:--|--:|--:|--:|--:|--:|--:|:--|:--|\n")
    for e in best_list:
        lines.append(
            f"| {e['attention']} | {fmt(e.get('avg_test_loss'))} | {fmt(e.get('final_test_loss'))} | {fmt(e.get('best_valid_loss'))} | {e.get('epochs') or '-'} | {fmt(e.get('total_duration_seconds'))} | {e.get('model_parameters') or '-'} | {e.get('flops_human') or '-'} | {e.get('source_group')} |\n"
        )

    # Notes on data sources
    lines.append("\n## 数据来源与解析说明\n")
    lines.append("- loss_logs/loss_log.txt：解析最终/最佳损失与轮数。\n")
    lines.append("- test_results/test_loss_log.txt：解析批次级测试损失与平均值。\n")
    lines.append("- test_result_*.txt：解析一次性汇报的测试损失。\n")
    lines.append("- research_data/comprehensive_metrics_*.json：若存在，补充训练时长、模型复杂度与FLOPs。\n")

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
    base_root = script_path.parents[2]

    entries = scan_runs(base_root)

    report_dir = base_root / "report"
    out_csv = report_dir / "attention_summary.csv"
    out_md = report_dir / "attention_report.md"

    write_csv_summary(entries, out_csv)
    generate_markdown(entries, out_md)

    print(f"Summary CSV: {out_csv}")
    print(f"Markdown Report: {out_md}")
    print(f"Total entries: {len(entries)}")


if __name__ == "__main__":
    main()