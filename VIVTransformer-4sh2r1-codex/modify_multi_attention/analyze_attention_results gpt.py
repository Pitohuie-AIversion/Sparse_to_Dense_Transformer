import argparse
import csv
import json
import re
import ast
import statistics
import math
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

yaml = None
try:
    import yaml
except ImportError:
    pass


def safe_json_load(json_path: Path) -> Dict[str, Any]:
    """Load JSON content from a file path, return empty dict on any error."""
    try:
        if not json_path.exists():
            return {}
        text = json_path.read_text(encoding="utf-8", errors="ignore")
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def safe_float(v: Any) -> Optional[float]:
    """Best-effort conversion to finite float. Return None if invalid."""
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            fv = float(v)
            return fv if math.isfinite(fv) else None
        s = str(v).strip()
        # Remove commas and surrounding quotes
        s = s.strip("'\"").replace(",", "")
        if s.lower() in {"nan", "inf", "+inf", "-inf"}:
            return None
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not m:
            return None
        fv = float(m.group(0))
        return fv if math.isfinite(fv) else None
    except Exception:
        return None


def parse_loss_log(loss_log: Path) -> Dict[str, Any]:
    """Parse a training loss log file to extract key metrics.
    Supports JSON logs and plain-text logs with lines containing 'train'/'val' losses.
    """
    out: Dict[str, Any] = {"training_status": "unknown"}
    if not loss_log or not loss_log.exists():
        out["training_status"] = "no_log_file"
        return out

    text = loss_log.read_text(encoding="utf-8", errors="ignore")
    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # Pass through known fields if present
            for k in [
                "epochs", "final_train_loss", "final_valid_loss",
                "best_valid_loss", "best_valid_epoch", "training_status"
            ]:
                if k in data:
                    out[k] = data[k]
            # Normalize status
            if out.get("training_status") is None:
                out["training_status"] = "ok"
            return out
    except Exception:
        pass

    # Try CSV-like parsing with headers such as: Epoch, Train Loss, Valid Loss, Test Loss
    try:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        header_idx = -1
        for i, ln in enumerate(lines):
            low = ln.lower()
            if "," in ln and ("epoch" in low) and (("train" in low) or ("valid" in low) or ("val" in low)) and ("loss" in low):
                header_idx = i
                break
        if header_idx >= 0:
            reader = csv.DictReader(lines[header_idx:])
            epochs_list: List[int] = []
            train_list: List[float] = []
            valid_list: List[float] = []
            for row in reader:
                if not row:
                    continue
                keymap = { (k or "").strip().lower(): k for k in row.keys() if k is not None }
                def getcol(names: List[str]) -> Optional[float]:
                    for name in names:
                        col = keymap.get(name)
                        if col is not None:
                            v = safe_float(row.get(col))
                            if v is not None:
                                return v
                    return None
                ep = getcol(["epoch"])  # epoch index
                tr = getcol(["train loss", "train_loss", "train"])  # training loss
                va = getcol(["valid loss", "validation loss", "val loss", "val_loss", "valid", "validation", "val"])  # validation loss
                if ep is not None:
                    try:
                        epochs_list.append(int(ep))
                    except Exception:
                        pass
                if tr is not None:
                    train_list.append(tr)
                if va is not None:
                    valid_list.append(va)
            if train_list:
                out["final_train_loss"] = train_list[-1]
            if valid_list:
                out["final_valid_loss"] = valid_list[-1]
                best_val = min(valid_list)
                out["best_valid_loss"] = best_val
                try:
                    idx = valid_list.index(best_val)
                    if epochs_list and len(epochs_list) == len(valid_list):
                        out["best_valid_epoch"] = int(epochs_list[idx])
                    else:
                        out["best_valid_epoch"] = idx + 1
                except Exception:
                    pass
            # Epochs fallback from list lengths
            if epochs_list:
                try:
                    out["epochs"] = max(epochs_list)
                except Exception:
                    pass
            if not out.get("epochs"):
                seq_len = max(len(train_list), len(valid_list))
                if seq_len:
                    out["epochs"] = seq_len
            # Set training status
            if out.get("epochs"):
                out["training_status"] = "completed"
            elif train_list or valid_list:
                out["training_status"] = "in_progress"
            else:
                out["training_status"] = "unknown"
            return out
    except Exception:
        pass

    # Fallback: regex parsing from text
    train_losses = [safe_float(m) for m in re.findall(r"train[^\n]*?(?:loss|Loss)[^\d\n]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)]
    val_losses = [safe_float(m) for m in re.findall(r"val(?:id)?(?:ation)?[^\n]*?(?:loss|Loss)[^\d\n]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)]
    train_losses = [x for x in train_losses if x is not None]
    val_losses = [x for x in val_losses if x is not None]

    # Epochs: robust detection - prefer totals from "Epoch X/Y", else use max single
    epochs_total_candidates = [int(m.group(2)) for m in re.finditer(r"[Ee]poch\s*(\d+)\s*/\s*(\d+)", text)]
    epochs_single_candidates = [int(m.group(1)) for m in re.finditer(r"[Ee]poch\s*(\d+)(?!\s*/)", text)]
    if epochs_total_candidates:
        out["epochs"] = max(epochs_total_candidates)
    elif epochs_single_candidates:
        out["epochs"] = max(epochs_single_candidates)

    if train_losses:
        out["final_train_loss"] = train_losses[-1]
    if val_losses:
        out["final_valid_loss"] = val_losses[-1]

    # Try explicit 'best valid loss' pattern from logs (preferred if present)
    m_best = re.search(r"best[_\s-]?val(?:id)?(?:ation)?[_\s-]?loss[\s:=]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:[^\n]*?[Ee]poch[^\d]*(\d+))?", text, re.IGNORECASE)
    if m_best:
        bvl = safe_float(m_best.group(1))
        if bvl is not None:
            out["best_valid_loss"] = bvl
            # Prefer epoch mentioned with the best loss if available
            if m_best.lastindex and m_best.lastindex >= 2:
                be = safe_float(m_best.group(2))
                if be is not None:
                    out["best_valid_epoch"] = int(be)
            # If epoch not explicitly present, try infer from sequence position
            if out.get("best_valid_epoch") is None and val_losses:
                try:
                    out["best_valid_epoch"] = val_losses.index(bvl) + 1
                except Exception:
                    pass
    elif val_losses:
        # Fallback: compute best from validation sequence
        best_val = min(val_losses)
        out["best_valid_loss"] = best_val
        try:
            out["best_valid_epoch"] = val_losses.index(best_val) + 1
        except Exception:
            pass

    # Fallback: derive epochs from number of parsed loss lines if still missing
    if not out.get("epochs"):
        seq_len = max(len(train_losses), len(val_losses))
        if seq_len:
            out["epochs"] = seq_len

    # Set training status
    if out.get("epochs"):
        out["training_status"] = "completed"
    elif train_losses or val_losses:
        out["training_status"] = "in_progress"
    else:
        out["training_status"] = "unknown"

    return out


def parse_test_loss_log(test_loss_log: Path) -> Dict[str, Any]:
    """Parse test loss log to compute avg/min/max from numeric values in file.
    Handles JSON or plain-text with numeric values.
    """
    out: Dict[str, Any] = {}
    if not test_loss_log or not test_loss_log.exists():
        return out

    text = test_loss_log.read_text(encoding="utf-8", errors="ignore")
    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for k in ["avg_test_loss", "min_test_loss", "max_test_loss"]:
                if k in data:
                    out[k] = data[k]
            return out
        elif isinstance(data, list):
            nums = [safe_float(x) for x in data]
            nums = [x for x in nums if x is not None]
            if nums:
                out["avg_test_loss"] = sum(nums) / len(nums)
                out["min_test_loss"] = min(nums)
                out["max_test_loss"] = max(nums)
                return out
    except Exception:
        pass

    # Try CSV-like parsing with headers such as: Batch, Test Loss
    try:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        header_idx = -1
        for i, ln in enumerate(lines):
            low = ln.lower()
            if "," in ln and ("batch" in low) and ("loss" in low):
                header_idx = i
                break
        if header_idx >= 0:
            reader = csv.DictReader(lines[header_idx:])
            vals: List[float] = []
            for row in reader:
                if not row:
                    continue
                keymap = { (k or "").strip().lower(): k for k in row.keys() if k is not None }
                col_name = None
                for name in ["test loss", "test_loss", "loss", "value"]:
                    if name in keymap:
                        col_name = keymap[name]
                        break
                if col_name is None:
                    # Try second column if present
                    keys = list(row.keys())
                    if len(keys) >= 2:
                        col_name = keys[1]
                if col_name is None:
                    continue
                v = safe_float(row.get(col_name))
                if v is not None:
                    vals.append(v)
            if vals:
                out["avg_test_loss"] = sum(vals) / len(vals)
                out["min_test_loss"] = min(vals)
                out["max_test_loss"] = max(vals)
                return out
    except Exception:
        pass

    # Look for an explicit average line like: Average Test Loss: 0.1234
    m_avg = re.search(r"Average\s+Test\s+Loss\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text, re.IGNORECASE)
    if m_avg:
        avg = safe_float(m_avg.group(1))
        if avg is not None:
            out["avg_test_loss"] = avg
            out["min_test_loss"] = avg
            out["max_test_loss"] = avg
            return out

    # Fallback: extract all floats from text
    nums = [safe_float(m) for m in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]
    nums = [x for x in nums if x is not None]
    if nums:
        out["avg_test_loss"] = sum(nums) / len(nums)
        out["min_test_loss"] = min(nums)
        out["max_test_loss"] = max(nums)
    return out


def parse_test_result_file(test_result_file: Path) -> Optional[float]:
    """Parse a one-shot test result file and return the numeric loss if found.
    Supports JSON with a scalar or dict containing 'final_test_loss', or plain text.
    """
    if not test_result_file or not test_result_file.exists():
        return None
    text = test_result_file.read_text(encoding="utf-8", errors="ignore").strip()
    # Try JSON
    try:
        data = json.loads(text)
        if isinstance(data, (int, float)):
            fv = float(data)
            return fv if math.isfinite(fv) else None
        if isinstance(data, dict):
            cand = data.get("final_test_loss") or data.get("test_loss") or data.get("loss")
            return safe_float(cand)
    except Exception:
        pass
    # Regex fallbacks
    m = re.search(r"final[_\s-]*test[_\s-]*loss\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text, re.IGNORECASE)
    if m:
        return safe_float(m.group(1))
    # First float in file
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return safe_float(m.group(0)) if m else None


def analyze_training_convergence(loss_log: Path) -> Dict[str, Any]:
    """Compute convergence metrics from parsed loss log.
    Outputs:
      - loss_decreasing_ratio: ratio of epochs where val loss decreased vs previous
      - epochs_after_best_valid: epochs after best validation loss
      - overfitting_ratio: last_val / best_val
      - train_loss_variance_late / valid_loss_variance_late: variance on last 20% epochs
      - convergence_epoch_90pct: first epoch reaching 90% of improvement from start to best (val)
    """
    parsed = parse_loss_log(loss_log) if loss_log and loss_log.exists() else {}
    out: Dict[str, Any] = {}
    vals = []
    trains = []
    # Reconstruct sequences from text if possible
    try:
        text = loss_log.read_text(encoding="utf-8", errors="ignore") if loss_log and loss_log.exists() else ""
        v_seq = [safe_float(m) for m in re.findall(
            r"val(?:id)?(?:ation)?[^\n]*?(?:loss|Loss)[^\d\n]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)]
        t_seq = [safe_float(m) for m in re.findall(
            r"train[^\n]*?(?:loss|Loss)[^\d\n]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)]
        vals = [x for x in v_seq if x is not None]
        trains = [x for x in t_seq if x is not None]
    except Exception:
        vals = []
        trains = []

    if vals:
        dec = 0
        for i in range(1, len(vals)):
            if vals[i] <= vals[i-1]:
                dec += 1
        out["loss_decreasing_ratio"] = dec / max(len(vals)-1, 1)
        best_v = min(vals)
        out["epochs_after_best_valid"] = (len(vals) - (vals.index(best_v) + 1)) if best_v in vals else None
        # Overfitting ratio: last / best
        if best_v is not None and vals[-1] is not None and best_v > 0:
            out["overfitting_ratio"] = vals[-1] / best_v

        # Variance on last 20% epochs (at least 3 points)
        def tail_var(seq):
            if not seq:
                return None
            start = max(0, int(len(seq) * 0.8))
            tail = seq[start:] if len(seq) - start >= 3 else seq[-3:]
            try:
                return statistics.pvariance(tail) if len(tail) >= 2 else 0.0
            except Exception:
                return None
        out["valid_loss_variance_late"] = tail_var(vals)
        if trains:
            out["train_loss_variance_late"] = tail_var(trains)

        # Convergence epoch at "90% to best": first epoch with val <= best + 0.1*(start-best)
        try:
            start_v = vals[0]
            if start_v is not None and best_v is not None and start_v > best_v:
                target = best_v + 0.1 * (start_v - best_v)
                ce = None
                for i, v in enumerate(vals, 1):
                    if v is not None and v <= target:
                        ce = i
                        break
                out["convergence_epoch_90pct"] = ce
        except Exception:
            pass
    return out


def read_research_metrics(research_dir: Path, attn: str) -> Dict[str, Any]:
    """Aggregate metrics from research_data JSON files if present.
    Also returns additional enriched fields when possible:
    - model_size_mb (estimated from model_parameters assuming float32)
    - training_epochs (from experiment_summary or convergence files)
    - attention_type (inferred from config or provided attn)
    - experiment_name (directory name of the experiment)
    - dataset_info (lightweight dataset metadata extracted from config if available)
    - performance_metrics (dict collecting key performance indicators discovered)
    - attention_visualization (presence and paths of viz artifacts)
    """
    out: Dict[str, Any] = {}
    if not research_dir or not research_dir.exists():
        return out

    # Determine base directories to search: prefer research_data subdir when present
    base_dirs: List[Path] = []
    rd_sub = research_dir / "research_data"
    if rd_sub.exists():
        base_dirs.append(rd_sub)
    base_dirs.append(research_dir)

    def _find_first(path_rel: str) -> Optional[Path]:
        for b in base_dirs:
            p = b / path_rel
            if p.exists():
                return p
        return None

    # system_info.json
    sys_path = _find_first("system_info.json")
    sys_info = safe_json_load(sys_path) if sys_path else None
    if sys_info:
        # Normalize some common keys
        out["system_info"] = {
            "python_version": sys_info.get("python_version") or sys_info.get("python"),
            "torch_version": sys_info.get("torch_version") or sys_info.get("pytorch_version"),
            "cuda_version": sys_info.get("cuda_version"),
            "gpu_name": sys_info.get("gpu_name"),
            "cpu_count": sys_info.get("cpu_count"),
            "total_ram_gb": sys_info.get("total_ram_gb") or sys_info.get("ram_gb"),
        }

    # comprehensive_metrics_{attn}.json
    comp_path = _find_first(f"comprehensive_metrics_{attn}.json")
    comp = safe_json_load(comp_path) if comp_path else None
    if comp:
        # These keys may include nested summaries
        for k in ["hardware_summary", "learning_rate_schedule", "optimizer_info", "convergence_metrics"]:
            if k in comp and isinstance(comp[k], dict):
                out[k] = comp[k]
        # Common top-level metrics
        for k in ["total_duration_seconds", "model_parameters", "flops_human", "params_profile"]:
            if k in comp:
                out[k] = comp[k]
        # Some pipelines store final_test_loss here
        if comp.get("final_test_loss") is not None:
            out["final_test_loss_collector"] = comp.get("final_test_loss")
        elif comp.get("experiment_summary") and isinstance(comp["experiment_summary"], dict):
            out["final_test_loss_collector"] = comp["experiment_summary"].get("final_test_loss")
        # Fallbacks from nested structures in comprehensive metrics
        if isinstance(comp.get("experiment_summary"), dict):
            es = comp["experiment_summary"]
            if out.get("total_duration_seconds") is None and es.get("total_duration_seconds") is not None:
                out["total_duration_seconds"] = es.get("total_duration_seconds")
            if out.get("training_epochs") is None and es.get("total_epochs") is not None:
                out["training_epochs"] = es.get("total_epochs")
        if isinstance(comp.get("model_complexity"), dict):
            mc = comp["model_complexity"]
            if out.get("model_parameters") is None and mc.get("total_parameters") is not None:
                out["model_parameters"] = mc.get("total_parameters")

    # convergence_analysis_{attn}.json
    conv_path = _find_first(f"convergence_analysis_{attn}.json")
    conv = safe_json_load(conv_path) if conv_path else None
    if conv:
        # Prefer nested convergence_statistics if available
        cs = conv.get("convergence_statistics") if isinstance(conv.get("convergence_statistics"), dict) else None
        if cs:
            if cs.get("best_val_loss") is not None:
                out["derived_best_valid_loss"] = cs.get("best_val_loss")
            if cs.get("best_val_epoch") is not None:
                out["derived_best_valid_epoch"] = cs.get("best_val_epoch")
            if out.get("training_epochs") is None and cs.get("total_epochs") is not None:
                out["training_epochs"] = cs.get("total_epochs")
        else:
            # Legacy flat keys
            bvl = conv.get("best_val_loss") or conv.get("best_valid_loss")
            if bvl is not None:
                out["derived_best_valid_loss"] = bvl
            bve = conv.get("best_val_epoch") or conv.get("best_valid_epoch")
            if bve is not None:
                out["derived_best_valid_epoch"] = bve
            if out.get("training_epochs") is None:
                out["training_epochs"] = conv.get("total_epochs") or conv.get("epochs")

    # Try to derive hardware_summary from hardware_metrics.json when not present
    if not out.get("hardware_summary"):
        candidate_hm_paths: List[Path] = []
        for b in base_dirs:
            candidate_hm_paths.extend([
                b / "hardware_metrics.json",
                b / "hardware_logs" / "hardware_metrics.json",
            ])
        for hm_path in candidate_hm_paths:
            hm = safe_json_load(hm_path)
            if not hm:
                continue
            try:
                epoch_metrics = hm.get("epoch_metrics") or []
                gpu_utils: List[float] = []
                gpu_mems: List[float] = []
                gpu_temps: List[float] = []
                cpu_percents: List[float] = []
                rams: List[float] = []
                for m in epoch_metrics:
                    gm = m.get("gpu_metrics")
                    if isinstance(gm, list) and gm:
                        g0 = gm[0]
                        u = g0.get("gpu_utilization_percent")
                        mem = g0.get("memory_used_mb")
                        temp = g0.get("temperature_celsius")
                        if isinstance(u, (int, float)):
                            gpu_utils.append(float(u))
                        if isinstance(mem, (int, float)):
                            gpu_mems.append(float(mem))
                        if isinstance(temp, (int, float)):
                            gpu_temps.append(float(temp))
                    if isinstance(m.get("cpu_percent"), (int, float)):
                        cpu_percents.append(float(m.get("cpu_percent")))
                    if isinstance(m.get("ram_used_mb"), (int, float)):
                        rams.append(float(m.get("ram_used_mb")))
                if gpu_utils or gpu_mems or gpu_temps or cpu_percents or rams:
                    out["hardware_summary"] = {
                        "avg_gpu_utilization": round(sum(gpu_utils) / len(gpu_utils), 2) if gpu_utils else None,
                        "max_gpu_memory_mb": max(gpu_mems) if gpu_mems else None,
                        "gpu_temperature_max": max(gpu_temps) if gpu_temps else None,
                        "avg_cpu_percent": round(sum(cpu_percents) / len(cpu_percents), 2) if cpu_percents else None,
                        "max_ram_mb": max(rams) if rams else None,
                    }
                    break
            except Exception:
                # Ignore parse issues and continue
                pass

    # Optional: attach config info if present in common places
    if yaml is not None and not out.get("config_info"):
        # Try using the dedicated helper to benefit from broader search
        try:
            # Determine experiment directory base for config lookup
            exp_base = research_dir if research_dir.name != "research_data" else research_dir.parent
            cfg = read_config_info(exp_base)
            if cfg:
                out["config_info"] = cfg
        except Exception:
            pass

    # Enrichment: compute additional friendly fields
    # model_size_mb: estimate using float32 parameters
    params = out.get("model_parameters")
    if isinstance(params, (int, float)) and params is not None:
        try:
            out["model_size_mb"] = round(float(params) * 4.0 / 1_000_000.0, 2)
        except Exception:
            pass

    # Ensure attention_type present
    if not out.get("attention_type"):
        if isinstance(out.get("config_info"), dict):
            # try to extract attention_type from config_info
            def _find_attention_type(node: Any) -> Optional[str]:
                if isinstance(node, dict):
                    for k in ["attention_type", "attention", "attn_type", "attention_name"]:
                        v = node.get(k)
                        if isinstance(v, str):
                            return v
                    for v in node.values():
                        t = _find_attention_type(v)
                        if t:
                            return t
                elif isinstance(node, list):
                    for v in node:
                        t = _find_attention_type(v)
                        if t:
                            return t
                return None
            a_type = _find_attention_type(out["config_info"]) or attn
            if a_type:
                out["attention_type"] = a_type
        else:
            out["attention_type"] = attn

    # experiment_name: prefer the experiment directory rather than the research_data folder name
    exp_dir_base = research_dir if research_dir.name != "research_data" else research_dir.parent
    out.setdefault("experiment_name", exp_dir_base.name)

    # dataset_info: extract lightweight info from config if available
    ds_info: Dict[str, Any] = {}
    cfg_info = out.get("config_info") if isinstance(out.get("config_info"), dict) else {}
    if cfg_info:
        # Common dataset fields
        for k in ["dataset", "dataset_name", "data_name", "data_path", "dataset_path"]:
            if cfg_info.get(k) is not None:
                ds_info[k] = cfg_info.get(k)
        # Nested common structures
        for node_key in ["data", "dataset", "datamodule", "dataloader", "loader", "train_data"]:
            node = cfg_info.get(node_key)
            if isinstance(node, dict):
                for kk in ["name", "dataset", "path", "root", "split", "batch_size", "num_workers"]:
                    if node.get(kk) is not None:
                        ds_info[f"{node_key}.{kk}"] = node.get(kk)
    if ds_info:
        out["dataset_info"] = ds_info

    # performance_metrics: consolidate available signals
    perf: Dict[str, Any] = {}
    if out.get("final_test_loss_collector") is not None:
        perf["final_test_loss"] = out["final_test_loss_collector"]
    if out.get("derived_best_valid_loss") is not None:
        perf["best_valid_loss"] = out["derived_best_valid_loss"]
    if out.get("derived_best_valid_epoch") is not None:
        perf["best_valid_epoch"] = out["derived_best_valid_epoch"]
    if out.get("training_epochs") is not None:
        perf["training_epochs"] = out["training_epochs"]
    if perf:
        out["performance_metrics"] = perf

    # attention_visualization: detect presence using helper
    try:
        viz = read_attention_visualization_data(exp_dir_base, attn)
        if viz:
            out["attention_visualization"] = {
                "has_image": bool(viz.get("attention_image")),
                "image_path": viz.get("attention_image"),
                "num_maps": viz.get("num_maps"),
                "num_stats": viz.get("num_stats"),
            }
        else:
            out["attention_visualization"] = {"has_image": False}
    except Exception:
        pass

    # Config files optionally present under research_dir/configs as a fallback (keep existing behavior)
    if yaml is not None and not out.get("config_info"):
        cfg_dir = research_dir / "configs"
        for yml in [cfg_dir / "config.yaml", cfg_dir / "config_server.yaml", research_dir / "config.yaml", research_dir / "config_server.yaml"]:
            if yml.exists():
                try:
                    cfg = yaml.safe_load(yml.read_text(encoding="utf-8", errors="ignore")) or {}
                    out.setdefault("config_info", cfg)
                    break
                except Exception:
                    pass

    return out


def read_hardware_summary(exp_dir: Path) -> Dict[str, Any]:
    """Read and summarize hardware monitoring data under an experiment directory.
    It searches common locations for hardware_metrics.json and computes normalized fields:
    - avg_gpu_utilization
    - max_gpu_memory_mb
    - gpu_temperature_max
    Also attempts to include CPU and RAM indicators if present.
    """
    if not exp_dir or not exp_dir.exists():
        return {}

    candidates = [
        exp_dir / "hardware_metrics.json",
        exp_dir / "hardware_logs" / "hardware_metrics.json",
        exp_dir / "research_data" / "hardware_metrics.json",
        exp_dir / "research_data" / "hardware_logs" / "hardware_metrics.json",
    ]

    hm: Dict[str, Any] = {}
    for p in candidates:
        hm = safe_json_load(p)
        if hm:
            break
    if not hm:
        return {}

    # Prefer epoch-wise aggregation to compute robust summary
    epoch_metrics = hm.get("epoch_metrics") or []
    gpu_utils: List[float] = []
    gpu_mems: List[float] = []
    gpu_temps: List[float] = []
    cpu_percents: List[float] = []
    rams: List[float] = []
    total_epoch_time = 0.0

    for m in epoch_metrics:
        try:
            if isinstance(m.get("epoch_duration_seconds"), (int, float)):
                total_epoch_time += float(m.get("epoch_duration_seconds"))
            if isinstance(m.get("cpu_percent"), (int, float)):
                cpu_percents.append(float(m.get("cpu_percent")))
            if isinstance(m.get("ram_used_mb"), (int, float)):
                rams.append(float(m.get("ram_used_mb")))
            gm = m.get("gpu_metrics")
            if isinstance(gm, list) and gm:
                g0 = gm[0]
                u = g0.get("gpu_utilization_percent")
                mem = g0.get("memory_used_mb")
                temp = g0.get("temperature_celsius")
                if isinstance(u, (int, float)):
                    gpu_utils.append(float(u))
                if isinstance(mem, (int, float)):
                    gpu_mems.append(float(mem))
                if isinstance(temp, (int, float)):
                    gpu_temps.append(float(temp))
        except Exception:
            continue

    # If the file already contains summarized keys, map them too as fallbacks
    summary_map_candidates = [hm]
    # Some implementations may store an inline summary under a key
    if isinstance(hm.get("summary"), dict):
        summary_map_candidates.append(hm["summary"]) 
    if hm.get("training_metrics") and isinstance(hm["training_metrics"], list):
        # Look for the last training_end metrics item
        try:
            last_tm = next((t for t in reversed(hm["training_metrics"]) if t.get("event") == "training_end"), None)
            if isinstance(last_tm, dict):
                summary_map_candidates.append(last_tm)
        except Exception:
            pass

    # Build output
    out: Dict[str, Any] = {
        "avg_gpu_utilization": round(sum(gpu_utils) / len(gpu_utils), 2) if gpu_utils else None,
        "max_gpu_memory_mb": max(gpu_mems) if gpu_mems else None,
        "gpu_temperature_max": max(gpu_temps) if gpu_temps else None,
        "avg_cpu_percent": round(sum(cpu_percents) / len(cpu_percents), 2) if cpu_percents else None,
        "max_ram_mb": max(rams) if rams else None,
        "total_duration_seconds": None,
        "epochs": len([m for m in epoch_metrics if m.get("event") == "epoch_end"]) if epoch_metrics else None,
    }

    # Derive total training duration
    if epoch_metrics:
        out["total_duration_seconds"] = round(total_epoch_time, 2)
    # Fallback: try to read from training_metrics
    for cand in summary_map_candidates:
        if out.get("total_duration_seconds") is None and isinstance(cand.get("total_duration_seconds"), (int, float)):
            out["total_duration_seconds"] = float(cand["total_duration_seconds"])  
        # Map alternate key names if present
        if out.get("avg_gpu_utilization") is None and isinstance(cand.get("avg_gpu_utilization_percent"), (int, float)):
            out["avg_gpu_utilization"] = float(cand.get("avg_gpu_utilization_percent"))
        if out.get("max_gpu_memory_mb") is None and isinstance(cand.get("max_gpu_memory_usage_mb"), (int, float)):
            out["max_gpu_memory_mb"] = float(cand.get("max_gpu_memory_usage_mb"))
        if out.get("gpu_temperature_max") is None and isinstance(cand.get("max_gpu_temperature_celsius"), (int, float)):
            out["gpu_temperature_max"] = float(cand.get("max_gpu_temperature_celsius"))

    # Also provide a concise gpu_metrics sub-dict for convenience
    out["gpu_metrics"] = {
        "avg_utilization_percent": out.get("avg_gpu_utilization"),
        "max_memory_mb": out.get("max_gpu_memory_mb"),
        "max_temperature_celsius": out.get("gpu_temperature_max"),
    }
    return out


def read_config_info(exp_dir: Path) -> Dict[str, Any]:
    """Load configuration YAML if present under the experiment directory or its nearby parents.
    Returns the raw parsed dict. Additionally tries to infer a top-level 'attention_type'.
    """
    if not exp_dir or not exp_dir.exists():
        return {}

    # Build candidate YAML locations: local, research_data, configs subfolder, and parent locations
    candidates: List[Path] = []
    search_bases: List[Path] = [exp_dir]
    # Include common subdirectories inside exp_dir
    for sub in ["research_data", "configs", str(Path("research_data") / "configs")]:
        candidates_base = exp_dir / sub
        if candidates_base.exists():
            search_bases.append(candidates_base)
    # Include parent and its configs
    if exp_dir.parent and exp_dir.parent.exists():
        search_bases.append(exp_dir.parent)
        parent_configs = exp_dir.parent / "configs"
        if parent_configs.exists():
            search_bases.append(parent_configs)

    seen: set = set()
    for base in search_bases:
        for name in ["config.yaml", "config_server.yaml"]:
            p = base / name
            if p.exists() and p not in seen:
                candidates.append(p)
                seen.add(p)

    # Fallback: glob any *.yaml inside configs folders if nothing found yet
    if not candidates:
        for base in search_bases:
            if base.name == "configs" and base.exists():
                for p in sorted(base.glob("*.yaml")):
                    candidates.append(p)

    cfg: Dict[str, Any] = {}
    for yml in candidates:
        if yml.exists() and yaml is not None:
            try:
                cfg = yaml.safe_load(yml.read_text(encoding="utf-8", errors="ignore")) or {}
                if cfg:
                    break
            except Exception:
                continue
    if not cfg:
        return {}

    # Try to infer attention_type from common places
    def _find_attention_type(node: Any) -> Optional[str]:
        if isinstance(node, dict):
            # exact matches
            for k in ["attention_type", "attention", "attn_type", "attention_name"]:
                v = node.get(k)
                if isinstance(v, str):
                    return v
            # nested search
            for v in node.values():
                t = _find_attention_type(v)
                if t:
                    return t
        elif isinstance(node, list):
            for v in node:
                t = _find_attention_type(v)
                if t:
                    return t
        return None

    attention_type = _find_attention_type(cfg)
    if attention_type:
        cfg.setdefault("attention_type", attention_type)
    return cfg


def read_attention_visualization_data(exp_dir: Path, attn: str) -> Dict[str, Any]:
    """Collect attention visualization artifacts for a given experiment and attention type.
    Returns a dict with paths (as strings) to found images and stats.
    Keys may include: attention_image, attention_maps, stats_files, num_maps, num_stats.
    """
    if not exp_dir or not exp_dir.exists():
        return {}

    results: Dict[str, Any] = {}

    # Prepare search bases: experiment dir and common subfolders
    bases: List[Path] = [exp_dir]
    for sub in ["attention", "visualizations", "plots", "images", "research_data", "attention_maps"]:
        p = exp_dir / sub
        if p.exists():
            bases.append(p)
    # Also scan immediate subdirectories of attention_maps or visualizations for nested outputs
    nested_candidates: List[Path] = []
    for b in bases:
        if b.name in {"attention_maps", "visualizations", "plots"} and b.exists():
            try:
                for child in b.iterdir():
                    if child.is_dir():
                        nested_candidates.append(child)
            except Exception:
                pass
    bases.extend(nested_candidates)

    # Common single-image visualizations
    single_names = [
        "attention_vis.png",
        "attention_visualization.png",
        "attention_heatmap.png",
        "spatial_attention_visualization.png",
        f"attention_{attn}.png",
    ]
    for b in bases:
        for name in single_names:
            p = b / name
            if p.exists():
                results["attention_image"] = str(p)
                break
        if results.get("attention_image"):
            break

    # Collected attention maps (per-epoch or per-layer)
    glob_patterns = [
        "epoch_*.png",
        f"{attn}_epoch_*.png",
        "layer_*_attention_maps.png",
        "*attention*_map*.png",
    ]
    maps: List[str] = []
    for b in bases:
        for pat in glob_patterns:
            for p in sorted(b.glob(pat)):
                if p.is_file():
                    maps.append(str(p))
    if maps:
        results["attention_maps"] = sorted(set(maps))
        results["num_maps"] = len(results["attention_maps"])

    # Stats JSON generated by analysis utilities
    stats_patterns = [
        f"attention_stats_{attn}_epoch_*.json",
        f"{attn}_attention_stats_epoch_*.json",
        "attention_stats_epoch_*.json",
    ]
    stats_files: List[str] = []
    for b in bases:
        for pat in stats_patterns:
            for p in sorted(b.glob(pat)):
                if p.is_file():
                    stats_files.append(str(p))
    if stats_files:
        results["stats_files"] = sorted(set(stats_files))
        results["num_stats"] = len(results["stats_files"]) 

    # Attach the requested attention type for reference
    results["attention_type"] = attn

    return results


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


def is_experiment_directory(path: Path) -> bool:
    """Check if a directory contains experiment data files"""
    # Check for common experiment indicators
    indicators = [
        (path / "loss_logs").exists(),
        (path / "test_results").exists(),
        any(path.glob("test_result_*.txt")),
        any(path.glob("loss_log.txt")),
        any(path.glob("comprehensive_metrics_*.json")),
        (path / "research_data").exists(),
        any(path.glob("*.json")),  # Any JSON files that might contain metrics
        any(path.glob("hardware_*.json")),
        any(path.glob("system_info.json"))
    ]
    return any(indicators)


def scan_runs(base_root: Path, dated_root_override: Optional[Path] = None, loss_cfg_root_override: Optional[Path] = None) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    print(f"Debug: Starting scan from base_root: {base_root}")

    # Check if base_root itself is an experiment directory
    if is_experiment_directory(base_root):
        attn = base_root.name
        print(f"Debug: Found direct experiment directory: {attn}")
        entry = collect_single_run(base_root, attn, source_group="direct")
        entries.append(entry)
        return entries

    # Enhanced candidate roots detection
    codex_root = base_root / "VIVTransformer-4sh2r1-codex"
    dated_root = Path(dated_root_override) if dated_root_override else codex_root / "modify_multi_attention" / "attention_results"
    
    # Enhanced loss_cfg_root detection - check multiple possible locations
    if loss_cfg_root_override:
        loss_cfg_root = Path(loss_cfg_root_override)
    else:
        # Try multiple possible locations for loss_config directories
        possible_loss_roots = [
            codex_root / "attention_results",
            base_root / "attention_results", 
            base_root,  # loss_config_* might be directly under base_root
            codex_root / "modify_multi_attention" / "attention_results"
        ]
        loss_cfg_root = None
        for possible_root in possible_loss_roots:
            if possible_root.exists() and any(p.name.startswith("loss_config_") for p in possible_root.iterdir() if p.is_dir()):
                loss_cfg_root = possible_root
                print(f"Debug: Found loss_config root at: {loss_cfg_root}")
                break
        
        if loss_cfg_root is None:
            loss_cfg_root = codex_root / "attention_results"  # fallback

    # 1) Dated runs under modify_multi_attention/attention_results
    # Support multiple layouts: nested, flat, and mixed
    if dated_root.exists():
        children = [p for p in dated_root.iterdir() if p.is_dir()]
        for child in sorted(children):
            # Enhanced experiment directory detection
            if is_experiment_directory(child):
                # Direct experiment directory (flat layout)
                attn = child.name
                entry = collect_single_run(child, attn, source_group=f"dated:{dated_root.name}")
                entries.append(entry)
            else:
                # Check subdirectories for experiments (nested layout)
                subdirs = [p for p in child.iterdir() if p.is_dir()]
                for attn_dir in sorted(subdirs):
                    if is_experiment_directory(attn_dir):
                        attn = attn_dir.name
                        entry = collect_single_run(attn_dir, attn, source_group=f"dated:{child.name}")
                        entries.append(entry)
                    else:
                        # Check one more level deep for deeply nested structures
                        deep_subdirs = [p for p in attn_dir.iterdir() if p.is_dir()]
                        for deep_dir in sorted(deep_subdirs):
                            if is_experiment_directory(deep_dir):
                                attn = deep_dir.name
                                entry = collect_single_run(deep_dir, attn, source_group=f"dated:{child.name}/{attn_dir.name}")
                                entries.append(entry)

    # 2) Enhanced loss_config_* runs scanning
    if loss_cfg_root and loss_cfg_root.exists():
        print(f"Debug: Scanning loss_cfg_root: {loss_cfg_root}")
        loss_config_dirs = [p for p in loss_cfg_root.iterdir() if p.is_dir() and p.name.startswith("loss_config_")]
        print(f"Debug: Found {len(loss_config_dirs)} loss_config directories: {[d.name for d in loss_config_dirs]}")
        
        for cfg_dir in sorted(loss_config_dirs):
            print(f"Debug: Processing loss_config directory: {cfg_dir.name}")
            
            # Check if cfg_dir itself contains experiment files (flat structure)
            if is_experiment_directory(cfg_dir):
                attn = cfg_dir.name.replace("loss_config_", "")
                print(f"Debug: Found flat experiment in {cfg_dir.name} with attention: {attn}")
                entry = collect_single_run(cfg_dir, attn, source_group=cfg_dir.name)
                entry["loss_config_id"] = cfg_dir.name
                entries.append(entry)
                continue
            
            # Enhanced scanning for nested structures
            subdirs = [p for p in cfg_dir.iterdir() if p.is_dir()]
            print(f"Debug: Found {len(subdirs)} subdirectories in {cfg_dir.name}: {[d.name for d in subdirs]}")
            
            for attn_dir in sorted(subdirs):
                if is_experiment_directory(attn_dir):
                    attn = attn_dir.name
                    print(f"Debug: Found experiment directory: {attn} in {cfg_dir.name}")
                    entry = collect_single_run(attn_dir, attn, source_group=cfg_dir.name)
                    entry["loss_config_id"] = cfg_dir.name
                    entries.append(entry)
                else:
                    # Check one level deeper for deeply nested structures
                    deep_subdirs = [p for p in attn_dir.iterdir() if p.is_dir()]
                    for deep_dir in sorted(deep_subdirs):
                        if is_experiment_directory(deep_dir):
                            attn = deep_dir.name
                            print(f"Debug: Found deep experiment directory: {attn} in {cfg_dir.name}/{attn_dir.name}")
                            entry = collect_single_run(deep_dir, attn, source_group=f"{cfg_dir.name}/{attn_dir.name}")
                            entry["loss_config_id"] = cfg_dir.name
                            entries.append(entry)
    else:
        print(f"Debug: loss_cfg_root does not exist or is None: {loss_cfg_root}")

    return entries


def find_file_in_multiple_locations(attn_dir: Path, relative_paths: List[str]) -> Optional[Path]:
    """Find a file in multiple possible locations within the experiment directory"""
    for rel_path in relative_paths:
        full_path = attn_dir / rel_path
        if full_path.exists():
            return full_path
    return None


def collect_single_run(attn_dir: Path, attn: str, source_group: str) -> Dict[str, Any]:
    # Enhanced file location logic - check multiple possible locations
    
    # Loss log file locations
    loss_log_paths = [
        "loss_logs/loss_log.txt",
        "loss_log.txt",
        "logs/loss_log.txt",
        "training_logs/loss_log.txt"
    ]
    loss_log = find_file_in_multiple_locations(attn_dir, loss_log_paths)
    
    # Test result file locations
    test_result = None
    test_result_patterns = [
        f"test_result_{attn}.txt",
        "test_result.txt",
        f"test_results/test_result_{attn}.txt",
        "test_results/test_result.txt",
        f"results/test_result_{attn}.txt",
        "results/test_result.txt"
    ]
    
    # Try specific patterns first, then glob patterns
    for pattern in test_result_patterns:
        test_file = attn_dir / pattern
        if test_file.exists():
            test_result = parse_test_result_file(test_file)
            break
    
    # If no specific file found, try glob patterns
    if test_result is None:
        for f in attn_dir.glob("test_result_*.txt"):
            test_result = parse_test_result_file(f)
            break
        if test_result is None:
            for f in attn_dir.glob("**/test_result_*.txt"):
                test_result = parse_test_result_file(f)
                break
    
    # Test loss log file locations
    test_loss_log_paths = [
        "test_results/test_loss_log.txt",
        "test_loss_log.txt",
        "logs/test_loss_log.txt",
        "results/test_loss_log.txt"
    ]
    test_loss_log = find_file_in_multiple_locations(attn_dir, test_loss_log_paths)

    loss_stats = parse_loss_log(loss_log) if loss_log else {"training_status": "no_log_file"}
    test_stats = parse_test_loss_log(test_loss_log) if test_loss_log else {}
    
    # Analyze training convergence from loss log
    convergence_analysis = analyze_training_convergence(loss_log) if loss_log else None

    # research_data might be under attn_dir/research_data or directly under attn_dir
    # Try both locations to ensure we find the data files
    research_dir = attn_dir / "research_data"
    if not research_dir.exists() or not any(research_dir.glob("*.json")):
        research_dir = attn_dir  # Fallback to attn_dir itself
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
    
    # Configuration info from YAML files
    if research_stats.get("config_info"):
        config_info = research_stats["config_info"]
        flattened_data.update({
            "config_device": config_info.get("device"),
            "config_batch_size": config_info.get("batch_size"),
            "config_attention_type": config_info.get("model_config", {}).get("attention_type"),
            "config_d_model": config_info.get("model_config", {}).get("d_model"),
            "config_num_heads": config_info.get("model_config", {}).get("num_heads"),
            "config_num_layers": config_info.get("model_config", {}).get("num_layers"),
            "config_base_weight": config_info.get("loss_config", {}).get("base_weight"),
            "config_svd_weights": str(config_info.get("loss_config", {}).get("svd_weights", [])),
            "config_topk": config_info.get("loss_config", {}).get("topk"),
            "config_gpu_monitoring": config_info.get("hardware_monitoring", {}).get("enable_gpu_monitoring"),
            "config_monitor_temperature": config_info.get("hardware_monitoring", {}).get("monitor_temperature"),
            "config_monitor_power": config_info.get("hardware_monitoring", {}).get("monitor_power_usage")
        })
        # expose seed if present (common keys: seed / random_seed / training.seed)
        try:
            flattened_data["random_seed"] = (
                config_info.get("seed")
                or config_info.get("random_seed")
                or (config_info.get("training", {}).get("seed") if isinstance(config_info.get("training"), dict) else None)
            )
        except Exception:
            pass

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
    
    # Ensure training_status is set correctly (after flattened_data to avoid override)
    collected_data["training_status"] = loss_stats.get("training_status", "unknown")
    
    # Map final_test_loss_collector to final_test_loss if available
    if research_stats.get("final_test_loss_collector") is not None and collected_data.get("final_test_loss") is None:
        collected_data["final_test_loss"] = research_stats["final_test_loss_collector"]

    # New: map derived best_valid_* and prefer training_epochs to fill missing fields without overriding valid values
    if collected_data.get("best_valid_loss") is None and research_stats.get("derived_best_valid_loss") is not None:
        collected_data["best_valid_loss"] = research_stats["derived_best_valid_loss"]
    if collected_data.get("best_valid_epoch") is None and research_stats.get("derived_best_valid_epoch") is not None:
        collected_data["best_valid_epoch"] = research_stats["derived_best_valid_epoch"]
    # Prefer derived training_epochs when epochs is missing or zero
    _epochs_current = collected_data.get("epochs")
    if (_epochs_current is None or _epochs_current == 0) and research_stats.get("training_epochs") is not None:
        collected_data["epochs"] = research_stats["training_epochs"]
    # If training_status unknown or no log, infer from epochs when possible
    _ts = collected_data.get("training_status")
    _epochs_final = collected_data.get("epochs") or 0
    if _ts in (None, "unknown", "no_log_file", "not_started"):
        if _epochs_final >= 2:
            collected_data["training_status"] = "completed"
        elif _epochs_final == 1:
            collected_data["training_status"] = "incomplete"
    
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
        "training_status",  # New field to track training completion status
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
        "random_seed",
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
    single_run_groups = 0
    
    for group_key, runs in attention_groups.items():
        total_groups += 1
        attn_name = group_key.split('_')[0]
        
        if len(runs) < 2:
            # Single run - assume consistent but note it
            single_run_groups += 1
            reproducibility["attention_consistency"][attn_name] = {
                "runs": 1,
                "mean_loss": runs[0].get("avg_test_loss"),
                "std_loss": 0.0,
                "cv": 0.0,
                "is_consistent": True,
                "note": "单次运行，无法验证一致性"
            }
            consistent_groups += 1
            continue
        
        # Performance consistency
        test_losses = [r.get("avg_test_loss") for r in runs if r.get("avg_test_loss") is not None]
        if len(test_losses) >= 2:
            mean_val = sum(test_losses) / len(test_losses) if len(test_losses) > 0 else 0.0
            denom = mean_val if mean_val != 0 else 1.0
            cv = statistics.stdev(test_losses) / denom  # Coefficient of variation
            
            reproducibility["attention_consistency"][attn_name] = {
                "runs": len(test_losses),
                "mean_loss": mean_val,
                "std_loss": statistics.stdev(test_losses) if len(test_losses) > 1 else 0.0,
                "cv": cv,
                "is_consistent": cv < 0.1  # CV < 10% considered consistent
            }
            
            if cv < 0.1:
                consistent_groups += 1
        
        # Configuration consistency (use actual keys)
        configs = {}
        for run in runs:
            for key in ['initial_lr', 'optimizer_type', 'model_parameters']:
                if run.get(key) is not None:
                    if key not in configs:
                        configs[key] = []
                    configs[key].append(run.get(key))
        
        config_consistent = True
        if len(runs) > 1:  # Only check consistency for multiple runs
            for key, values in configs.items():
                if len(set(str(v) for v in values)) > 1:  # More than one unique value
                    config_consistent = False
                    reproducibility["issues"].append(f"{attn_name}: 配置不一致 - {key}")
        
        reproducibility["config_consistency"][attn_name] = config_consistent
        
        # Environment consistency (use keys written by system_info)
        env_keys = ['torch_version', 'cuda_version', 'gpu_name']
        env_consistent = True
        if len(runs) > 1:  # Only check consistency for multiple runs
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

    # 数据质量评估
    quality_report = generate_data_quality_report(entries)
    lines.append("\n## 数据质量评估\n")
    lines.append(f"- 实验总数: {quality_report['total_experiments']}\n")
    lines.append(f"- 平均质量评分: {quality_report['average_quality_score']:.2f}/10\n")
    if quality_report.get('common_issues'):
        lines.append("### 常见问题（Top-3）\n")
        _issues_sorted = sorted(quality_report['common_issues'].items(), key=lambda x: x[1], reverse=True)[:3]
        for k, v in _issues_sorted:
            lines.append(f"- {k}: {v} 次\n")

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


def validate_experiment_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean a single experiment entry, adding fallbacks where possible."""
    cleaned = dict(data)
    # Normalize numeric fields
    for k in [
        "final_train_loss", "final_valid_loss", "final_test_loss",
        "best_valid_loss", "best_valid_epoch", "epochs",
        "avg_test_loss", "min_test_loss", "max_test_loss",
        "total_duration_seconds",
        "avg_gpu_utilization", "max_gpu_memory_mb", "avg_cpu_percent", "max_ram_mb", "gpu_temperature_max",
        "initial_lr", "final_lr", "min_lr", "max_lr",
        "weight_decay", "momentum", "beta1", "beta2", "eps",
        "loss_variance", "gradient_norm_avg", "gradient_norm_max",
        "early_stopping_epoch", "plateau_epochs",
        "train_loss_variance_late", "valid_loss_variance_late",
        "epochs_after_best_valid", "overfitting_ratio", "loss_decreasing_ratio",
        "convergence_epoch_90pct",
    ]:
        if k in cleaned and cleaned[k] is not None:
            val = safe_float(cleaned[k])
            cleaned[k] = val if val is not None else cleaned[k]

    # Fallback: fill final_test_loss from test_result_file_loss or avg_test_loss
    ftl = cleaned.get("final_test_loss")
    if ftl is None or not isinstance(ftl, (int, float)) or not math.isfinite(float(ftl)):
        tr = cleaned.get("test_result_file_loss")
        tr_val = safe_float(tr)
        if tr_val is not None:
            cleaned["final_test_loss"] = tr_val
        elif cleaned.get("avg_test_loss") is not None:
            avg_val = safe_float(cleaned.get("avg_test_loss"))
            if avg_val is not None:
                cleaned["final_test_loss"] = avg_val

    # Fallback: best_valid_loss from derived fields if still missing
    if cleaned.get("best_valid_loss") is None:
        d = cleaned.get("derived_best_valid_loss") or cleaned.get("best_val_loss")
        d_val = safe_float(d)
        if d_val is not None:
            cleaned["best_valid_loss"] = d_val
    if cleaned.get("best_valid_epoch") is None:
        be = cleaned.get("derived_best_valid_epoch")
        be_val = safe_float(be)
        if be_val is not None:
            cleaned["best_valid_epoch"] = int(be_val)

    # NEW: Fallback from nested convergence_metrics in comprehensive metrics
    if cleaned.get("best_valid_loss") is None:
        cm = cleaned.get("convergence_metrics")
        if isinstance(cm, dict):
            cm_bvl = safe_float(cm.get("best_valid_loss") or cm.get("best_val_loss"))
            if cm_bvl is not None:
                cleaned["best_valid_loss"] = cm_bvl
            if cleaned.get("best_valid_epoch") is None:
                cm_be = safe_float(cm.get("best_valid_epoch") or cm.get("best_val_epoch"))
                if cm_be is not None:
                    cleaned["best_valid_epoch"] = int(cm_be)

    # Last-resort fallback: use final_valid_loss and epochs as proxies
    if cleaned.get("best_valid_loss") is None:
        fvl = safe_float(cleaned.get("final_valid_loss"))
        if fvl is not None:
            cleaned["best_valid_loss"] = fvl
            if cleaned.get("best_valid_epoch") is None:
                ep = safe_float(cleaned.get("epochs"))
                if ep is not None:
                    cleaned["best_valid_epoch"] = int(ep)

    # Fallback: epochs
    if not cleaned.get("epochs"):
        te = safe_float(cleaned.get("training_epochs"))
        if te is not None:
            cleaned["epochs"] = int(te)
    # Try to recover epochs from nested convergence_metrics if still missing
    if not cleaned.get("epochs"):
        cm = cleaned.get("convergence_metrics")
        if isinstance(cm, dict):
            te2 = safe_float(cm.get("total_epochs") or cm.get("epochs"))
            if te2 is not None:
                cleaned["epochs"] = int(te2)

    # Training status normalization
    ts = cleaned.get("training_status")
    if ts not in {"ok", "completed", "in_progress", "failed", "no_log_file", "unknown", "not_started"}:
        cleaned["training_status"] = "ok"

    return cleaned


def generate_data_quality_report(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary of data completeness and quality across entries."""
    total = len(entries)
    issues: Dict[str, int] = {}

    def inc(key: str):
        issues[key] = issues.get(key, 0) + 1

    for e in entries:
        if e.get("final_test_loss") is None:
            inc("Missing final_test_loss")
        if e.get("best_valid_loss") is None:
            inc("Missing best_valid_loss")
        if not e.get("epochs"):
            inc("Missing epochs")
        if e.get("total_duration_seconds") is None:
            inc("Missing total_duration_seconds")
        if e.get("model_parameters") is None:
            inc("Missing model_parameters")
        if e.get("avg_gpu_utilization") is None:
            inc("Missing avg_gpu_utilization")
        # Unusual epochs
        if isinstance(e.get("epochs"), int) and e.get("epochs") is not None and e.get("epochs") <= 1:
            inc("Unusual epochs value")

    # Quality scoring: simple heuristic
    quality_scores: List[float] = []
    for e in entries:
        score = 10.0
        if e.get("final_test_loss") is None:
            score -= 2.0
        if e.get("best_valid_loss") is None:
            score -= 1.5
        if not e.get("epochs"):
            score -= 1.0
        if e.get("total_duration_seconds") is None:
            score -= 1.0
        if e.get("model_parameters") is None:
            score -= 0.5
        if e.get("avg_gpu_utilization") is None:
            score -= 0.5
        quality_scores.append(max(0.0, score))

    avg_score = sum(quality_scores) / total if total > 0 else 0.0

    # Distribution by status
    dist: Dict[str, int] = {}
    for e in entries:
        st = e.get("status", "unknown")
        dist[st] = dist.get(st, 0) + 1

    recs: List[str] = []
    if issues.get("Missing final_test_loss", 0) > 0:
        recs.append("确保在测试阶段保存 final_test_loss（或在分析脚本中启用回退提取）")
    if issues.get("Unusual epochs value", 0) > 0:
        recs.append("检查训练是否提前停止或日志是否不完整（epochs 值异常低）")
    if issues.get("Missing avg_gpu_utilization", 0) > 0:
        recs.append("启用硬件监控或保证 research_data/hardware_metrics.json 正确生成")

    return {
        "total_experiments": total,
        "average_quality_score": round(avg_score, 2),
        "quality_distribution": dist,
        "common_issues": dict(sorted(issues.items(), key=lambda x: x[1], reverse=True)),
        "recommendations": recs,
    }


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
    
    # Generate data quality report
    quality_report = generate_data_quality_report(entries)
    quality_report_file = report_dir / "data_quality_report.json"
    
    # Write quality report to JSON file
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(quality_report_file, 'w', encoding='utf-8') as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)

    print(f"Scanning base_root: {base_root}")
    if dated_root_override:
        print(f"Using dated_root override: {dated_root_override}")
    if loss_cfg_root_override:
        print(f"Using loss_cfg_root override: {loss_cfg_root_override}")
    if args.config_file:
        print(f"Using config_file: {Path(args.config_file).resolve()}")
    print(f"Summary CSV: {out_csv}")
    print(f"Markdown Report: {out_md}")
    print(f"Data Quality Report: {quality_report_file}")
    print(f"Total entries: {len(entries)}")
    
    # Print quality summary to console
    print(f"\n=== 数据质量概览 ===")
    print(f"总实验数量: {quality_report['total_experiments']}")
    print(f"数据质量分布:")
    for quality, count in quality_report['quality_distribution'].items():
        percentage = (count / quality_report['total_experiments']) * 100 if quality_report['total_experiments'] > 0 else 0
        print(f"  {quality}: {count} ({percentage:.1f}%)")
    print(f"平均质量评分: {quality_report['average_quality_score']:.2f}/10")
    
    if quality_report['common_issues']:
        print(f"\n主要数据问题:")
        for issue, count in list(quality_report['common_issues'].items())[:3]:
            print(f"  - {issue}: {count}次")
    
    if quality_report['recommendations']:
        print(f"\n改进建议:")
        for rec in quality_report['recommendations'][:2]:
            print(f"  - {rec}")


if __name__ == "__main__":
    main()
