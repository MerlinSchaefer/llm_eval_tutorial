from __future__ import annotations

import datetime as dt
import json
from typing import List, Optional

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# -- Globals (kept minimal) --
_client: Optional[MlflowClient] = None
_experiment_id: Optional[str] = None


def _client_ok() -> MlflowClient:
    global _client
    if _client is None:
        _client = MlflowClient()
    return _client


def set_tracking_uri(uri: str) -> None:
    """Convenience wrapper; also resets the client cache."""
    global _client
    mlflow.set_tracking_uri(uri)
    _client = None  # reset to pick up new URI


def set_experiment(name: str) -> str:
    """Set experiment by name; creates it if missing. Returns experiment_id."""
    global _experiment_id
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        mlflow.set_experiment(name)  # creates
        exp = mlflow.get_experiment_by_name(name)
    assert exp is not None
    _experiment_id = exp.experiment_id
    return _experiment_id


def get_experiment_id() -> Optional[str]:
    return _experiment_id


# ---------- Listing helpers ----------


def list_experiments_df() -> pd.DataFrame:
    """Return experiments as a DataFrame."""
    client = _client_ok()
    rows = []
    for e in client.search_experiments():
        rows.append(
            {
                "name": e.name,
                "experiment_id": e.experiment_id,
                "lifecycle_stage": e.lifecycle_stage,
                "artifact_location": e.artifact_location,
            }
        )
    return pd.DataFrame(rows).sort_values("name")


def search_runs_df(
    experiment_id: Optional[str] = None, max_results: int = 50
) -> pd.DataFrame:
    """Return recent runs for an experiment as a DataFrame (most recent first)."""
    client = _client_ok()
    exp_id = experiment_id or _experiment_id
    if not exp_id:
        raise ValueError(
            "No experiment_id provided and none set. Call set_experiment(name) or pass experiment_id."
        )
    runs = client.search_runs(
        [exp_id],
        filter_string="",
        run_view_type=mlflow.entities.ViewType.ALL,
        max_results=max_results,
        order_by=["attributes.start_time DESC"],
    )
    rows = []
    for r in runs:
        info, data = r.info, r.data
        start = (
            dt.datetime.fromtimestamp(info.start_time / 1000.0)
            if info.start_time
            else None
        )
        # _end = (
        #     dt.datetime.fromtimestamp(info.end_time / 1000.0) if info.end_time else None
        # )
        dur_ms = (
            (info.end_time - info.start_time)
            if (info.end_time and info.start_time)
            else None
        )
        p, t = data.params, data.tags
        rows.append(
            {
                "run_name": t.get("mlflow.runName", ""),
                "run_id": info.run_id,
                "status": info.status,
                "start": start,
                "duration_s": None if dur_ms is None else round(dur_ms / 1000.0, 3),
                "model": p.get("pydantic_ai.model_name")
                or t.get("pydantic_ai.model_name"),
                "provider": p.get("pydantic_ai.provider")
                or t.get("pydantic_ai.provider"),
                "prompt_preview": (
                    p.get("input")
                    or t.get("pydantic_ai.prompt")
                    or t.get("mlflow.prompt")
                    or ""
                )[:160],
            }
        )
    return pd.DataFrame(rows)


# ---------- Run inspectors ----------


def show_run_details(run_id: str) -> None:
    """Print params, metrics, tags, and top-level artifacts for a run."""
    client = _client_ok()
    run = client.get_run(run_id)
    info, data = run.info, run.data

    print("=== RUN INFO ===")
    print("run_id:", info.run_id)
    print("status:", info.status)
    print(
        "start:",
        dt.datetime.fromtimestamp(info.start_time / 1000.0)
        if info.start_time
        else None,
    )
    print(
        "end  :",
        dt.datetime.fromtimestamp(info.end_time / 1000.0) if info.end_time else None,
    )

    print("\n=== PARAMS (first 30) ===")
    for k, v in list(data.params.items())[:30]:
        print(f"{k}: {v}")

    print("\n=== METRICS (first 30) ===")
    for k, v in list(data.metrics.items())[:30]:
        print(f"{k}: {v}")

    print("\n=== TAGS (first 30) ===")
    for k, v in list(data.tags.items())[:30]:
        print(f"{k}: {v}")

    print("\n=== ARTIFACTS (top level) ===")
    arts = client.list_artifacts(run_id, "")
    for a in arts:
        print(("- [DIR] " if a.is_dir else "- [FILE] ") + a.path)


# ---------- Trace artifact helpers ----------


def _list_artifacts_recursive(run_id: str, path: str = "") -> List[str]:
    client = _client_ok()
    out: List[str] = []
    for a in client.list_artifacts(run_id, path):
        if a.is_dir:
            out.extend(_list_artifacts_recursive(run_id, a.path))
        else:
            out.append(a.path)
    return out


def _load_json_artifact(run_id: str, path: str):
    client = _client_ok()
    local_path = client.download_artifacts(run_id, path)
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_trace_json_path(run_id: str) -> Optional[str]:
    # Heuristics: prefer files with "trace" in their name; else any JSON
    paths = _list_artifacts_recursive(run_id, "")
    candidates = [
        p for p in paths if p.lower().endswith(".json") and "trace" in p.lower()
    ]
    if not candidates:
        candidates = [p for p in paths if p.lower().endswith(".json")]
    return candidates[0] if candidates else None


def _print_trace_tree(obj, indent: int = 0) -> None:
    pad = "  " * indent
    if isinstance(obj, dict):
        name = obj.get("name") or obj.get("span_name") or obj.get("operation_name")
        typ = obj.get("span_type") or obj.get("type")
        if name or typ:
            print(f"{pad}• {name or '<span>'} [{typ or ''}]")
        # children-like keys
        for key in ("children", "spans", "events", "nodes"):
            if key in obj and isinstance(obj[key], list):
                for child in obj[key]:
                    _print_trace_tree(child, indent + 1)
        # short inputs/outputs/attributes
        for key in ("inputs", "outputs", "attributes"):
            if key in obj:
                print(f"{pad}{key}:")
                _print_trace_tree(obj[key], indent + 1)
    elif isinstance(obj, list):
        for item in obj:
            _print_trace_tree(item, indent)
    else:
        s = str(obj)
        print(f"{pad}{s[:160]}")


def show_trace_if_any(run_id: str) -> None:
    """Try to find a JSON trace artifact for run_id and pretty-print a tree."""
    path = _find_trace_json_path(run_id)
    if not path:
        print("No JSON trace artifact found.")
        return
    print(f"Trace artifact: {path}\n")
    data = _load_json_artifact(run_id, path)
    _print_trace_tree(data)


def llm_snapshot(run_id: str) -> None:
    """Print a compact view of model, provider, prompt & output (if logged)."""
    client = _client_ok()
    run = client.get_run(run_id)
    p, t = run.data.params, run.data.tags

    likely_prompt = (
        p.get("input") or t.get("mlflow.prompt") or t.get("pydantic_ai.prompt")
    )
    likely_output = (
        p.get("output") or t.get("mlflow.output") or t.get("pydantic_ai.output")
    )

    print("Model:", p.get("pydantic_ai.model_name") or t.get("pydantic_ai.model_name"))
    print("Provider:", p.get("pydantic_ai.provider") or t.get("pydantic_ai.provider"))
    print("\n--- PROMPT (first 800 chars) ---\n", (likely_prompt or "")[:800])
    print("\n--- OUTPUT (first 800 chars) ---\n", (likely_output or "")[:800])


def show_last_trace(experiment_name: Optional[str] = None) -> None:
    """Convenience: find the most recent finished run and print its trace."""
    client = _client_ok()
    exp_id = None
    if experiment_name:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp:
            print(f"Experiment '{experiment_name}' not found.")
            return
        exp_id = exp.experiment_id
    else:
        exp_id = _experiment_id
        if not exp_id:
            print(
                "No experiment set. Call set_experiment(name) or pass experiment_name."
            )
            return

    runs = client.search_runs(
        [exp_id], order_by=["attributes.start_time DESC"], max_results=1
    )
    if not runs:
        print("No runs found.")
        return
    run_id = runs[0].info.run_id
    print(f"Latest run_id: {run_id}\n")
    show_trace_if_any(run_id)


def _require_exp_id(explicit_id=None):
    exp_id = explicit_id or get_experiment_id()
    if not exp_id:
        raise ValueError(
            "No experiment set. Call mv.set_experiment('duckdb-rag') or pass experiment_id=..."
        )
    return exp_id


def search_traces_df(
    experiment_id=None,
    max_results=200,
    order_desc=True,
    extract_fields=None,
    filter_string=None,
):
    """
    List traces for an experiment as a DataFrame (new MLflow Tracing API).
    Useful columns: request_id, status, execution_time_ms, request, response, trace
    """
    exp_id = _require_exp_id(experiment_id)
    order_by = ["timestamp_ms DESC"] if order_desc else ["timestamp_ms ASC"]
    df = mlflow.search_traces(
        experiment_ids=[exp_id],
        max_results=max_results,
        order_by=order_by,
        extract_fields=extract_fields,  # e.g. ["rag.ask.inputs", "rag.ask.outputs"]
        filter_string=filter_string,
    )
    return df


def latest_trace_row(experiment_id=None):
    df = search_traces_df(experiment_id=experiment_id, max_results=1)
    return None if df.empty else df.iloc[0]


def _to_dict_safe(obj):
    if obj is None:
        return {}
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if isinstance(obj, dict):
        return obj
    try:
        return json.loads(
            json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o)))
        )
    except Exception:
        return {"value": str(obj)}


def _print_span_tree(spans, indent=0):
    pad = "  " * indent
    for s in spans or []:
        name = s.get("name") or s.get("attributes", {}).get("name") or "<span>"
        typ = s.get("span_type") or s.get("attributes", {}).get("span_type", "")
        print(f"{pad}• {name} [{typ}]")
        # children could be under 'children' or nested in 'spans'
        if "children" in s and isinstance(s["children"], list):
            _print_span_tree(s["children"], indent + 1)
        elif "spans" in s and isinstance(s["spans"], list):
            _print_span_tree(s["spans"], indent + 1)


def show_latest_trace_tree(experiment_name=None):
    """
    Pretty-prints the most recent trace (as a span tree) for the given experiment
    or the one set via mv.set_experiment(...).
    """
    exp_id = None
    if experiment_name:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp:
            print(f"Experiment '{experiment_name}' not found.")
            return
        exp_id = exp.experiment_id

    row = latest_trace_row(experiment_id=exp_id)
    if row is None:
        print(
            "No traces found. Make at least one agent call with mlflow.pydantic_ai.autolog() enabled."
        )
        return

    trace_obj = _to_dict_safe(row.get("trace"))
    spans = trace_obj.get("spans") or trace_obj.get("children") or []
    print(
        f"Trace request_id: {row.get('request_id')}, status: {row.get('status')}, "
        f"exec_ms: {row.get('execution_time_ms')}"
    )
    _print_span_tree(spans)
