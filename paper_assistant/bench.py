"""Run the figure agent (or the one-shot ablation) on the task suite in tasks/figure_tasks.json.

A figure counts as clean when its script ran, it saved a vector file, and the automatic
checks found no issues. Clean is necessary, not sufficient: look at the figures too
(each task's folder under runs/ keeps the script, the PDF and the preview).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .figure_agent import FigureRun, make_figure, make_figure_oneshot
from .llm import USAGE_FIELDS, estimate_cost
from .workspace import Workspace

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


def load_tasks(names: list[str] | None = None) -> list[dict]:
    tasks = json.loads((TASKS_DIR / "figure_tasks.json").read_text(encoding="utf-8"))
    return [t for t in tasks if not names or t["name"] in names]


def run_bench(llm, model: str, method: str, out: Path, names: list[str] | None = None,
              limit: int | None = None, max_tool_calls: int = 20) -> str:
    tasks = load_tasks(names)[:limit]
    out_dir = Path(out) / f"{method}-{model}-{time.strftime('%Y%m%d-%H%M%S')}"
    rows = []
    for i, task in enumerate(tasks, 1):
        ws = Workspace(out_dir / task["name"], [TASKS_DIR / "data" / d for d in task["data"]])
        start = time.time()
        if method == "agent":
            run = make_figure(llm, ws, task["task"], max_tool_calls)
        else:
            run = make_figure_oneshot(llm, ws, task["task"])
        row = _row(task["name"], run, model, time.time() - start)
        rows.append(row)
        (ws.root / "_result.json").write_text(json.dumps({**row, "caption": run.caption, "latex": run.latex,
                                                          "transcript": run.transcript}, indent=2), encoding="utf-8")
        mark = "clean " if row["clean"] else "ISSUES"
        print(f"[{i}/{len(tasks)}] {mark} {task['name']:<22} runs={row['runs']} views={row['views']} "
              f"issues={row['n_issues']}  ${row['cost_usd'] or 0:.3f}")
    report = _report(rows, f"{method}, {model}")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return f"{report}\n\nSaved to {out_dir}"


def _row(name: str, run: FigureRun, model: str, seconds: float) -> dict:
    last = run.last_run
    usage = {f: run.usage[f] for f in USAGE_FIELDS}
    ran = last is not None and last.exit_code == 0
    issues = last.issues if last else ["no script was run"]
    return {
        "task": name,
        "finished": run.finished,
        "ran": ran,
        "vector": bool(last and last.saved_vector),
        "issues": issues,
        "n_issues": len(issues),
        "clean": run.finished and ran and bool(last and last.figures) and not issues,
        "runs": run.tool_calls.count("run_python") or (1 if last else 0),
        "views": run.tool_calls.count("view_image"),
        "tool_calls": len(run.tool_calls),
        "usage": usage,
        "cost_usd": estimate_cost(model, usage),
        "seconds": round(seconds, 1),
    }


def _report(rows: list[dict], tag: str) -> str:
    n = len(rows)
    clean = sum(r["clean"] for r in rows)
    lines = [
        f"## Figure tasks: {tag}",
        "",
        f"Clean figures (ran, saved a vector file, no check issues): {clean}/{n}",
        f"Mean script runs: {sum(r['runs'] for r in rows) / n:.1f}; mean previews viewed: {sum(r['views'] for r in rows) / n:.1f}; "
        f"approx. cost: ${sum(r['cost_usd'] or 0 for r in rows):.2f}",
        "",
        "| task | clean | runs | views | remaining issues |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['task']} | {'yes' if r['clean'] else 'no'} | {r['runs']} | {r['views']} | {'; '.join(r['issues']) or '-'} |")
    return "\n".join(lines)
