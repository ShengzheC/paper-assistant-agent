"""Command line.

    python -m paper_assistant figure "Phase portrait of x'' + 0.3x' + x = 0 ..." --out figs/oscillator
    python -m paper_assistant figure "Plot accuracy vs. epoch ..." --data results.csv --out figs/curves
    python -m paper_assistant polish intro.tex
    python -m paper_assistant bench --limit 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bench import run_bench
from .figure_agent import make_figure
from .llm import DEFAULT_MODEL, USAGE_FIELDS, ClaudeLLM, estimate_cost
from .polish import polish, word_diff
from .workspace import Workspace


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(errors="replace")  # consoles with legacy code pages cannot print every character
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=DEFAULT_MODEL)
    common.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])

    parser = argparse.ArgumentParser(prog="python -m paper_assistant", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fig = sub.add_parser("figure", parents=[common], help="make a figure or toy example from a description")
    fig.add_argument("task", help="what the figure should show")
    fig.add_argument("--out", required=True, help="output folder (the agent's workspace)")
    fig.add_argument("--data", nargs="*", default=[], help="data files to copy into the workspace")
    fig.add_argument("--max-tool-calls", type=int, default=20)
    fig.add_argument("--verbose", action="store_true", help="print each step")
    pol = sub.add_parser("polish", parents=[common], help="polish a LaTeX paragraph (math and citations stay untouched)")
    pol.add_argument("file")
    pol.add_argument("--out", help="where to write the revision (default: <file>.polished.tex)")
    pol.add_argument("--instructions", default="", help="extra guidance, e.g. 'shorten to 120 words'")
    bench = sub.add_parser("bench", parents=[common], help="run the figure task suite")
    bench.add_argument("--method", choices=["agent", "oneshot"], default="agent")
    bench.add_argument("--task", action="append", help="only run this task (repeatable)")
    bench.add_argument("--limit", type=int, default=None)
    bench.add_argument("--max-tool-calls", type=int, default=20)
    bench.add_argument("--out", default="runs")
    args = parser.parse_args(argv)

    llm = ClaudeLLM(args.model, args.effort)
    if args.command == "figure":
        ws = Workspace(args.out, [Path(d) for d in args.data])
        run = make_figure(llm, ws, args.task, args.max_tool_calls)
        if args.verbose:
            _print_transcript(run.transcript)
        usage = {f: run.usage[f] for f in USAGE_FIELDS}
        (ws.root / "_transcript.json").write_text(json.dumps(run.transcript, indent=2), encoding="utf-8")
        print(f"\nstop: {run.stop}; tool calls: {len(run.tool_calls)}; approx. cost: ${estimate_cost(args.model, usage) or 0:.3f}")
        if run.last_run:
            print("automatic checks: " + ("passed" if not run.last_run.issues else "; ".join(run.last_run.issues)))
        if run.finished:
            print(f"files: {', '.join(run.files)} (in {ws.root})\n\ncaption: {run.caption}\n\n{run.latex}")
    elif args.command == "polish":
        source = Path(args.file)
        result = polish(llm, source.read_text(encoding="utf-8"), args.instructions)
        target = Path(args.out) if args.out else source.with_suffix(".polished.tex")
        target.write_text(result.revised, encoding="utf-8")
        print(word_diff(result.original, result.revised))
        print("\nchanges:\n- " + "\n- ".join(result.changes))
        status = "all math, citations and references unchanged" if result.ok else "WARNING: " + "; ".join(result.problems)
        print(f"\ncheck: {status} (attempts: {result.attempts})\nwrote {target}")
    else:
        print(run_bench(llm, args.model, args.method, Path(args.out), args.task, args.limit, args.max_tool_calls))


def _print_transcript(transcript: list[dict]) -> None:
    for entry in transcript:
        if "thinking" in entry:
            print(f"  [thinking] {entry['thinking'].strip()}")
        elif "text" in entry:
            print(f"  [{entry['role']}] {entry['text'].strip()[:1500]}")
        elif "tool_call" in entry:
            shown = {k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v) for k, v in entry["input"].items()}
            print(f"  [call] {entry['tool_call']}({json.dumps(shown)})")
        else:
            body = entry["output"].splitlines()
            print("  [result]\n    " + "\n    ".join(body[:20] + ([f"... ({len(body) - 20} more lines)"] if len(body) > 20 else [])))


if __name__ == "__main__":
    main()
