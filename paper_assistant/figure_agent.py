"""The figure agent: write a script, run it, read the checks, look at the preview, revise, finish.

The loop is written by hand so each step is visible: a tool budget, one reminder if the model
replies in prose, validation of the final answer, and a transcript. `make_figure_oneshot` is
the ablation without feedback: one script, run once.
"""

from __future__ import annotations

import base64
import io
from collections import Counter
from dataclasses import dataclass, field

from PIL import Image

from .llm import USAGE_FIELDS
from .workspace import RunResult, Workspace, WorkspaceError

MAX_IMAGE_EDGE = 1600  # previews are about 800 px wide; larger images are downscaled first

STYLE = """Defaults unless the task says otherwise: save a PDF (vector, for LaTeX); single-column width 3.3 in, or 6.9 in when the task asks for double-column; font size 8-9 pt; colorblind-friendly colors (for example the 'tab10' palette); axis labels with units where they apply; no figure title (the caption carries it), although short panel titles are fine; a fixed random seed; constrained or tight layout."""

SYSTEM_PROMPT = """You are a research assistant who makes publication-quality figures and small numerical toy examples (simulations, optimization traces, dynamical systems) for machine-learning papers. You work only through the tools, inside a workspace folder.

Workflow:
1. If data files are listed, look at them with read_file first.
2. Write one self-contained Python script (numpy and matplotlib only) that computes everything and saves the figure, then run it with run_python and fix any errors.
3. Every saved figure is checked automatically (axis labels, font sizes, width, legends) and rendered to a preview PNG. Read the checks, then open the preview with view_image and look for what the checks cannot see: overlapping text, a legend covering data, poor contrast, misleading axis ranges, wasted space, or a plot that does not show what the task asked for.
4. Revise and re-run until the checks pass and the figure looks right, then call finish.

""" + STYLE + """

You have at most {max_tool_calls} tool calls."""

ONESHOT_PROMPT = """You are a research assistant who writes Python scripts (numpy and matplotlib only) that make publication-quality figures and small numerical toy examples for machine-learning papers. Reply by calling submit_script once with the complete script. It will be saved as figure.py and run once, in a folder that contains the listed data files.

""" + STYLE

_PATH = {"type": "string", "description": "path relative to the workspace folder"}

TOOL_SCHEMAS = [
    {
        "name": "list_files",
        "description": "List the files in the workspace.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "read_file",
        "description": "Read a text file such as a CSV or a script. Returns at most max_lines lines (default 40).",
        "input_schema": {
            "type": "object",
            "properties": {"path": _PATH, "max_lines": {"type": "integer"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a script or text file in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"path": _PATH, "content": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_python",
        "description": (
            "Run a Python script in the workspace. Returns the exit code, the end of stdout/stderr, and for "
            "every saved figure its size, the automatic check results, and the name of a PNG preview."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"path": _PATH},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "view_image",
        "description": "Look at a PNG or JPEG in the workspace, usually a _preview_<n>.png from run_python.",
        "input_schema": {
            "type": "object",
            "properties": {"path": _PATH},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "finish",
        "description": "End the task once the figure is right. List the final figure files and give a caption and a LaTeX snippet.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "what the figure shows and what you changed along the way"},
                "figure_files": {"type": "array", "items": {"type": "string"}},
                "caption": {"type": "string"},
                "latex": {"type": "string", "description": "a figure environment with \\includegraphics, \\caption and \\label"},
            },
            "required": ["summary", "figure_files", "caption", "latex"],
            "additionalProperties": False,
        },
    },
]

SUBMIT_SCRIPT_SCHEMA = {
    "name": "submit_script",
    "description": "Submit the complete Python script.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
        "additionalProperties": False,
    },
}

REMINDER = "Continue with the tools, or call finish if the figure is done."
BUDGET_NOTE = "You have used your tool budget. Call finish now."


@dataclass
class FigureRun:
    finished: bool = False
    summary: str = ""
    files: list[str] = field(default_factory=list)
    caption: str = ""
    latex: str = ""
    stop: str = "no_finish"  # finished | no_finish | tool_budget | refusal | max_tokens
    tool_calls: list[str] = field(default_factory=list)
    last_run: RunResult | None = None
    requests: int = 0
    usage: Counter = field(default_factory=Counter)
    transcript: list[dict] = field(default_factory=list)


def make_figure(llm, ws: Workspace, task: str, max_tool_calls: int = 20) -> FigureRun:
    """Let the model work on the figure with tools until it calls finish or runs out of budget."""
    message = f"Task: {task}\n\nFiles in the workspace: {', '.join(ws.list_files()) or 'none'}"
    run = FigureRun()
    messages: list[dict] = [{"role": "user", "content": message}]
    run.transcript.append({"role": "user", "text": message})
    system = SYSTEM_PROMPT.format(max_tool_calls=max_tool_calls)
    reminded = False

    while run.requests < max_tool_calls + 3:
        response = _call(llm, run, system, TOOL_SCHEMAS, messages)
        if response.stop_reason in ("refusal", "max_tokens"):
            run.stop = response.stop_reason
            return run

        calls = [block for block in response.content if block.type == "tool_use"]
        if not calls:
            if reminded:
                return run
            reminded = True
            messages.append({"role": "user", "content": REMINDER})
            run.transcript.append({"role": "user", "text": REMINDER})
            continue

        results = []
        for call in calls:
            if call.name == "finish":
                problem = _check_finish(ws, call.input)
                if problem is None:
                    run.finished, run.stop = True, "finished"
                    run.summary, run.files = call.input["summary"], list(call.input["figure_files"])
                    run.caption, run.latex = call.input["caption"], call.input["latex"]
                    return run
                output, is_error = f"Error: {problem}", True
            elif len(run.tool_calls) >= max_tool_calls:
                output, is_error = f"Error: {BUDGET_NOTE}", True
            else:
                run.tool_calls.append(call.name)
                output, is_error = run_tool(ws, run, call.name, dict(call.input))
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": output, "is_error": is_error})
            logged = output if isinstance(output, str) else f"[image {call.input.get('path')}]"
            run.transcript.append({"role": "tool", "name": call.name, "output": logged, "is_error": is_error})
        if len(run.tool_calls) >= max_tool_calls:
            results.append({"type": "text", "text": BUDGET_NOTE})
        messages.append({"role": "user", "content": results})

    run.stop = "tool_budget"
    return run


def make_figure_oneshot(llm, ws: Workspace, task: str) -> FigureRun:
    """Ablation without feedback: the model writes one script, which is run once."""
    previews = []
    for name in ws.list_files():
        if name.endswith(".csv"):
            head = "\n".join(ws.read(name).splitlines()[:6])
            previews.append(f"--- {name} (first lines) ---\n{head}")
    message = f"Task: {task}\n\n" + ("\n\n".join(previews) if previews else "No data files.")
    run = FigureRun()
    run.transcript.append({"role": "user", "text": message})
    messages = [{"role": "user", "content": message}]
    response = _call(llm, run, ONESHOT_PROMPT, [SUBMIT_SCRIPT_SCHEMA], messages)
    calls = [block for block in response.content if block.type == "tool_use" and block.name == "submit_script"]
    if response.stop_reason in ("refusal", "max_tokens") or not calls:
        run.stop = response.stop_reason if response.stop_reason != "end_turn" else "no_script"
        return run
    ws.write("figure.py", calls[0].input["content"])
    run.last_run = ws.run("figure.py")
    run.finished, run.stop = True, "finished"
    run.files = sorted({f["file"] for f in run.last_run.figures})
    return run


def run_tool(ws: Workspace, run: FigureRun, name: str, args: dict):
    """Execute one tool call. Returns (content, is_error); content is text or a list of blocks."""
    try:
        if name == "list_files":
            return "\n".join(ws.list_files()) or "(empty)", False
        if name == "read_file":
            lines = ws.read(args["path"]).splitlines()
            limit = max(1, int(args.get("max_lines") or 40))
            more = f"\n... ({len(lines) - limit} more lines)" if len(lines) > limit else ""
            return "\n".join(lines[:limit]) + more, False
        if name == "write_file":
            ws.write(args["path"], args["content"])
            return f"Wrote {args['path']} ({len(args['content'].splitlines())} lines).", False
        if name == "run_python":
            run.last_run = ws.run(args["path"])
            return run.last_run.report(), False
        if name == "view_image":
            return _image_blocks(ws, args["path"]), False
        return f"Error: unknown tool {name!r}", True
    except (WorkspaceError, KeyError, TypeError, ValueError) as e:
        return f"Error: {e}", True


def _check_finish(ws: Workspace, args: dict) -> str | None:
    """Why a finish call cannot be accepted, or None if it can."""
    files = args.get("figure_files") or []
    if not files:
        return "list the saved figure files in figure_files"
    try:
        missing = [f for f in files if not ws.path(f).is_file()]
    except WorkspaceError as e:
        return str(e)
    return f"these figure files do not exist: {', '.join(missing)}" if missing else None


def _image_blocks(ws: Workspace, rel: str) -> list[dict]:
    path = ws.path(rel)
    if path.suffix.lower() not in (".png", ".jpg", ".jpeg") or not path.is_file():
        raise WorkspaceError(f"view_image needs an existing PNG or JPEG file, got {rel}")
    with Image.open(path) as image:
        image = image.convert("RGBA")
        flat = Image.new("RGB", image.size, "white")
        flat.paste(image, mask=image.getchannel("A"))  # transparent areas become white
    flat.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
    buffer = io.BytesIO()
    flat.save(buffer, format="PNG")
    data = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    return [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
        {"type": "text", "text": f"{rel} ({flat.width}x{flat.height} px)"},
    ]


def _call(llm, run: FigureRun, system: str, tools: list[dict], messages: list[dict]):
    response = llm.create(system=system, tools=tools, messages=messages)
    run.requests += 1
    for f in USAGE_FIELDS:
        run.usage[f] += getattr(response.usage, f, 0) or 0
    # Append the full content (thinking blocks included) so the history stays valid.
    messages.append({"role": "assistant", "content": response.content})
    for block in response.content:
        if block.type == "thinking" and getattr(block, "thinking", ""):
            run.transcript.append({"role": "assistant", "thinking": block.thinking})
        elif block.type == "text" and block.text.strip():
            run.transcript.append({"role": "assistant", "text": block.text})
        elif block.type == "tool_use":
            run.transcript.append({"role": "assistant", "tool_call": block.name, "input": dict(block.input)})
    return response
