"""The figure agent and the polisher, driven by a scripted model so no API key is needed."""

import pytest

from paper_assistant.figure_agent import BUDGET_NOTE, SUBMIT_SCRIPT_SCHEMA, make_figure, make_figure_oneshot
from paper_assistant.llm import ScriptedLLM
from paper_assistant.polish import polish
from paper_assistant.workspace import Workspace

SCRIPT = """
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 8
fig, ax = plt.subplots(figsize=(3.3, 2.4))
ax.plot([0, 1, 2], [0, 1, 4])
ax.set(xlabel="epoch", ylabel="loss")
fig.savefig("curve.pdf")
"""

FINISH = {
    "summary": "Loss curve.",
    "figure_files": ["curve.pdf"],
    "caption": "Training loss.",
    "latex": "\\begin{figure}\\includegraphics{curve.pdf}\\caption{Training loss.}\\end{figure}",
}


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "ws", timeout=60)


def test_write_run_view_finish(ws):
    llm = ScriptedLLM([
        [("write_file", {"path": "plot.py", "content": SCRIPT})],
        [("run_python", {"path": "plot.py"})],
        [("view_image", {"path": "_preview_1.png"})],
        [("finish", FINISH)],
    ])
    run = make_figure(llm, ws, "Plot a loss curve.")
    assert run.finished and run.files == ["curve.pdf"]
    assert run.tool_calls == ["write_file", "run_python", "view_image"]
    assert run.last_run.exit_code == 0 and run.last_run.issues == []
    # The run report and then the rendered preview (as an image block) go back to the model.
    assert "checks passed" in llm.requests[2]["messages"][-1]["content"][0]["content"]
    image_result = llm.requests[3]["messages"][-1]["content"][0]["content"]
    assert image_result[0]["type"] == "image" and image_result[0]["source"]["media_type"] == "image/png"


def test_finish_needs_existing_files(ws):
    llm = ScriptedLLM([
        [("finish", {**FINISH, "figure_files": ["missing.pdf"]})],
        [("finish", {**FINISH, "figure_files": ["../outside.pdf"]})],
        "giving up",
        "still nothing",
    ])
    run = make_figure(llm, ws, "Plot something.")
    assert not run.finished
    assert "do not exist: missing.pdf" in llm.requests[1]["messages"][-1]["content"][0]["content"]
    assert "outside the workspace" in llm.requests[2]["messages"][-1]["content"][0]["content"]


def test_tool_budget_is_enforced(ws):
    call = [("list_files", {})]
    llm = ScriptedLLM([call, call, call, [("finish", FINISH)]])
    run = make_figure(llm, ws, "Plot something.", max_tool_calls=2)
    assert run.tool_calls == ["list_files", "list_files"]
    assert llm.requests[3]["messages"][-1]["content"][-1] == {"type": "text", "text": BUDGET_NOTE}


def test_oneshot_runs_the_submitted_script_once(ws):
    llm = ScriptedLLM([[("submit_script", {"content": SCRIPT})]])
    run = make_figure_oneshot(llm, ws, "Plot a loss curve.")
    assert llm.requests[0]["tools"] == [SUBMIT_SCRIPT_SCHEMA]
    assert run.last_run.exit_code == 0 and run.files == ["curve.pdf"]


PARAGRAPH = r"Flow matching \cite{lipman2023} learns $v_\theta(x,t)$, see Fig.~\ref{fig:paths}."


def test_polish_retries_when_a_citation_is_dropped():
    llm = ScriptedLLM([
        {"revised": r"Flow matching learns $v_\theta(x,t)$ (Fig.~\ref{fig:paths}).", "changes": ["shorter"]},
        {"revised": r"Flow matching \cite{lipman2023} learns $v_\theta(x,t)$ (Fig.~\ref{fig:paths}).", "changes": ["shorter"]},
    ])
    result = polish(llm, PARAGRAPH)
    assert result.ok and result.attempts == 2
    feedback = llm.requests[1]["messages"][-1]["content"]
    assert "missing or changed: \\cite{lipman2023}" in feedback


def test_polish_reports_failure_after_the_last_attempt():
    changed_math = {"revised": r"Flow matching \cite{lipman2023} learns $v(x,t)$, see Fig.~\ref{fig:paths}.", "changes": []}
    result = polish(ScriptedLLM([changed_math, changed_math]), PARAGRAPH)
    assert not result.ok and result.attempts == 2
    assert any("$v_\\theta(x,t)$" in p for p in result.problems)
