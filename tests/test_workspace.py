import pytest

from paper_assistant.workspace import Workspace, WorkspaceError

GOOD = """
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 8
fig, ax = plt.subplots(figsize=(3.3, 2.4))
ax.plot([0, 1, 2], [0, 1, 4])
ax.set(xlabel="epoch", ylabel="loss")
fig.savefig("curve.pdf")
fig.savefig("curve.png", dpi=200)
print("done")
"""


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "ws", timeout=60)


def test_saved_figures_are_checked_and_previewed(ws):
    ws.write("plot.py", GOOD)
    result = ws.run("plot.py")
    assert result.exit_code == 0 and "done" in result.stdout
    assert [f["file"] for f in result.figures] == ["curve.pdf", "curve.png"]
    assert result.figures[0]["preview"] == result.figures[1]["preview"] == "_preview_1.png"  # same figure, one preview
    assert (ws.root / "_preview_1.png").is_file()
    assert result.saved_vector and result.issues == []
    assert "checks passed" in result.report()


def test_raster_only_output_is_flagged(ws):
    ws.write("plot.py", GOOD.replace('fig.savefig("curve.pdf")\n', ""))
    result = ws.run("plot.py")
    assert "no vector file saved; save a .pdf for LaTeX" in result.issues


def test_errors_and_timeouts_are_reported(tmp_path):
    ws = Workspace(tmp_path / "ws", timeout=3)
    ws.write("bad.py", "raise ValueError('boom')\n")
    result = ws.run("bad.py")
    assert result.exit_code != 0 and "ValueError: boom" in result.stderr
    ws.write("slow.py", "while True:\n    pass\n")
    assert ws.run("slow.py").exit_code is None


def test_stale_previews_are_removed_between_runs(ws):
    ws.write("plot.py", GOOD)
    ws.run("plot.py")
    ws.write("empty.py", "print('no figure')\n")
    result = ws.run("empty.py")
    assert result.figures == [] and not (ws.root / "_preview_1.png").exists()
    assert "No figure was saved." in result.report()


def test_data_files_are_copied_in(tmp_path):
    data = tmp_path / "results.csv"
    data.write_text("a,b\n1,2\n", encoding="utf-8")
    ws = Workspace(tmp_path / "ws", [data])
    assert ws.read("results.csv").startswith("a,b")


@pytest.mark.parametrize(
    "path, message",
    [("../escape.py", "outside"), ("_runner.py", "reserved"), ("tool.exe", "can only write")],
)
def test_writes_are_restricted(ws, path, message):
    with pytest.raises(WorkspaceError, match=message):
        ws.write(path, "x")
