"""Run a script with matplotlib patched so every saved figure is checked and previewed.

Used by Workspace.run as `python -m paper_assistant._boot script.py` inside the workspace.
Writes _figures.json (one record per saved file) and one _preview_<n>.png per figure.
"""

import json
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from paper_assistant.figlint import lint_figure  # noqa: E402

PREVIEW_DPI = 110


def main() -> None:
    script = sys.argv[1]
    records: list[dict] = []
    previews: dict[int, str] = {}
    original_savefig = Figure.savefig

    def savefig(fig, fname, *args, **kwargs):
        original_savefig(fig, fname, *args, **kwargs)
        if id(fig) not in previews:
            previews[id(fig)] = f"_preview_{len(previews) + 1}.png"
            keep = {k: v for k, v in kwargs.items() if k in ("bbox_inches", "pad_inches", "facecolor", "transparent")}
            original_savefig(fig, previews[id(fig)], dpi=PREVIEW_DPI, **keep)
        width, height = fig.get_size_inches()
        records.append({
            "file": str(fname),
            "preview": previews[id(fig)],
            "size_in": [round(float(width), 2), round(float(height), 2)],
            "issues": lint_figure(fig),
        })

    Figure.savefig = savefig
    sys.argv = [script]
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        Path("_figures.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
