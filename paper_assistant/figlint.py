"""Automatic checks for paper figures, run on every figure a script saves.

The checks are simple on purpose. They catch the usual reasons a figure looks wrong once
it is in a paper: missing axis labels, text that is too small at the printed size, a figure
wider than the page, several series without a legend.
"""

from __future__ import annotations

from matplotlib.figure import Figure

MIN_FONT_PT = 7.0
MAX_WIDTH_IN = 7.2  # full text width of common two-column templates is about 6.9-7.0 in


def lint_figure(fig: Figure) -> list[str]:
    """Return human-readable problems with `fig` (an empty list means it passed)."""
    issues = []
    width, _ = fig.get_size_inches()
    if width > MAX_WIDTH_IN:
        issues.append(f"figure is {width:.1f} in wide; a column is about 3.3 in and the full text width about 6.9 in")
    problems: dict[str, list[int]] = {}  # problem -> axes numbers, so multi-panel figures stay readable
    for number, ax in enumerate(_data_axes(fig), 1):
        if not _has_label(ax, "x"):
            problems.setdefault("missing x-axis label", []).append(number)
        if not _has_label(ax, "y"):
            problems.setdefault("missing y-axis label", []).append(number)
        if len(_labeled_series(ax)) > 1 and ax.get_legend() is None and not fig.legends:
            problems.setdefault("several labeled series but no legend", []).append(number)
    for problem, numbers in problems.items():
        issues.append(f"{problem} on axes {', '.join(map(str, numbers))}")
    smallest = _smallest_font(fig)
    if smallest is not None and smallest < MIN_FONT_PT:
        issues.append(f"smallest text is {smallest:.1f} pt; use at least {MIN_FONT_PT:g} pt at the saved size")
    return issues


def _data_axes(fig: Figure) -> list:
    """Visible axes that show data, excluding colorbars."""
    return [
        ax for ax in fig.axes
        if ax.axison and not hasattr(ax, "_colorbar")
        and (ax.lines or ax.collections or ax.images or ax.patches or ax.containers)
    ]


def _has_label(ax, which: str) -> bool:
    """A label on this axes, on an axes it shares that axis with, or a figure-level sup-label."""
    get = (lambda a: a.get_xlabel()) if which == "x" else (lambda a: a.get_ylabel())
    shared = ax.get_shared_x_axes() if which == "x" else ax.get_shared_y_axes()
    if any(get(other) for other in shared.get_siblings(ax)):
        return True
    sup = getattr(ax.figure, "_supxlabel" if which == "x" else "_supylabel", None)
    return sup is not None and bool(sup.get_text())


def _labeled_series(ax) -> set[str]:
    artists = list(ax.lines) + list(ax.collections) + list(ax.containers)
    return {a.get_label() for a in artists if a.get_label() and not a.get_label().startswith("_")}


def _smallest_font(fig: Figure) -> float | None:
    texts = list(fig.texts)
    for legend in fig.legends:
        texts += legend.get_texts()
    for ax in fig.axes:
        texts += [ax.xaxis.label, ax.yaxis.label, ax.title, *ax.texts]
        texts += ax.get_xticklabels() + ax.get_yticklabels()
        if ax.get_legend() is not None:
            texts += ax.get_legend().get_texts()
    sizes = [t.get_fontsize() for t in texts if t.get_visible() and t.get_text().strip()]
    return min(sizes) if sizes else None
