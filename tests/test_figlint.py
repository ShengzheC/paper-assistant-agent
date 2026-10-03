import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from paper_assistant.figlint import lint_figure  # noqa: E402


@pytest.fixture(autouse=True)
def readable_defaults():
    with plt.rc_context({"font.size": 8}):
        yield
    plt.close("all")


def drawn(fig):
    fig.canvas.draw()  # tick labels get their text when the figure is drawn
    return lint_figure(fig)


def test_clean_figure_passes():
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot([0, 1, 2], [0, 1, 4])
    ax.set(xlabel="epoch", ylabel="loss")
    assert drawn(fig) == []


def test_missing_labels_are_reported():
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot([0, 1], [0, 1])
    assert drawn(fig) == ["missing x-axis label on axes 1", "missing y-axis label on axes 1"]


def test_multi_panel_problems_are_grouped():
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.0))
    for ax in axes:
        ax.plot([0, 1], [0, 1])
    axes[0].set_ylabel("x")
    assert drawn(fig) == ["missing x-axis label on axes 1, 2, 3", "missing y-axis label on axes 2, 3"]


def test_tiny_text_and_wide_figure_are_reported():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot([0, 1], [0, 1])
    ax.set(xlabel="x", ylabel="y")
    ax.tick_params(labelsize=5)
    issues = drawn(fig)
    assert any("10.0 in wide" in i for i in issues)
    assert any("smallest text is 5.0 pt" in i for i in issues)


def test_several_series_need_a_legend():
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot([0, 1], [0, 1], label="ours")
    ax.plot([0, 1], [1, 0], label="baseline")
    ax.set(xlabel="x", ylabel="y")
    assert drawn(fig) == ["several labeled series but no legend on axes 1"]
    ax.legend()
    assert drawn(fig) == []


def test_bar_containers_count_as_series():
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.bar([0, 1], [1, 2], width=0.4, label="MVTec-AD")
    ax.bar([0.4, 1.4], [2, 1], width=0.4, label="VisA")
    ax.set(xlabel="variant", ylabel="AUROC (%)")
    assert drawn(fig) == ["several labeled series but no legend on axes 1"]


def test_shared_axes_need_only_one_label():
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.0), sharey=True)
    for ax in axes:
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("t")
    axes[0].set_ylabel("x")
    assert drawn(fig) == []


def test_colorbars_and_hidden_axes_are_skipped():
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.0, 2.5))
    image = left.imshow(np.random.default_rng(0).random((5, 5)))
    left.set(xlabel="x", ylabel="y")
    fig.colorbar(image, ax=left)
    right.plot([0, 1], [0, 1])
    right.axis("off")
    assert drawn(fig) == []
