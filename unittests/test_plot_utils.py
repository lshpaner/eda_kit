import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import pytest
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt

from eda_toolkit._plot_utils import (
    _save_figure,
    _resolve_save_name,
    _apply_thousands,
    _add_best_fit,
    _get_label,
    _get_palette,
    _plot_density_overlays,
    _resolve_density_colors,
    _fit_distribution,
    _qq_plot,
    _cdf_exceedance_plot,
    _adjust_pvalues,
)


@pytest.fixture
def simple_data():
    return pd.DataFrame(
        {
            "x": np.random.normal(size=50),
            "group": np.random.choice([0, 1], size=50),
        }
    )


@pytest.fixture
def fig_ax():
    fig, ax = plt.subplots()
    return fig, ax


def test_save_figure_no_filename_noop(fig_ax):
    fig, _ = fig_ax
    _save_figure(fig=fig)


def test_save_figure_bare_stem_no_dir_raises(fig_ax):
    """Bare stem (no extension) with no output directory must raise, not silently no-op."""
    fig, _ = fig_ax
    with pytest.raises(ValueError, match="no.*file extension"):
        _save_figure(fig=fig, filename="test")


def test_save_figure_png_and_svg(fig_ax, tmp_path):
    fig, _ = fig_ax

    with patch.object(fig, "savefig") as mock_save:
        _save_figure(
            fig=fig,
            image_path_png=str(tmp_path),
            image_path_svg=str(tmp_path),
            filename="plot",
        )

        assert mock_save.call_count == 2


# ---------------------------------------------------------------------------
# _save_figure: full-path saving, alias, auto-mkdir (refactor)
# ---------------------------------------------------------------------------


def test_save_figure_full_path_no_directory_arg(fig_ax, tmp_path):
    """image_filename with an extension saves verbatim, no image_path_* needed."""
    fig, _ = fig_ax
    target = tmp_path / "verbatim.png"
    _save_figure(fig=fig, image_filename=str(target))
    assert target.exists()


def test_save_figure_legacy_filename_alias(fig_ax, tmp_path):
    """The old filename= keyword still works via the alias."""
    fig, _ = fig_ax
    _save_figure(fig=fig, image_path_png=str(tmp_path), filename="legacy")
    assert (tmp_path / "legacy.png").exists()


def test_save_figure_legacy_dir_plus_stem(fig_ax, tmp_path):
    """Extensionless image_filename + directory writes {dir}/{stem}.png|.svg."""
    fig, _ = fig_ax
    _save_figure(
        fig=fig,
        image_path_png=str(tmp_path),
        image_path_svg=str(tmp_path),
        image_filename="stem",
    )
    assert (tmp_path / "stem.png").exists()
    assert (tmp_path / "stem.svg").exists()


def test_save_figure_creates_missing_directories(fig_ax, tmp_path):
    """Nested output directories are created automatically."""
    fig, _ = fig_ax
    target = tmp_path / "a" / "b" / "c" / "nested.png"
    _save_figure(fig=fig, image_filename=str(target))
    assert target.exists()


# ---------------------------------------------------------------------------
# _resolve_save_name: single vs multi naming (refactor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "user_filename, multi, expected",
    [
        ("", False, "auto"),
        (None, False, "auto"),
        ("myname", False, os.path.join("myname", "auto")),
        ("out/dir/", False, os.path.join("out/dir/", "auto")),
        ("dir/file.png", False, "dir/file.png"),
        ("dir/file.png", True, os.path.join("dir", "auto.png")),
        ("file.png", True, "auto.png"),
        ("file.svg", False, "file.svg"),
    ],
)
def test_resolve_save_name(user_filename, multi, expected):
    assert _resolve_save_name(user_filename, "auto", multi=multi) == expected


# ---------------------------------------------------------------------------
# _apply_thousands: comma grouping on numeric axes (refactor)
# ---------------------------------------------------------------------------


def test_apply_thousands_commas_large_numbers(fig_ax):
    fig, ax = fig_ax
    ax.plot([0, 250000, 1400000], [1, 2, 3])
    _apply_thousands(ax, axis="x")
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert any("," in lbl for lbl in labels)


def test_apply_thousands_preserves_decimals(fig_ax):
    fig, ax = fig_ax
    ax.plot([0.0, 0.25, 0.5, 0.75], [1, 2, 3, 4])
    _apply_thousands(ax, axis="x")
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert any("." in lbl for lbl in labels)
    assert not any("," in lbl for lbl in labels)


def test_apply_thousands_axis_x_only_leaves_y_untouched(fig_ax):
    fig, ax = fig_ax
    ax.plot([0, 1000000], [0, 1000000])
    _apply_thousands(ax, axis="x")
    fig.canvas.draw()
    x_labels = [t.get_text() for t in ax.get_xticklabels()]
    y_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert any("," in lbl for lbl in x_labels)
    assert not any("," in lbl for lbl in y_labels)


def test_add_best_fit_with_legend(fig_ax):
    _, ax = fig_ax
    x = np.arange(10)
    y = 2 * x + 1

    _add_best_fit(
        ax=ax,
        x=x,
        y=y,
        linestyle="--",
        linecolor="blue",
        show_legend=True,
    )

    assert ax.legend_ is not None


def test_add_best_fit_without_legend(fig_ax):
    _, ax = fig_ax
    x = np.arange(10)
    y = 2 * x + 1

    _add_best_fit(
        ax=ax,
        x=x,
        y=y,
        linestyle="-",
        linecolor="red",
        show_legend=False,
    )

    assert ax.legend_ is None


def test_get_label_default():
    assert _get_label("x") == "x"


def test_get_label_with_mapping():
    assert _get_label("x", {"x": "X Label"}) == "X Label"


def test_get_palette_length():
    palette = _get_palette(3)
    assert len(palette) == 3


def test_resolve_density_colors_none():
    out = _resolve_density_colors(["kde", "norm"], None)
    assert out == {"kde": None, "norm": None}


def test_resolve_density_colors_str():
    out = _resolve_density_colors(["kde"], "red")
    assert out == {"kde": "red"}


def test_resolve_density_colors_list():
    out = _resolve_density_colors(["kde", "norm"], ["red", "blue"])
    assert out["norm"] == "blue"


def test_resolve_density_colors_list_length_mismatch():
    with pytest.raises(ValueError):
        _resolve_density_colors(["kde", "norm"], ["red"])


def test_resolve_density_colors_dict():
    out = _resolve_density_colors(["kde"], {"kde": "green"})
    assert out["kde"] == "green"


def test_resolve_density_colors_invalid_type():
    with pytest.raises(TypeError):
        _resolve_density_colors(["kde"], 123)


def test_fit_distribution_mle():
    data = np.random.normal(size=100)
    dist, params = _fit_distribution(data, "norm", fit_method="MLE")
    assert callable(dist.pdf)


def test_fit_distribution_mm():
    data = np.random.normal(size=100)
    dist, params = _fit_distribution(data, "norm", fit_method="MM")
    assert params is not None


def test_fit_distribution_invalid_method():
    with pytest.raises(ValueError):
        _fit_distribution(np.array([1, 2, 3]), "norm", fit_method="BAD")


def test_plot_density_overlays_kde(simple_data, fig_ax):
    _, ax = fig_ax

    _plot_density_overlays(
        ax=ax,
        data=simple_data,
        col="x",
        density_function=["kde"],
        density_fit="MLE",
        hue=None,
        log_scale=False,
        density_color=None,
    )


def test_plot_density_overlays_parametric(simple_data, fig_ax):
    _, ax = fig_ax

    _plot_density_overlays(
        ax=ax,
        data=simple_data,
        col="x",
        density_function=["norm"],
        density_fit="MLE",
        hue=None,
        log_scale=False,
        density_color="blue",
    )


def test_plot_density_overlays_invalid_dist(simple_data, fig_ax):
    _, ax = fig_ax

    with pytest.raises(ValueError):
        _plot_density_overlays(
            ax=ax,
            data=simple_data,
            col="x",
            density_function=["not_a_dist"],
            density_fit="MLE",
            hue=None,
            log_scale=False,
            density_color=None,
        )


def test_qq_plot_theoretical(fig_ax):
    _, ax = fig_ax
    data = np.random.normal(size=50)

    _qq_plot(
        ax=ax,
        data=data,
        dist_obj=stats.norm,
        params=stats.norm.fit(data),
        label="norm",
        scale="linear",
        label_fontsize=10,
        tick_fontsize=10,
    )


def test_qq_plot_empirical(fig_ax):
    _, ax = fig_ax
    data = np.random.normal(size=50)
    ref = np.random.normal(size=50)

    _qq_plot(
        ax=ax,
        data=data,
        dist_obj=None,
        params=(),
        label="emp",
        scale="linear",
        label_fontsize=10,
        tick_fontsize=10,
        qq_type="empirical",
        reference_data=ref,
    )


def test_qq_plot_too_few_points(fig_ax):
    _, ax = fig_ax
    with pytest.raises(ValueError):
        _qq_plot(
            ax=ax,
            data=np.array([1.0]),
            dist_obj=None,
            params=(),
            label="bad",
            scale="linear",
            label_fontsize=10,
            tick_fontsize=10,
        )


def test_cdf_exceedance_lower(fig_ax):
    _, ax = fig_ax
    data = np.random.normal(size=100)

    _cdf_exceedance_plot(
        ax=ax,
        data=data,
        dist_obj=stats.norm,
        params=stats.norm.fit(data),
        label="norm",
        scale="linear",
        tail="lower",
        label_fontsize=10,
    )


def test_cdf_exceedance_upper(fig_ax):
    _, ax = fig_ax
    data = np.random.normal(size=100)

    _cdf_exceedance_plot(
        ax=ax,
        data=data,
        dist_obj=stats.norm,
        params=stats.norm.fit(data),
        label="norm",
        scale="linear",
        tail="upper",
        label_fontsize=10,
    )


def test_cdf_exceedance_both(fig_ax):
    _, ax = fig_ax
    data = np.random.normal(size=100)

    _cdf_exceedance_plot(
        ax=ax,
        data=data,
        dist_obj=stats.norm,
        params=stats.norm.fit(data),
        label="norm",
        scale="linear",
        tail="both",
        label_fontsize=10,
    )


# ---------------------------------------------------------------------------
# _adjust_pvalues: multiple-comparison correction
# ---------------------------------------------------------------------------


def _symmetric_pvals(k, seed=0):
    """A symmetric matrix of pairwise p-values with 1.0 on the diagonal.

    Drawn log-uniformly over [1e-6, 1] rather than uniformly, so the matrix contains p-values small enough to survive a Bonferroni multiplier without clipping at 1.0. A flat uniform draw is almost all large values and cannot exercise the multiplier.
    """
    rng = np.random.default_rng(seed)
    P = 10.0 ** rng.uniform(-6, 0, size=(k, k))
    P = (P + P.T) / 2
    np.fill_diagonal(P, 1.0)
    labels = [f"v{i}" for i in range(k)]
    return pd.DataFrame(P, index=labels, columns=labels)


def _bh_reference(p):
    """Benjamini-Hochberg, written independently of the implementation under test."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    out = np.empty(m)
    running = 1.0
    # Walk from the largest p downward, enforcing monotonicity as we go.
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running = min(running, p[idx] * m / rank)
        out[idx] = min(running, 1.0)
    return out


def test_adjust_pvalues_bonferroni_counts_unique_pairs():
    """m is k*(k-1)/2, not k*(k-1). Correcting every off-diagonal cell would count each pair twice and over-penalize by a factor of two."""
    k = 6
    pm = _symmetric_pvals(k)
    adj = _adjust_pvalues(pm, "bonferroni")

    iu = np.triu_indices(k, 1)
    raw = pm.to_numpy()[iu]
    m = k * (k - 1) // 2

    small = raw < (0.9 / m)  # cells that will not clip at 1.0
    assert small.any()
    assert np.allclose(adj.to_numpy()[iu][small], raw[small] * m)


def test_adjust_pvalues_fdr_bh_matches_reference():
    k = 8
    pm = _symmetric_pvals(k, seed=5)
    adj = _adjust_pvalues(pm, "fdr_bh")
    iu = np.triu_indices(k, 1)
    assert np.allclose(adj.to_numpy()[iu], _bh_reference(pm.to_numpy()[iu]))


def test_adjust_pvalues_fdr_bh_is_monotone():
    """The step-up procedure must preserve the ordering of the raw p-values."""
    pm = _symmetric_pvals(7, seed=2)
    adj = _adjust_pvalues(pm, "fdr_bh")
    iu = np.triu_indices(7, 1)
    raw, out = pm.to_numpy()[iu], adj.to_numpy()[iu]
    order = np.argsort(raw)
    assert np.all(np.diff(out[order]) >= -1e-12)


@pytest.mark.parametrize("method", ["bonferroni", "fdr_bh"])
def test_adjust_pvalues_never_shrinks_and_clips_at_one(method):
    pm = _symmetric_pvals(6, seed=9)
    adj = _adjust_pvalues(pm, method).to_numpy()
    assert (adj >= pm.to_numpy() - 1e-12).all()
    assert (adj <= 1.0).all()


@pytest.mark.parametrize("method", ["bonferroni", "fdr_bh"])
def test_adjust_pvalues_symmetric_with_untouched_diagonal(method):
    pm = _symmetric_pvals(5, seed=4)
    adj = _adjust_pvalues(pm, method).to_numpy()
    assert np.allclose(adj, adj.T, equal_nan=True)
    assert np.allclose(np.diag(adj), 1.0)


def test_adjust_pvalues_nan_pairs_excluded_from_test_count():
    """A pair with too few complete observations never ran a test, so it must not inflate m, and it must stay NaN."""
    k = 4
    pm = _symmetric_pvals(k, seed=1)
    pm.iloc[0, 2] = np.nan
    pm.iloc[2, 0] = np.nan

    adj = _adjust_pvalues(pm, "bonferroni")
    assert np.isnan(adj.iloc[0, 2]) and np.isnan(adj.iloc[2, 0])

    # 6 unique pairs, one untestable, so the multiplier must be 5 and not 6.
    testable = [
        (i, j)
        for i, j in zip(*np.triu_indices(k, 1))
        if not np.isnan(pm.to_numpy()[i, j])
    ]
    m = len(testable)
    assert m == 5
    i, j = min(testable, key=lambda ij: pm.to_numpy()[ij])
    assert np.isclose(adj.to_numpy()[i, j], pm.to_numpy()[i, j] * m)


def test_adjust_pvalues_all_nan_does_not_blow_up():
    """m == 0 must return cleanly rather than dividing by zero."""
    pm = pd.DataFrame(
        [[1.0, np.nan], [np.nan, 1.0]], index=["a", "b"], columns=["a", "b"]
    )
    adj = _adjust_pvalues(pm, "fdr_bh")
    assert np.isnan(adj.loc["a", "b"])
    assert np.allclose(np.diag(adj.to_numpy()), 1.0)


def test_adjust_pvalues_preserves_labels():
    pm = _symmetric_pvals(4)
    adj = _adjust_pvalues(pm, "fdr_bh")
    assert list(adj.index) == list(pm.index)
    assert list(adj.columns) == list(pm.columns)


def test_adjust_pvalues_does_not_mutate_input():
    pm = _symmetric_pvals(5, seed=8)
    before = pm.copy()
    _adjust_pvalues(pm, "bonferroni")
    pd.testing.assert_frame_equal(pm, before)
