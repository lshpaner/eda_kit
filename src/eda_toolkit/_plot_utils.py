import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import scipy.stats as stats
import warnings
from typing import Optional, List, Dict, Union, Tuple

################################################################################
# Figure Saving Utility
################################################################################


def _save_figure(
    *,
    fig=None,
    image_path_png=None,
    image_path_svg=None,
    image_filename=None,
    filename=None,  # legacy alias, keeps old call sites working
    bbox_inches="tight",
    dpi=None,
):
    """
    Save a matplotlib figure to PNG and/or SVG.

    If ``image_filename`` has an extension it is saved verbatim (no directory
    needed); otherwise it is used as a stem with ``image_path_png`` /
    ``image_path_svg``. ``filename`` is a legacy alias for ``image_filename``.
    """
    # accept the old `filename` keyword as an alias for `image_filename`
    if image_filename is None:
        image_filename = filename
    if image_filename is None:
        return
    fig = fig or plt.gcf()

    stem, ext = os.path.splitext(os.path.basename(image_filename))
    targets = []

    # full path with extension -> save verbatim, no dirs required
    if ext:
        targets.append(image_filename)

    # legacy: directory + base name, one file per requested format
    if image_path_png:
        targets.append(os.path.join(image_path_png, f"{stem}.png"))
    if image_path_svg:
        targets.append(os.path.join(image_path_svg, f"{stem}.svg"))

    # A filename was provided but nothing was saveable: bare stem, no extension,
    # no output directory. Fail loudly instead of silently no-op'ing.
    if not targets:
        raise ValueError(
            f"Cannot save figure: `image_filename='{image_filename}'` has no "
            "file extension and no `image_path_png` / `image_path_svg` was given. "
            "Add an extension (e.g. '.png', '.jpg', '.svg') or pass an output "
            "directory."
        )

    for path in targets:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(path, bbox_inches=bbox_inches, dpi=dpi)  # dpi ignored for vector


def _resolve_save_name(user_filename, auto_stem, multi=False):
    """Build the per-figure save name.
    Single-file callers (multi=False) get the user's exact filename.
    Multi-file callers (multi=True) keep the user's dir + extension but
    use the auto name so the many files stay distinct."""
    if not user_filename:
        return auto_stem  # dirs-only (legacy)
    directory, base = os.path.split(user_filename)
    _, ext = os.path.splitext(base)
    if not ext:  # bare path -> directory
        return os.path.join(user_filename, auto_stem)
    if multi:
        return os.path.join(directory, f"{auto_stem}{ext}")
    return user_filename  # single file -> honor it


################################################################################
# Utilities For Stacked Crosstab Plots
################################################################################


# Helper: apply legend with optional reversal
def _apply_legend(ax, labels, loc, fontsize, reverse, bbox_to_anchor=None, ncols=1):
    handles, leg_labels = ax.get_legend_handles_labels()
    if reverse:
        handles, leg_labels = handles[::-1], leg_labels[::-1]
    ax.legend(
        handles,
        leg_labels,
        loc=loc,
        fontsize=fontsize,
        bbox_to_anchor=bbox_to_anchor,
        ncols=ncols,
    )


# Helper: annotate stacked bar segments with values
def _annotate_stacked(ax, data, fmt_func, kind, tick_fontsize):
    for bar_idx, (_, row) in enumerate(data.iterrows()):
        cumulative = 0.0
        for val in row:
            if val > 0:
                label = fmt_func(val)
                if kind == "barh":
                    ax.text(
                        cumulative + val / 2,
                        bar_idx,
                        label,
                        ha="center",
                        va="center",
                        fontsize=tick_fontsize,
                        color="white",
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        bar_idx,
                        cumulative + val / 2,
                        label,
                        ha="center",
                        va="center",
                        fontsize=tick_fontsize,
                        color="white",
                        fontweight="bold",
                    )
            cumulative += val


################################################################################
# Best-Fit Line Utilities
################################################################################


def _add_best_fit(
    *,
    ax,
    x,
    y,
    linestyle,
    linecolor,
    show_legend: bool,
    legend_loc: str = "best",
) -> None:
    """
    Add a linear least-squares best-fit line to an existing Axes.

    This utility computes a first-order (linear) polynomial fit using
    ``numpy.polyfit`` and overlays the resulting line on the provided
    Matplotlib Axes. The fitted equation is added as the line label and
    the legend is optionally shown or removed.
    """

    m, b = np.polyfit(x, y, 1)

    ax.plot(
        x,
        m * x + b,
        color=linecolor,
        linestyle=linestyle,
        label=f"y = {m:.2f}x + {b:.2f}",
    )

    if show_legend:
        ax.legend(loc=legend_loc)
    else:
        if ax.legend_ is not None:
            ax.legend_.remove()


################################################################################
# Labeling and Palette Utilities
################################################################################


def _get_label(var: str, label_names: Optional[Dict[str, str]] = None) -> str:
    """
    Return a display label for a variable.

    If label_names is provided and contains the variable, return the mapped
    label. Otherwise return the variable name.
    """
    if not label_names:
        return var

    ## used, for example, in scatter_fit_plot
    return label_names.get(var, var)


def _get_palette(n_colors):
    """
    Returns a 'tab10' color palette with the specified number of colors.
    """
    return sns.color_palette("tab10", n_colors=n_colors)


def _apply_thousands(ax, axis="both"):
    """Comma-group numeric tick labels while preserving decimals.

    axis : "both", "x", or "y" — caller picks the numeric axis/axes.
    """
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:,.10g}")
    for a in (
        (ax.xaxis, ax.yaxis) if axis == "both" else (getattr(ax, f"{axis}axis"),)
    ):
        a.set_major_formatter(fmt)


################################################################################
# Density Overlay Plotting Utils
################################################################################


def _plot_density_overlays(
    *,
    ax,
    data: pd.DataFrame,
    col: str,
    density_function: List[str],
    density_fit: str,
    hue: Optional[str],
    log_scale: bool,
    density_color: Optional[Union[str, List[str], Dict[str, str]]],
    **kwargs,
) -> None:
    """
    Plot density overlays (KDE and/or parametric PDFs).

    Supports:
    - single color for all densities
    - list of colors aligned with density_function
    - dict mapping {density_name: color}
    """

    x = data[col].dropna().values
    if len(x) <= 1:
        return

    x_grid = np.linspace(x.min(), x.max(), 500)

    # ------------------------------------------------------------------
    # Normalize density colors
    # ------------------------------------------------------------------
    if density_color is None:
        color_map = {d: None for d in density_function}

    elif isinstance(density_color, str):
        color_map = {d: density_color for d in density_function}

    elif isinstance(density_color, list):
        if len(density_color) != len(density_function):
            raise ValueError(
                "When density_color is a list, its length must match "
                "density_function."
            )
        color_map = dict(zip(density_function, density_color))

    elif isinstance(density_color, dict):
        color_map = {d: density_color.get(d) for d in density_function}

    else:
        raise TypeError(
            "density_color must be a str, list[str], dict[str, str], or None."
        )

    # ------------------------------------------------------------------
    # Plot density overlays
    # ------------------------------------------------------------------
    for dist_name in density_function:
        curve_color = color_map.get(dist_name)

        # KDE
        if dist_name == "kde":
            sns.kdeplot(
                data=data,
                x=col,
                ax=ax,
                hue=hue,
                color=curve_color if hue is None else None,
                log_scale=log_scale,
                label="kde",
                **kwargs,
            )
            continue

        # Parametric distributions
        if not hasattr(stats, dist_name):
            raise ValueError(
                f"Unknown density_function '{dist_name}'. "
                "Use 'kde' or a valid scipy.stats distribution name "
                "(e.g., 'norm', 'lognorm', 'gamma')."
            )

        dist = getattr(stats, dist_name)

        try:
            params = dist.fit(x) if density_fit == "MLE" else dist.fit(x, method="MM")
            pdf = dist.pdf(x_grid, *params)

            ax.plot(
                x_grid,
                pdf,
                label=dist_name,
                color=curve_color,
            )

        except Exception as e:
            warnings.warn(
                f"Could not fit '{dist_name}' for '{col}': {e}",
                UserWarning,
            )


################################################################################
# Resolve Density Colors
################################################################################


def _resolve_density_colors(
    density_function: List[str],
    density_color: Optional[Union[str, List[str], Dict[str, str]]],
) -> Dict[str, Optional[str]]:

    if density_color is None:
        return {d: None for d in density_function}

    if isinstance(density_color, str):
        return {d: density_color for d in density_function}

    if isinstance(density_color, list):
        if len(density_color) != len(density_function):
            raise ValueError(
                "When density_color is a list, its length must match "
                "density_function."
            )
        return dict(zip(density_function, density_color))

    if isinstance(density_color, dict):
        return {d: density_color.get(d) for d in density_function}

    raise TypeError("density_color must be a str, list[str], dict[str, str], or None.")


################################################################################
# Distribution Fitting Utilities
################################################################################


def _fit_distribution(
    data: np.ndarray,
    dist_name: str,
    fit_method: str = "MLE",
):
    """
    Fit a scipy.stats distribution using MLE or Method of Moments.
    """
    dist = getattr(stats, dist_name)

    if fit_method == "MLE":
        params = dist.fit(data)
    elif fit_method == "MM":
        params = dist.fit(data, method="MM")
    else:
        raise ValueError("fit_method must be 'MLE' or 'MM'")

    return dist, params


################################################################################
# Quantile–Quantile Plotting Utilities
################################################################################


def _qq_plot(
    ax,
    data: np.ndarray,
    dist_obj,
    params: Tuple,
    label: str,
    scale: str,
    label_fontsize: int,
    tick_fontsize: int,
    qq_type: str = "theoretical",
    reference_data: Optional[np.ndarray] = None,
    show_reference: bool = True,
    color: Optional[str] = None,
):
    """
    Quantile–Quantile plot.

    qq_type:
        - "theoretical": sample vs fitted distribution
        - "empirical": sample vs reference_data
    """

    # ---------------------------
    # Validation
    # ---------------------------
    if data is None or len(data) < 2:
        raise ValueError("QQ plot requires at least 2 data points.")

    if qq_type == "empirical":
        if reference_data is None or len(reference_data) < 2:
            raise ValueError(
                "Empirical QQ plot requires reference_data with >= 2 values."
            )

    # ---------------------------
    # Quantiles
    # ---------------------------
    if qq_type == "theoretical":
        osm, osr = stats.probplot(
            data,
            dist=dist_obj,
            sparams=params,
            plot=None,
        )[0]
        xlabel = "Theoretical Quantiles"

    else:  # empirical
        q = np.linspace(0.01, 0.99, 100)
        osm = np.quantile(reference_data, q)
        osr = np.quantile(data, q)
        xlabel = "Reference Quantiles"

    # ---------------------------
    # Scatter
    # ---------------------------
    ax.scatter(
        osm,
        osr,
        s=15,
        alpha=0.7,
        color=color,
        label=label,
    )

    # ---------------------------
    # Reference line (theoretical only)
    # ---------------------------
    if qq_type == "theoretical" and show_reference:
        ax.plot(
            [osm.min(), osm.max()],
            [osm.min(), osm.max()],
            linestyle="--",
            color=color,
            alpha=0.5,
            label=f"{label} reference",
        )

    # ---------------------------
    # Formatting
    # ---------------------------
    if scale == "log":
        ax.set_yscale("log")

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel("Sample Quantiles", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)


################################################################################
# CDF and Exceedance Plotting Utilities
################################################################################


def _cdf_exceedance_plot(
    ax,
    data: np.ndarray,
    dist_obj,
    params: Tuple,
    label: str,
    scale: str,
    tail: str,
    label_fontsize: int,
    color: Optional[str] = None,
):
    """
    Plot CDF or exceedance probability with optional log scaling.
    """
    x = np.sort(data)
    cdf = dist_obj.cdf(x, *params)

    # Determine which curves to plot and the y-axis label
    if tail == "lower":
        y = cdf
        ax.plot(x, y, label=label, color=color)
        ylabel = "CDF"

    elif tail == "upper":
        y = 1 - cdf
        ax.plot(x, y, label=label, color=color)
        ylabel = "Exceedance Probability"

    else:  # both
        ax.plot(x, cdf, label=f"{label} CDF", alpha=0.8, color=color)
        ax.plot(
            x,
            1 - cdf,
            label=f"{label} Exceedance",
            alpha=0.8,
            linestyle="--",
            color=color,
        )
        ylabel = "Probability"

    # Apply scale BEFORE setting labels
    if scale == "log":
        ax.set_yscale("log")

    # Set axis labels with explicit font size
    ax.set_xlabel("x", fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)


################################################################################
# Multiple-Comparison Correction Utility
################################################################################


def _adjust_pvalues(pval_matrix, method):
    """
    Apply a multiple-comparison correction to a symmetric matrix of pairwise
    p-values.

    The correction is computed over the set of *unique* pairs
    (the strict upper triangle), so the number of tests is
    k * (k - 1) / 2 rather than k * (k - 1). Correcting every off-diagonal cell
    independently would count each pair twice and over-penalize by a factor of two.
    Adjusted values are written back symmetrically. NaN cells (pairs with too few
    complete observations to test) are excluded from the test count and left as
    NaN. The diagonal is untouched.

    Parameters:
    -----------
    pval_matrix : pandas.DataFrame
        Square, symmetric matrix of raw pairwise p-values.

    method : str
        Correction to apply. Either "bonferroni" (multiply each p-value by the
        number of tests) or "fdr_bh" (Benjamini-Hochberg step-up procedure
        controlling the false discovery rate).

    Returns:
    --------
    pandas.DataFrame
        A new matrix of adjusted p-values with the same index, columns, and
        symmetry as the input. Values are clipped to a maximum of 1.0.
    """
    adjusted = pval_matrix.copy()
    k = len(pval_matrix)

    # Collect the unique pairs from the strict upper triangle.
    iu = np.triu_indices(k, k=1)
    raw = pval_matrix.to_numpy()[iu]

    # Only testable pairs count toward the correction; NaN pairs never ran a test.
    testable = ~np.isnan(raw)
    m = int(testable.sum())
    if m == 0:
        return adjusted

    p = raw[testable]

    if method == "bonferroni":
        p_adj = np.minimum(p * m, 1.0)
    else:  # fdr_bh
        order = np.argsort(p)
        ranked = p[order]
        # Step-up: scale by m / rank, then enforce monotonicity from the largest down.
        scaled = ranked * m / np.arange(1, m + 1)
        scaled = np.minimum.accumulate(scaled[::-1])[::-1]
        p_adj = np.empty(m, dtype=float)
        p_adj[order] = np.minimum(scaled, 1.0)

    out = np.full(raw.shape, np.nan, dtype=float)
    out[testable] = p_adj

    values = adjusted.to_numpy(dtype=float, copy=True)
    values[iu] = out
    # Mirror the upper triangle back onto the lower triangle.
    values[(iu[1], iu[0])] = out

    return pd.DataFrame(values, index=pval_matrix.index, columns=pval_matrix.columns)
