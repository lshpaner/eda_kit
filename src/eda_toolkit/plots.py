################################################################################
############################### Library Imports ################################
import pandas as pd
import numpy as np
import math
import scipy.stats as stats
from scipy.stats import gaussian_kde
import itertools  # Import itertools for combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import textwrap
import os
import warnings
from sklearn.preprocessing import PowerTransformer, RobustScaler
from tqdm import tqdm

from typing import Optional, Union, List, Dict, Tuple, Any

from pathlib import Path

PathLike = Union[str, Path]

from ._plot_utils import (
    _save_figure,
    _resolve_save_name,
    _apply_legend,
    _annotate_stacked,
    _add_best_fit,
    _get_label,
    _apply_thousands,
    _plot_density_overlays,
    _fit_distribution,
    _qq_plot,
    _cdf_exceedance_plot,
    _get_palette,
    _adjust_pvalues,
)

################################################################################
############################## Plot Distributions ##############################
################################################################################


def kde_distributions(*args, **kwargs):
    warnings.warn(
        "`kde_distributions` is deprecated. Use `plot_distributions` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_distributions(*args, **kwargs)


def plot_distributions(
    df: pd.DataFrame,
    vars_of_interest: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (5, 5),
    subplot_figsize: Optional[Tuple[int, int]] = None,
    hist_color: str = "#0000FF",
    density_color: Optional[str] = None,
    mean_color: str = "#000000",
    median_color: str = "#000000",
    hist_edgecolor: str = "#000000",
    hue: Optional[str] = None,
    fill: bool = True,
    fill_alpha: float = 1.0,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    w_pad: float = 1.0,
    h_pad: float = 1.0,
    image_path_png: Optional[str] = None,
    image_path_svg: Optional[str] = None,
    image_filename: Optional[str] = None,
    bbox_inches: Optional[str] = None,
    single_var_image_filename: Optional[str] = None,
    y_axis_label: str = "Density",
    plot_type: str = "hist",
    density_function: Optional[Union[str, List[str]]] = "kde",
    density_fit: str = "MLE",
    log_scale_vars: Optional[Union[str, List[str]]] = None,
    bins: Union[str, int, List[Any], np.ndarray] = "auto",
    binwidth: Optional[float] = None,
    label_fontsize: int = 10,
    tick_fontsize: int = 10,
    text_wrap: int = 50,
    disable_sci_notation: bool = False,
    stat: str = "density",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    plot_mean: bool = False,
    plot_median: bool = False,
    std_dev_levels: Optional[Union[List[int], int]] = None,
    std_color: Union[str, List[str]] = "#808080",
    label_names: Optional[Dict[str, str]] = None,
    show_legend: bool = True,
    legend_loc: str = "best",
    custom_xlabels: Optional[Dict[str, str]] = None,
    custom_titles: Optional[Dict[str, str]] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Generate histogram and/or density distribution plots (KDE or parametric)
    for columns in a DataFrame.

    This function provides a flexible way to visualize the distribution of
    data for specified columns in a DataFrame. It supports both kernel density
    estimation (KDE) and histograms, with options to customize various aspects
    of the plots, including colors, labels, binning, scaling, and statistical
    overlays.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.

    vars_of_interest : list of str, optional
        List of column names for which to generate distribution plots. If
        'all', plots will be generated for all numeric columns.

    figsize : tuple of int, optional (default=(5, 5))
        Size of each individual plot. This parameter is used when there is only
        one variable being plotted or when generating separate plots for each
        variable of interest.

    subplot_figsize : tuple of int, optional
        Size of the overall grid of subplots when there are multiple variables
        being plotted in a single subplot grid. This parameter is ignored when
        only one variable is plotted or when using `single_var_image_filename`.

    hist_color : str, optional (default='#0000FF')
        Color of the histogram bars.

    density_color : str, optional (default=None)
        Color of the KDE / density curve.

    mean_color : str, optional (default='#000000')
        Color of the mean line if `plot_mean` is True.

    median_color : str, optional (default='#000000')
        Color of the median line if `plot_median` is True.

    hist_edgecolor : str, optional (default='#000000')
        Color of the histogram bar edges.

    hue : str, optional
        Column name to group data by, adding different colors for each group.

    fill : bool, optional (default=True)
        Whether to fill the histogram bars with color.

    fill_alpha : float, optional (default=1)
        Alpha transparency for the fill color of the histogram bars, where
        0 is fully transparent and 1 is fully opaque.

    n_rows : int, optional
        Number of rows in the subplot grid. If not provided, it will be
        calculated automatically.

    n_cols : int, optional
        Number of columns in the subplot grid. If not provided, it will be
        calculated automatically.

    w_pad : float, optional (default=1.0)
        Width padding between subplots.

    h_pad : float, optional (default=1.0)
        Height padding between subplots.

    image_path_png : str, optional
        Directory path to save the PNG image of the overall distribution plots.

    image_path_svg : str, optional
        Directory path to save the SVG image of the overall distribution plots.

    image_filename : str, optional
        Filename for the overall (subplot grid) figure. If it includes a file
        extension it is saved verbatim and no output directory is required;
        otherwise it is used as a base stem with ``image_path_png`` and/or
        ``image_path_svg``.

    bbox_inches : str, optional
        Bounding box to use when saving the figure. For example, 'tight'.

    single_var_image_filename : str, optional
        Filename for the separate per-variable plots. May be a bare stem
        (legacy: requires ``image_path_png`` / ``image_path_svg``) or a full
        path with extension (saved verbatim, no directory needed). The
        variable name is appended to the stem, before the extension. When this
        is used, ``figsize`` controls the individual plot size and
        ``subplot_figsize`` is ignored.

    y_axis_label : str, optional (default='Density')
        The label to display on the y-axis. If set to `None`, no y-axis label
        will be displayed.

    plot_type : str, optional (default='hist')
        The type of plot to generate ('hist', 'density', or 'both').

    log_scale_vars : str or list of str, optional
        Variable name(s) to apply log scaling. Can be a single string or a
        list of strings.

    bins : int or sequence, optional (default='auto')
        Specification of histogram bins.

    binwidth : float, optional
        Width of each bin, overrides bins but can be used with binrange.

    label_fontsize : int, optional (default=10)
        Font size for axis labels, including xlabel, ylabel, and tick marks.

    tick_fontsize : int, optional (default=10)
        Font size for tick labels on the axes.

    text_wrap : int, optional (default=50)
        Maximum number of characters allowed in plot titles and axis labels
        before wrapping them onto the next line.

    disable_sci_notation : bool, optional (default=False)
        Toggle to disable scientific notation on axes.

    stat : str, optional (default='density')
        Aggregate statistic to compute in each bin (e.g., 'count', 'frequency',
        'probability', 'percent', 'density').

    xlim : tuple of (float, float), optional
        Limits for the x-axis.

    ylim : tuple of (float, float), optional
        Limits for the y-axis.

    plot_mean : bool, optional (default=False)
        Whether to plot the mean as a vertical line.

    plot_median : bool, optional (default=False)
        Whether to plot the median as a vertical line.

    std_dev_levels : list of int, optional
        Levels of standard deviation to plot around the mean.

    std_color : str or list of str, optional (default='#808080')
        Color(s) for the standard deviation lines.

    label_names : dict, optional
        Custom labels for the variables of interest. Keys should be column
        names, and values should be the corresponding labels to display.

    show_legend : bool, optional (default=True)
        Whether to show the legend on the plots.

    legend_loc : str, optional (default="best")
        Location of the legend on the plot. Passed directly to
        `matplotlib.axes.Axes.legend`. Common options include
        "best", "upper right", "upper left", "lower left",
        "lower right", and "center".

    custom_xlabels : dict, optional
        Dictionary to customize x-axis labels. Keys are column names, and values
        are the desired labels. If a value is `None`, no x-axis label is displayed
        for the corresponding column.

    custom_titles : dict, optional
        Dictionary to customize plot titles. Keys are column names, and values
        are the desired titles. If a value is `None`, no title is displayed
        for the corresponding column.

    **kwargs : additional keyword arguments
        Additional keyword arguments passed to the Seaborn plotting function.

    Returns:
    --------
    None
        This function does not return any value but generates and optionally
        saves distribution plots for the specified columns in the DataFrame.

    Raises:
    -------
    ValueError
        If `plot_type` is not one of ['hist', 'density', 'both'].

    ValueError
        If `stat` is not one of ['count', 'frequency', 'probability',
        'percent', 'density'].

    ValueError
        If any variable specified in `log_scale_vars` is not in the DataFrame.

    ValueError
        If `fill` is set to False but `hist_edgecolor` or `fill_alpha` is
        specified.

    ValueError
        If `bins` and `binwidth` are both set, which can affect performance.

    ValueError
        If `subplot_figsize` is provided when only one plot is being created.

    Warnings:
    ---------
    UserWarning
        If both `bins` and `binwidth` are set, a warning about performance
        impacts is raised.
    """

    get_label = lambda v: _get_label(v, label_names)

    # Handle the "all" option for vars_of_interest
    if vars_of_interest == "all":
        vars_of_interest = df.select_dtypes(include=np.number).columns.tolist()

    if vars_of_interest is None:
        print("Error: No variables of interest provided.")
        return

    # Set defaults for optional parameters
    if std_dev_levels is None:
        std_dev_levels = []  # Empty list if not provided

    # Dynamically calculate n_rows and n_cols if not provided
    num_vars = len(vars_of_interest)

    # If only one variable is being plotted
    if num_vars == 1:
        n_rows, n_cols = 1, 1
        if subplot_figsize is not None:
            raise ValueError(
                f"Cannot use subplot_figsize when there is only one "
                f"plot. Use figsize instead."
            )
    else:
        # Calculate columns based on square root
        if n_rows is None or n_cols is None:
            n_cols = int(np.ceil(np.sqrt(num_vars)))
            n_rows = int(np.ceil(num_vars / n_cols))

        # Adjust figsize for subplots if multiple plots
        if subplot_figsize is None:
            figsize = (figsize[0] * n_cols, figsize[1] * n_rows)
        else:
            figsize = subplot_figsize

    # Convert log_scale_vars to list if it's a single string
    if isinstance(log_scale_vars, str):
        log_scale_vars = [log_scale_vars]

    # Ensure std_dev_levels is a list if it's specified
    if isinstance(std_dev_levels, int):
        std_dev_levels = [std_dev_levels]

    # Ensure std_color is a list with enough colors
    if isinstance(std_color, str):
        std_color = [std_color] * len(std_dev_levels)
    elif isinstance(std_color, list) and len(std_color) < len(std_dev_levels):
        raise ValueError(
            f"Not enough colors specified in 'std_color'. "
            f"You have {len(std_color)} color(s) but {len(std_dev_levels)} "
            f"standard deviation level(s). "
            f"Please provide at least as many colors as standard deviation levels."
        )

    # Validate plot_type parameter
    valid_plot_types = ["hist", "density", "both"]

    if plot_type.lower() not in valid_plot_types:
        raise ValueError(
            f"Invalid plot_type value. Expected one of {valid_plot_types}, "
            f"got '{plot_type}' instead."
        )

    # Validate stat parameter
    valid_stats = [
        "count",
        "frequency",
        "probability",
        "proportion",
        "percent",
        "density",
    ]
    if stat.lower() not in valid_stats:
        raise ValueError(
            f"Invalid stat value. Expected one of {valid_stats}, "
            f"got '{stat}' instead."
        )

    # Check if all log_scale_vars are in the DataFrame
    if log_scale_vars:
        invalid_vars = [var for var in log_scale_vars if var not in df.columns]
        if invalid_vars:
            raise ValueError(f"Invalid log_scale_vars: {invalid_vars}")

    # Check if edgecolor is being set while fill is False
    if not fill and hist_edgecolor != "#000000":
        raise ValueError("Cannot change edgecolor when fill is set to False")

    # Check if fill_alpha is being set while fill is False
    if not fill and fill_alpha != 0.6:
        raise ValueError("Cannot set fill_alpha when fill is set to False")

    # Warn if both bins and binwidth are set
    if bins != "auto" and binwidth is not None:
        warnings.warn(
            "Specifying both bins and binwidth may affect performance.",
            UserWarning,
        )

    # Normalize density_function
    if density_function is None:
        density_function = []
    elif isinstance(density_function, str):
        density_function = [density_function]

    # Validate density_function entries
    invalid_density = [
        d for d in density_function if d != "kde" and not hasattr(stats, d)
    ]

    available_density_functions = sorted(
        name
        for name in dir(stats)
        if hasattr(getattr(stats, name), "pdf") and hasattr(getattr(stats, name), "fit")
    )

    if invalid_density:
        wrapped = textwrap.fill(", ".join(available_density_functions), width=100)

        raise ValueError(
            "Invalid density_function value(s): "
            f"{invalid_density}.\n"
            "Valid options are 'kde' or any distribution available in scipy.stats:\n"
            f"{wrapped}"
        )

    density_fit = density_fit.upper()
    if density_fit not in ["MLE", "MM"]:
        raise ValueError("density_fit must be 'MLE' or 'MM'")

    # Create subplots grid
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    # Flatten the axes array to simplify iteration
    axes = np.atleast_1d(axes).flatten()

    # Iterate over the provided column list and corresponding axes
    for ax, col in zip(axes[: len(vars_of_interest)], vars_of_interest):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            # Determine if log scale should be applied to this variable
            log_scale = col in log_scale_vars if log_scale_vars else False

            # Filter out non-positive values if log_scale is True
            data = df[df[col] > 0] if log_scale else df

            if custom_xlabels and col in custom_xlabels:
                xlabel = custom_xlabels[col]
            else:
                # Add "(Log)" to the label if log_scale is applied
                xlabel = f"{get_label(col)} (Log)" if log_scale else get_label(col)

            # Determine custom title
            if custom_titles and col in custom_titles:
                title = custom_titles[col]
            else:
                # Modify the title to include "(Log Scaled)" if log_scale is applied
                title = f"Distribution of {get_label(col)} {'(Log Scaled)' if log_scale else ''}"

            # Calculate mean and median if needed
            mean_value = data[col].mean() if plot_mean or std_dev_levels else None
            median_value = data[col].median() if plot_median else None
            std_value = data[col].std() if std_dev_levels else None

            try:
                if plot_type == "hist":
                    sns.histplot(
                        data=data,
                        x=col,
                        kde=False,
                        ax=ax,
                        hue=hue,
                        color=hist_color if hue is None and fill else None,
                        edgecolor=hist_edgecolor,
                        stat=stat.lower(),
                        fill=fill,
                        alpha=fill_alpha,  # Apply for transparency
                        log_scale=log_scale,
                        bins=bins,
                        binwidth=binwidth,
                        **kwargs,
                    )
                elif plot_type == "density":
                    _plot_density_overlays(
                        ax=ax,
                        data=data,
                        col=col,
                        density_function=density_function,
                        density_fit=density_fit,
                        hue=hue,
                        log_scale=log_scale,
                        density_color=density_color,
                        **kwargs,
                    )
                elif plot_type == "both":
                    sns.histplot(
                        data=data,
                        x=col,
                        kde=False,  # No need, since plot_type controls it
                        ax=ax,
                        hue=hue,
                        color=hist_color if hue is None and fill else None,
                        edgecolor=hist_edgecolor,
                        stat=stat.lower(),
                        fill=fill,
                        alpha=fill_alpha,  # Apply for transparency
                        log_scale=log_scale,
                        bins=bins,
                        binwidth=binwidth,
                        **kwargs,
                    )
                    _plot_density_overlays(
                        ax=ax,
                        data=data,
                        col=col,
                        density_function=density_function,
                        density_fit=density_fit,
                        hue=hue,
                        log_scale=log_scale,
                        density_color=density_color,
                        **kwargs,
                    )
                # Plot mean as a vertical dotted line if plot_mean is True
                if plot_mean and mean_value is not None:
                    ax.axvline(
                        mean_value,
                        color=mean_color,
                        linestyle="--",
                        label="Mean",
                    )

                # Plot median as a vertical dotted line if plot_median is True
                if plot_median and median_value is not None:
                    ax.axvline(
                        median_value,
                        color=median_color,
                        linestyle="--",
                        label="Median",
                    )

                # Plot standard deviation bands if std_dev_levels is specified
                if std_dev_levels and mean_value is not None and std_value is not None:
                    for level, color in zip(std_dev_levels, std_color):
                        ax.axvline(
                            mean_value + level * std_value,
                            color=color,
                            linestyle="--",
                            label=f"±{level} Std Dev",
                        )
                        ax.axvline(
                            mean_value - level * std_value,
                            color=color,
                            linestyle="--",
                        )

                # After plotting logic, before showing the legend
                handles, labels = ax.get_legend_handles_labels()

                if show_legend and len(handles) > 0:
                    ax.legend(loc=legend_loc)
                else:
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()

            except Exception as e:
                # Handle different Python versions or issues w/ legends & labels
                if "No artists with labels found to put in legend." in str(e):
                    print(f"Warning encountered while plotting '{col}': {str(e)}")
                    handles, labels = ax.get_legend_handles_labels()
                    if show_legend and len(handles) > 0 and len(labels) > 0:
                        ax.legend(loc=legend_loc)
                    else:
                        if ax.get_legend() is not None:
                            ax.get_legend().remove()

            finally:
                pass  # Ensures the try block is valid

            ax.set_xlabel(
                "\n".join(textwrap.wrap(xlabel, width=text_wrap)) if xlabel else None,
                fontsize=label_fontsize,
            )

            ax.set_ylabel(
                (
                    "\n".join(textwrap.wrap(y_axis_label.capitalize(), width=text_wrap))
                    if y_axis_label
                    else None
                ),
                fontsize=label_fontsize,
            )

            # Apply the title
            ax.set_title(
                "\n".join(textwrap.wrap(title, width=text_wrap)) if title else None,
                fontsize=label_fontsize,
            )

            ax.tick_params(
                axis="both", labelsize=tick_fontsize
            )  # Control tick fontsize separately

            # Set axis limits if specified
            if xlim is not None:
                ax.set_xlim(xlim)

            if ylim is not None:
                ax.set_ylim(ylim)

            # Disable scientific notation if requested
            if disable_sci_notation:
                ax.xaxis.set_major_formatter(
                    mticker.ScalarFormatter(useMathText=False),
                )
                ax.yaxis.set_major_formatter(
                    mticker.ScalarFormatter(useMathText=False),
                )

    # Hide any remaining axes
    for ax in axes[len(vars_of_interest) :]:
        ax.axis("off")

    # Adjust layout with specified padding
    plt.tight_layout(w_pad=w_pad, h_pad=h_pad)

    # Save files if paths are provided
    if image_path_png or image_path_svg or image_filename:
        _save_figure(
            fig=fig,
            image_path_png=image_path_png,
            image_path_svg=image_path_svg,
            filename=image_filename,
            image_filename=image_filename,
        )
    plt.show()

    # Generate separate plots for each variable of interest if provided
    if vars_of_interest:
        for var in vars_of_interest:
            fig, ax = plt.subplots(figsize=figsize)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                title = f"Distribution of {var}"

                # Determine if log scale should be applied to this variable
                log_scale = var in log_scale_vars if log_scale_vars else False

                # Filter out non-positive values if log_scale is True
                data = df[df[var] > 0] if log_scale else df

                try:
                    if plot_type == "hist":
                        sns.histplot(
                            data=data,
                            x=var,
                            kde=False,
                            ax=ax,
                            hue=hue,
                            color=hist_color if hue is None and fill else None,
                            edgecolor=hist_edgecolor,
                            stat=stat.lower(),
                            fill=fill,
                            alpha=fill_alpha,  # Apply for transparency
                            log_scale=log_scale,
                            bins=bins,
                            binwidth=binwidth,
                            **kwargs,
                        )
                    elif plot_type == "density":
                        _plot_density_overlays(
                            ax=ax,
                            data=data,
                            col=var,
                            density_function=density_function,
                            density_fit=density_fit,
                            hue=hue,
                            log_scale=log_scale,
                            density_color=density_color,
                            **kwargs,
                        )
                    elif plot_type == "both":
                        sns.histplot(
                            data=data,
                            x=var,
                            kde=False,  # No need, since plot_type controls this
                            ax=ax,
                            hue=hue,
                            color=hist_color if hue is None and fill else None,
                            edgecolor=hist_edgecolor,
                            stat=stat.lower(),
                            fill=fill,
                            alpha=fill_alpha,  # Apply for transparency
                            log_scale=log_scale,
                            bins=bins,
                            binwidth=binwidth,
                            **kwargs,
                        )
                        _plot_density_overlays(
                            ax=ax,
                            data=data,
                            col=col,
                            density_function=density_function,
                            density_fit=density_fit,
                            hue=hue,
                            log_scale=log_scale,
                            density_color=density_color,
                            **kwargs,
                        )

                    # Plot mean as a vertical dotted line if plot_mean is True
                    if plot_mean:
                        mean_value = data[var].mean()
                        ax.axvline(
                            mean_value,
                            color=mean_color,
                            linestyle="--",
                            label="Mean",
                        )

                    # Plot median as vertical dotted line if plot_median is True
                    if plot_median:
                        median_value = data[var].median()
                        ax.axvline(
                            median_value,
                            color=median_color,
                            linestyle="--",
                            label="Median",
                        )

                    # Plot std. deviation bands if std_dev_levels is specified
                    if std_dev_levels:
                        std_value = data[var].std()
                        for level, color in zip(std_dev_levels, std_color):
                            ax.axvline(
                                mean_value + level * std_value,
                                color=color,
                                linestyle="--",
                            )
                            ax.axvline(
                                mean_value - level * std_value,
                                color=color,
                                linestyle="--",
                                label=f"±{level} Std Dev",
                            )

                    # After plotting logic, before showing the legend
                    handles, labels = ax.get_legend_handles_labels()

                    if show_legend and len(handles) > 0 and len(labels) > 0:
                        ax.legend(loc=legend_loc)
                    else:
                        if ax.get_legend() is not None:
                            ax.get_legend().remove()

                except Exception as e:
                    # Handle different Python versions or issues w/ legends & labels
                    if "No artists with labels found to put in legend." in str(e):
                        print(f"Warning encountered while plotting '{var}': {str(e)}")
                        handles, labels = ax.get_legend_handles_labels()
                        if show_legend and len(handles) > 0 and len(labels) > 0:
                            ax.legend(loc=legend_loc)
                        else:
                            if ax.get_legend() is not None:
                                ax.get_legend().remove()

                finally:
                    pass  # Ensures the try block is valid

                xlabel = (
                    custom_xlabels[var]
                    if custom_xlabels and var in custom_xlabels
                    else f"{get_label(var)} (Log)" if log_scale else get_label(var)
                )

                ax.set_xlabel(
                    (
                        "\n".join(textwrap.wrap(xlabel, width=text_wrap))
                        if xlabel
                        else None
                    ),
                    fontsize=label_fontsize,
                )

                ax.set_ylabel(
                    (
                        "\n".join(
                            textwrap.wrap(y_axis_label.capitalize(), width=text_wrap)
                        )
                        if y_axis_label
                        else None
                    ),
                    fontsize=label_fontsize,
                )
                ax.set_title(
                    "\n".join(textwrap.wrap(title, width=text_wrap)) if title else None,
                    fontsize=label_fontsize,
                )
                ax.tick_params(
                    axis="both", labelsize=tick_fontsize
                )  # Control tick fontsize separately

                # Set axis limits if specified~
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)

                # Disable scientific notation if requested
                if disable_sci_notation:
                    ax.xaxis.set_major_formatter(
                        mticker.ScalarFormatter(useMathText=False)
                    )
                    ax.yaxis.set_major_formatter(
                        mticker.ScalarFormatter(useMathText=False)
                    )

            plt.tight_layout()

            # Save files for the variable of interest if a target is provided.
            # single_var_image_filename may be a bare base name (legacy: needs a
            # directory) or a full path with extension (saves verbatim, no dir).
            if single_var_image_filename:
                directory, base = os.path.split(single_var_image_filename)
                stem, ext = os.path.splitext(base)
                per_var_stem = f"{stem}_{var}"
                per_var_path = (
                    os.path.join(directory, f"{per_var_stem}{ext}")
                    if ext
                    else per_var_stem
                )
                _save_figure(
                    fig=fig,
                    image_path_png=image_path_png,
                    image_path_svg=image_path_svg,
                    image_filename=per_var_path,
                    bbox_inches=bbox_inches,
                )
            plt.close(
                fig
            )  # Close figure after saving to avoid displaying it multiple times


################################################################################
###################### Stacked Bar Plots W/ Crosstab Options ###################
################################################################################


def stacked_crosstab_plot(
    df: pd.DataFrame,
    col: str,
    func_col: List[str],
    legend_labels_list: List[List[str]],
    title: List[str],
    kind: str = "bar",
    width: float = 0.9,
    rot: int = 0,
    custom_order: Optional[List] = None,
    image_path_png: Optional[str] = None,
    image_path_svg: Optional[str] = None,
    save_formats: Optional[List[str]] = None,
    image_filename: Optional[str] = None,
    color: Optional[Union[List[str], str]] = None,
    output: str = "both",
    return_dict: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    outer_pad: float = 10.0,
    h_pad: float = 5.0,
    file_prefix: Optional[str] = None,
    logscale: bool = False,
    plot_type: str = "both",
    show_legend: bool = True,
    legend_loc: str = "best",
    reverse_legend: bool = True,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    title_wrap: int = 50,
    tick_wrap: int = 20,
    thousands_sep: bool = True,
    pct_format: bool = True,
    show_values: bool = False,
    subtitle: Optional[Union[str, List[str]]] = None,
    remove_stacks: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Generates stacked or regular bar plots and crosstabs for specified columns.

    This function allows users to create stacked bar plots (or regular bar plots
    if stacks are removed) and corresponding crosstabs for specific columns
    in a DataFrame. It provides options to customize the appearance, including
    font sizes for axis labels, tick labels, and title text wrapping, and to
    choose between regular or normalized plots.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.

    col : str
        The name of the column in the DataFrame to be analyzed.

    func_col : list of str
        List of columns in the DataFrame that will be used to generate the
        crosstabs and stack the bars in the plot.

    legend_labels_list : list of list of str
        List of legend labels corresponding to each column in `func_col`.

    title : list of str
        List of titles for each plot generated. Each entry is embedded in the
        auto-generated title string: "Prevalence of {title} by {col}".

    kind : str, optional (default='bar')
        The kind of plot to generate ('bar' or 'barh' for horizontal bars).

    width : float, optional (default=0.9)
        The width of the bars in the bar plot.

    rot : int, optional (default=0)
        The rotation angle of the x-axis labels.

    custom_order : list, optional
        Specifies a custom order for the categories in `col`.

    image_path_png : str, optional
        Directory path where generated PNG plot images will be saved.

    image_path_svg : str, optional
        Directory path where generated SVG plot images will be saved.

    image_filename : str, optional
        Full destination path for the saved plot(s). If it includes an
        extension, that extension selects the format unless ``save_formats``
        is given; the file is saved verbatim and no directory is required. The
        ``func_col`` entry is appended to the stem so each generated figure
        stays distinct. When provided, ``image_filename`` takes precedence over
        the ``image_path_png`` / ``image_path_svg`` directory scheme.

    save_formats : list of str, optional (default=None)
        File formats to write ('png' and/or 'svg'). With the
        ``image_path_png`` / ``image_path_svg`` scheme these select which
        directories are written. With ``image_filename`` they select which
        formats are written; if omitted there, the format is taken from the
        filename's extension (defaulting to 'png'). If neither a directory nor
        ``image_filename`` is given, nothing is saved.

    color : list of str, optional
        List of colors to use for the plots. If not provided, a default
        color scheme is used.

    output : str, optional (default='both')
        Specify the output type: "plots_only", "crosstabs_only", or "both".

    return_dict : bool, optional (default=False)
        Specify whether to return the crosstabs dictionary.

    figsize : tuple of int, optional
        A (width, height) tuple controlling the figure size in inches.
        Defaults to (12, 8) if not provided.

    outer_pad : float, optional (default=10.0)
        The outer padding around the entire figure, passed to
        `tight_layout(pad=...)`.

    h_pad : float, optional (default=5.0)
        The vertical padding between subplots in inches, passed to
        `tight_layout(h_pad=...)`. Increase this if the title of the lower
        plot overlaps the x-axis labels of the upper plot.

    file_prefix : str, optional
        Prefix for the filename when output includes plots.

    logscale : bool, optional (default=False)
        Apply log scale to the y-axis.

    plot_type : str, optional (default='both')
        Specify the type of plot to generate: "both", "regular", or "normalized".

    show_legend : bool, optional (default=True)
        Specify whether to show the legend.

    legend_loc : str, optional (default="best")
        Location of the legend, passed directly to Matplotlib's `legend(loc=...)`.
        Common options include "best", "upper right", "upper left", "lower left",
        and "lower right".

    reverse_legend : bool, optional (default=True)
        If True, reverses the legend entry order so it matches the visual
        stack order (top-to-bottom in the legend matches top-to-bottom in
        the bars). Matplotlib's default order is the inverse of the stack.

    label_fontsize : int, optional (default=12)
        Font size for axis labels and titles.

    tick_fontsize : int, optional (default=10)
        Font size for tick labels on the axes.

    title_wrap : int, optional (default=50)
        The maximum character width of the plot title text before wrapping
        to a new line.

    tick_wrap : int, optional (default=20)
        The maximum character width of axis tick label text before wrapping
        to a new line. Applied to x-axis tick labels for vertical bar plots
        and y-axis tick labels for horizontal bar plots.

    thousands_sep : bool, optional (default=True)
        If True, applies a thousands separator (e.g. 1,000,000) to the count
        axis of non-normalized plots. For vertical bar plots this is the
        y-axis; for horizontal bar plots (`kind='barh'`) this is the x-axis.
        Has no effect on normalized plots, which display percentages.

    pct_format : bool, optional (default=True)
        If True, formats the count axis of normalized plots as percentages
        (e.g. 0.25 → 25%). For vertical bar plots this is the y-axis; for
        horizontal bar plots (`kind='barh'`) this is the x-axis.

    show_values : bool, optional (default=False)
        If True, annotates each bar segment with its value. Regular plots
        show counts; normalized plots show percentages. Segments too small
        to fit a label are skipped automatically.

    subtitle : str or list of str, optional
        Subtitle text displayed as italic text below each figure. Accepts
        either a single string (applied to all plots) or a list of strings
        with one entry per `func_col`. If a list, it must match the length
        of `func_col`.

    remove_stacks : bool, optional (default=False)
        If True, removes stacks and creates a regular bar plot using only
        the `col` parameter. Only works when `plot_type` is set to 'regular'.

    xlim : tuple, optional
        Tuple specifying the limits of the x-axis.

    ylim : tuple, optional
        Tuple specifying the limits of the y-axis.

    Returns:
    --------
    crosstabs_dict : dict
        Dictionary of crosstabs DataFrames if `return_dict` is True.

    None
        If `return_dict` is False.

    Raises:
    -------
    ValueError
        If `remove_stacks` is used when `plot_type` is not set to "regular".

    ValueError
        If `output` is not one of ["both", "plots_only", "crosstabs_only"].

    ValueError
        If `plot_type` is not one of ["both", "regular", "normalized"].

    ValueError
        If the lengths of `title`, `func_col`, and `legend_labels_list` are not
        equal.

    ValueError
        If `subtitle` is provided and its length does not match `func_col`.

    KeyError
        If any columns in `col` or `func_col` are missing in the DataFrame.

    ValueError
        If a format in `save_formats` is not an available save target (no
        corresponding image path, or not derivable from `image_filename`).
    """

    # Check if remove_stacks is used correctly
    if remove_stacks and plot_type != "regular":
        raise ValueError(
            "`remove_stacks` can only be used when `plot_type` is set to 'regular'."
        )

    # Check if the output parameter is valid
    valid_outputs = ["both", "plots_only", "crosstabs_only"]
    if output not in valid_outputs:
        raise ValueError(
            f"Invalid output type: {output}. Valid options are {valid_outputs}"
        )

    # Check if the plot_type parameter is valid
    valid_plot_types = ["both", "regular", "normalized"]
    if plot_type not in valid_plot_types:
        raise ValueError(
            f"Invalid plot type: {plot_type}. Valid options are {valid_plot_types}"
        )

    # Normalize subtitle to a list if a bare string is passed
    if isinstance(subtitle, str):
        subtitle = [subtitle] * len(func_col)

    # Validate subtitle length if provided
    if subtitle is not None and len(subtitle) != len(func_col):
        raise ValueError(
            f"Length mismatch: `subtitle` has {len(subtitle)} entries but "
            f"`func_col` has {len(func_col)}. They must be equal."
        )
    # Ensure save_formats is a list even if None, string, or tuple is passed
    save_formats = save_formats or []
    if isinstance(save_formats, str):
        save_formats = [save_formats]
    elif isinstance(save_formats, tuple):
        save_formats = list(save_formats)

    # Initialize the dictionary to store crosstabs
    crosstabs_dict = {}

    # Default color settings
    if color is None:
        color = ["#00BFC4", "#F8766D"]  # Default colors
    elif isinstance(color, str):
        color = [color]  # Ensure a single color is passed as a list

    # Check if all required columns are present in the DataFrame
    missing_cols = [
        col_name for col_name in [col] + func_col if col_name not in df.columns
    ]
    if missing_cols:
        raise KeyError(f"Columns missing in DataFrame: {missing_cols}")

    if not (len(title) == len(func_col) == len(legend_labels_list)):
        raise ValueError(
            f"Length mismatch: Ensure that the lengths of `title`, `func_col`, "
            f"and `legend_labels_list` are equal. Current lengths are: "
            f"title={len(title)}, func_col={len(func_col)}, "
            f"legend_labels_list={len(legend_labels_list)}. "
            "Check for missing items or commas."
        )

    # Work on a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Setting custom order if provided
    if custom_order:
        df_copy[col] = pd.Categorical(
            df_copy[col], categories=custom_order, ordered=True
        )
        df_copy.sort_values(by=col, inplace=True)

    # Generate plots if output is "both" or "plots_only"
    if output in ["both", "plots_only"]:
        if file_prefix is None:
            file_prefix = f"{col}_{'_'.join(func_col)}"

        # Set default values for figsize if not provided
        if figsize is None:
            figsize = (12, 8)

        # Determine the number of subplots based on the plot_type parameter
        if plot_type == "both":
            nrows = 2
        else:
            nrows = 1

        # Loop through each condition and create the plots
        for idx, (truth, legend, tit) in enumerate(
            zip(func_col, legend_labels_list, title)
        ):
            image_path = {}

            if image_filename:
                # image_filename drives the destination; save_formats (or the
                # filename's own extension) drive which formats are written
                if_dir, if_base = os.path.split(image_filename)
                if_stem, if_ext = os.path.splitext(if_base)
                base_stem = f"{if_stem}_{truth}"
                fmts = save_formats or ([if_ext.lstrip(".")] if if_ext else ["png"])
                for fmt in fmts:
                    image_path[fmt] = (
                        os.path.join(if_dir, f"{base_stem}.{fmt}")
                        if if_dir
                        else f"{base_stem}.{fmt}"
                    )
            else:
                if image_path_png:
                    image_path["png"] = os.path.join(
                        image_path_png, f"{file_prefix}_{truth}.png"
                    )
                if image_path_svg:
                    image_path["svg"] = os.path.join(
                        image_path_svg, f"{file_prefix}_{truth}.svg"
                    )

            fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize)

            if remove_stacks:
                # Create a regular bar plot using only the `col` parameter
                counts = df_copy[col].value_counts()
                title1 = f"Distribution of {col.replace('_', ' ').title()}"
                xlabel1 = f"{col.replace('_', ' ')}"
                ylabel1 = "Count"
                counts.plot(
                    kind=kind,
                    ax=axes[0] if plot_type == "both" else axes,
                    color=color[0],
                    width=width,
                    rot=rot,
                    fontsize=label_fontsize,
                    logy=logscale,
                )
                ax0 = axes[0] if plot_type == "both" else axes

                if kind == "barh":
                    ax0.set_xlabel(ylabel1, fontsize=label_fontsize)
                    ax0.set_ylabel(xlabel1, fontsize=label_fontsize)
                else:
                    ax0.set_xlabel(xlabel1, fontsize=label_fontsize)
                    ax0.set_ylabel(ylabel1, fontsize=label_fontsize)

                # Set axis limits if specified
                if xlim:
                    ax0.set_xlim(xlim)
                if ylim:
                    ax0.set_ylim(ylim)

                ax0.set_title(
                    "\n".join(textwrap.wrap(title1, width=title_wrap)),
                    fontsize=label_fontsize,
                )
                ax0.tick_params(axis="both", labelsize=tick_fontsize)

                # Apply thousands separator to the count axis
                if thousands_sep:
                    fmt = mticker.FuncFormatter(lambda val, _: f"{val:,.0f}")
                    if kind == "barh":
                        ax0.xaxis.set_major_formatter(fmt)
                    else:
                        ax0.yaxis.set_major_formatter(fmt)

                # Apply tick_wrap to tick labels
                if kind == "barh":
                    ax0.set_yticklabels(
                        [
                            "\n".join(textwrap.wrap(str(l), width=tick_wrap))
                            for l in counts.index
                        ],
                        fontsize=tick_fontsize,
                    )
                else:
                    ax0.set_xticklabels(
                        [
                            "\n".join(textwrap.wrap(str(l), width=tick_wrap))
                            for l in counts.index
                        ],
                        rotation=rot,
                        fontsize=tick_fontsize,
                    )

                # Annotate bars with counts
                if show_values:
                    for bar_idx, val in enumerate(counts):
                        if kind == "barh":
                            ax0.text(
                                val,
                                bar_idx,
                                f" {int(val):,}",
                                ha="left",
                                va="center",
                                fontsize=tick_fontsize,
                                fontweight="bold",
                            )
                        else:
                            ax0.text(
                                bar_idx,
                                val,
                                f"{int(val):,}",
                                ha="center",
                                va="bottom",
                                fontsize=tick_fontsize,
                                fontweight="bold",
                            )

                if show_legend:
                    _apply_legend(
                        ax0, legend, legend_loc, label_fontsize, reverse_legend
                    )
                else:
                    leg = ax0.get_legend()
                    if leg is not None:
                        leg.remove()

            else:
                # Define crosstabdest to avoid UnboundLocalError
                crosstabdest = pd.crosstab(df_copy[col], df_copy[truth])
                try:
                    crosstabdest.columns = legend  # Rename columns
                except ValueError:
                    raise ValueError(
                        f"Length mismatch: Crosstab columns "
                        f"({len(crosstabdest.columns)}) and legend "
                        f"({len(legend)}). Check the length of your "
                        f"`legend_labels_list`, `func_col`, and `title` to ensure "
                        f"you are not missing an item, comma, or have an extra "
                        f"item."
                    )

                if plot_type in ["both", "regular"]:
                    # Plot the first graph (absolute counts)
                    title1 = f"Prevalence of {tit} by {col.replace('_', ' ').title()}"
                    xlabel1 = f"{col.replace('_', ' ').title()}"
                    ylabel1 = "Count"
                    crosstabdest.plot(
                        kind=kind,
                        stacked=True,
                        ax=axes[0] if plot_type == "both" else axes,
                        color=color,
                        width=width,
                        rot=rot,
                        fontsize=label_fontsize,
                    )

                    ax0 = axes[0] if plot_type == "both" else axes
                    ax0.set_title(
                        "\n".join(textwrap.wrap(title1, width=title_wrap)),
                        fontsize=label_fontsize,
                    )

                    if kind == "barh":
                        ax0.set_xlabel(ylabel1, fontsize=label_fontsize)
                        ax0.set_ylabel(xlabel1, fontsize=label_fontsize)
                    else:
                        ax0.set_xlabel(xlabel1, fontsize=label_fontsize)
                        ax0.set_ylabel(ylabel1, fontsize=label_fontsize)

                    # Set axis limits if specified
                    if xlim:
                        ax0.set_xlim(xlim)
                    if ylim:
                        ax0.set_ylim(ylim)

                    ax0.tick_params(axis="both", labelsize=tick_fontsize)

                    # Apply thousands separator to the count axis
                    if thousands_sep:
                        fmt = mticker.FuncFormatter(lambda val, _: f"{val:,.0f}")
                        if kind == "barh":
                            ax0.xaxis.set_major_formatter(fmt)
                        else:
                            ax0.yaxis.set_major_formatter(fmt)

                    # Apply tick_wrap to tick labels
                    if kind == "barh":
                        ax0.set_yticklabels(
                            [
                                "\n".join(textwrap.wrap(str(l), width=tick_wrap))
                                for l in crosstabdest.index
                            ],
                            fontsize=tick_fontsize,
                        )
                    else:
                        ax0.set_xticklabels(
                            [
                                "\n".join(textwrap.wrap(str(l), width=tick_wrap))
                                for l in crosstabdest.index
                            ],
                            rotation=rot,
                            fontsize=tick_fontsize,
                        )

                    # Annotate stacked bars with counts
                    if show_values:
                        _annotate_stacked(
                            ax0,
                            crosstabdest,
                            lambda v: f"{int(v):,}",
                            kind,
                            tick_fontsize,
                        )

                    if show_legend:
                        _apply_legend(
                            ax0, legend, legend_loc, label_fontsize, reverse_legend
                        )
                    else:
                        leg = ax0.get_legend()
                        if leg is not None:
                            leg.remove()

                if plot_type in ["both", "normalized"]:
                    # Plotting the second, normalized stacked bar graph
                    title2 = (
                        f"Prevalence of {tit} by {col.replace('_', ' ').title()} "
                        f"(Normalized)"
                    )
                    xlabel2 = f"{col.replace('_', ' ').title()}"
                    ylabel2 = "Percentage"
                    crosstabdestnorm = crosstabdest.div(
                        crosstabdest.sum(1),
                        axis=0,
                    )
                    crosstabdestnorm.plot(
                        kind=kind,
                        stacked=True,
                        ylabel="Percentage",
                        ax=axes[1] if plot_type == "both" else axes,
                        color=color,
                        width=width,
                        rot=rot,
                        fontsize=label_fontsize,
                        logy=logscale,
                    )

                    ax1 = axes[1] if plot_type == "both" else axes
                    ax1.set_title(
                        "\n".join(textwrap.wrap(title2, width=title_wrap)),
                        fontsize=label_fontsize,
                    )

                    if kind == "barh":
                        ax1.set_xlabel(ylabel2, fontsize=label_fontsize)
                        ax1.set_ylabel(xlabel2, fontsize=label_fontsize)
                    else:
                        ax1.set_xlabel(xlabel2, fontsize=label_fontsize)
                        ax1.set_ylabel(ylabel2, fontsize=label_fontsize)

                    # Set axis limits if specified
                    if xlim:
                        ax1.set_xlim(xlim)
                    if ylim:
                        ax1.set_ylim(ylim)

                    ax1.tick_params(axis="both", labelsize=tick_fontsize)

                    # Apply percentage formatter to the normalized count axis
                    if pct_format:
                        pct_fmt = mticker.FuncFormatter(lambda val, _: f"{val:.0%}")
                        if kind == "barh":
                            ax1.xaxis.set_major_formatter(pct_fmt)
                        else:
                            ax1.yaxis.set_major_formatter(pct_fmt)

                    # Apply tick_wrap to tick labels
                    if kind == "barh":
                        ax1.set_yticklabels(
                            [
                                "\n".join(textwrap.wrap(str(l), width=tick_wrap))
                                for l in crosstabdestnorm.index
                            ],
                            fontsize=tick_fontsize,
                        )
                    else:
                        ax1.set_xticklabels(
                            [
                                "\n".join(textwrap.wrap(str(l), width=tick_wrap))
                                for l in crosstabdestnorm.index
                            ],
                            rotation=rot,
                            fontsize=tick_fontsize,
                        )

                    # Annotate stacked bars with percentages
                    if show_values:
                        _annotate_stacked(
                            ax1,
                            crosstabdestnorm,
                            lambda v: f"{v:.1%}",
                            kind,
                            tick_fontsize,
                        )

                    if show_legend:
                        _apply_legend(
                            ax1, legend, legend_loc, label_fontsize, reverse_legend
                        )
                    else:
                        leg = ax1.get_legend()
                        if leg is not None:
                            leg.remove()

            fig.align_ylabels()
            fig.tight_layout(w_pad=5, pad=outer_pad, h_pad=h_pad)

            # Add subtitle below the figure if provided
            if subtitle is not None:
                fig.text(
                    0.5,
                    -0.02,
                    subtitle[idx],
                    ha="center",
                    va="top",
                    fontsize=label_fontsize - 2,
                    style="italic",
                )

            # Ensure save_formats is a list even if a string or tuple is passed
            if isinstance(save_formats, str):
                save_formats = [save_formats]
            elif isinstance(save_formats, tuple):
                save_formats = list(save_formats)

            # Determine which formats are valid targets
            if image_filename:
                valid_formats = list(image_path.keys())
            else:
                valid_formats = []
                if image_path_png:
                    valid_formats.append("png")
                if image_path_svg:
                    valid_formats.append("svg")

            # Throw an error if an invalid format is specified
            for save_format in save_formats:
                if save_format not in valid_formats:
                    missing_path = f"image_path_{save_format}"
                    raise ValueError(
                        f"Invalid save format '{save_format}'. To save in this "
                        f"format, you must first pass input for '{missing_path}'. "
                        f"Valid options are: {valid_formats}"
                    )

            # image_filename writes every resolved target; the legacy dir path
            # still filters by save_formats exactly as before
            if image_filename:
                to_save = list(image_path.values())
            else:
                to_save = [image_path[f] for f in save_formats if f in image_path]

            for full_path in to_save:
                directory = os.path.dirname(full_path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                plt.savefig(full_path, bbox_inches="tight")
                print(f"Plot saved as {full_path}")

            plt.show()

    # Always generate crosstabs if return_dict=True
    if return_dict:
        legend_counter = 0
        # First run of the crosstab, accounting for totals only
        for col_results in func_col:
            crosstab_df = pd.crosstab(
                df_copy[col],
                df_copy[col_results],
                margins=True,
                margins_name="Total",
            )
            # Rename columns
            crosstab_df.rename(
                columns={
                    **{
                        col: legend_labels_list[legend_counter][i]
                        for i, col in enumerate(crosstab_df.columns)
                        if col != "Total"
                    },
                    "Total": "Total",
                },
                inplace=True,
            )
            # Re-do the crosstab, this time, accounting for normalized data
            crosstab_df_norm = pd.crosstab(
                df_copy[col],
                df_copy[col_results],
                normalize="index",
                margins=True,
                margins_name="Total",
            )
            crosstab_df_norm = crosstab_df_norm.mul(100).round(2)
            crosstab_df_norm.rename(
                columns={
                    **{
                        col: f"{legend_labels_list[legend_counter][i]}_%"
                        for i, col in enumerate(crosstab_df_norm.columns)
                        if col != "Total"
                    },
                    "Total": "Total_%",
                },
                inplace=True,
            )
            crosstab_df = pd.concat([crosstab_df, crosstab_df_norm], axis=1)
            legend_counter += 1
            crosstabs_dict[col_results] = crosstab_df

    # Display crosstabs only if required by output
    if output in ["both", "crosstabs_only"]:
        for col_results, crosstab_df in crosstabs_dict.items():
            print()
            print("Crosstab for " + col_results)
            print()
            print(crosstab_df)
            print()

    # Return the crosstabs_dict only if return_dict is True
    if return_dict:
        return crosstabs_dict


################################################################################
############################ Box and Violin Plots ##############################
################################################################################


def box_violin_plot(
    df: pd.DataFrame,
    metrics_list: List[str],
    metrics_comp: List[str],
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    image_path_png: Optional[str] = None,
    image_path_svg: Optional[str] = None,
    image_filename: Optional[str] = None,
    show_legend: bool = True,
    legend_loc: str = "best",
    plot_type: str = "boxplot",
    xlabel_rot: int = 0,
    show_plot: str = "both",
    rotate_plot: bool = False,
    custom_order: Optional[List] = None,
    individual_figsize: Tuple[int, int] = (6, 4),
    subplot_figsize: Optional[Tuple[int, int]] = None,
    suptitle: Optional[str] = None,
    suptitle_y: float = 1.02,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    text_wrap: int = 50,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Create and save individual or subplot-based boxplots or violin plots for
    specified metrics and comparisons.

    This function generates individual plots, subplots, or both for the
    specified metrics and comparison categories in a DataFrame. It provides
    extensive customization options for plot appearance, saving options, and
    display preferences, including support for axis limits, label customization,
    and rotated layouts.

    Figures are saved only when ``image_filename`` is provided. If it includes
    an extension the figure is saved verbatim with no directory required;
    otherwise it is used as a stem with ``image_path_png`` / ``image_path_svg``.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.

    metrics_list : list of str
        List of column names representing the metrics to be plotted.

    metrics_comp : list of str or str
        List of column names representing the comparison categories. A plain
        string is automatically coerced to a single-element list.

    n_rows : int, optional
        Number of rows in the subplot grid. Calculated automatically if not
        provided.

    n_cols : int, optional
        Number of columns in the subplot grid. Calculated automatically if not
        provided.

    image_path_png : str, optional
        Directory path to save plots in PNG format. If not specified, plots
        will not be saved as PNG.

    image_path_svg : str, optional
        Directory path to save plots in SVG format. If not specified, plots
        will not be saved as SVG.

    image_filename : str, optional
        Filename used when saving figures. No files are saved if this is not
        provided. May be a bare stem (legacy: requires ``image_path_png`` /
        ``image_path_svg``) or a full path with extension (saved verbatim, no
        directory needed). For individual plots the metric name and plot type
        are appended to the stem; for the subplot grid the plot type is
        appended. The extension, if present, is preserved.

    show_legend : bool, optional (default=True)
        Whether to display the legend on the plots.

    legend_loc : str, optional (default="best")
        Location of the legend, passed to Matplotlib's legend(loc=...).

    plot_type : str, optional (default='boxplot')
        Type of plot to generate. Options are "boxplot" or "violinplot".
        Shorthand aliases "box" and "violin" are also accepted.

    xlabel_rot : int, optional (default=0)
        Rotation angle for x-axis labels.

    show_plot : str, optional (default='both')
        Specify the type of plots to display. Options are "individual",
        "subplots", or "both".

    rotate_plot : bool, optional (default=False)
        If True, rotates the plots by swapping the x and y axes.

    custom_order : list, optional
        Specifies the display order of categories along the comparison axis.
        For example, passing ``["B", "A", "C"]`` will render group "B" first
        regardless of the order it appears in the data. If None, the default
        order observed in the data is used. For horizontal plots
        (``rotate_plot=True``), this controls the y-axis order.

    individual_figsize : tuple of int, optional (default=(6, 4))
        Dimensions (width, height) for individual plots.

    subplot_figsize : tuple of int, optional
        Dimensions (width, height) of the subplots. Defaults to a size
        proportional to the number of rows and columns.

    suptitle : str, optional
        Figure-level title rendered above the subplot grid via
        ``fig.suptitle``. Font size is controlled by ``label_fontsize``.
        If None, no figure-level title is added. Only applies to the
        subplot grid, not individual plots.

    suptitle_y : float, optional (default=1.02)
        Vertical position of the figure-level suptitle as a fraction of the
        figure height. Values above 1.0 place the title above the subplot
        grid. Increase to add more space between the title and the plots.

    label_fontsize : int, optional (default=12)
        Font size for axis labels.

    tick_fontsize : int, optional (default=10)
        Font size for axis tick labels.

    text_wrap : int, optional (default=50)
        Maximum number of characters allowed in plot titles and axis labels
        before wrapping them onto the next line.

    xlim : tuple of float, optional
        Limits for the x-axis as (min, max).

    ylim : tuple of float, optional
        Limits for the y-axis as (min, max).

    **kwargs : additional keyword arguments
        Additional parameters passed to the Seaborn plotting function.
        Note: layout-related keys ("figsize", "subplot_figsize",
        "individual_figsize") are automatically stripped and will not be
        forwarded to Seaborn. If ``palette`` is passed here, it takes
        precedence over the internal default palette and is applied
        consistently across all subplots.

    Returns:
    --------
    None
        This function does not return a value. It generates and optionally
        saves or displays the specified plots.

    Raises:
    -------
    ValueError
        - If ``show_plot`` is not one of "individual", "subplots", or "both".
        - If ``rotate_plot`` is not a boolean value.
        - If ``individual_figsize`` or ``subplot_figsize`` is not a tuple or
          list of two numbers.
        - If ``plot_type`` is not one of "boxplot", "violinplot", "box",
          or "violin".
        - If ``image_filename`` is provided without an extension and neither
          ``image_path_png`` nor ``image_path_svg`` is specified.

    Notes:
    ------
    - Automatically calculates ``n_rows`` and ``n_cols`` if not provided based
      on the number of plots.
    - Supports rotating the plot layout using the ``rotate_plot`` parameter.
    - Figures are saved via ``_save_figure`` only when ``image_filename`` is
      provided.
    - If ``palette`` is supplied via ``**kwargs``, it is resolved once before
      any plotting loop and applied consistently to all subplots.
    - ``custom_order`` is passed directly to Seaborn's ``order`` parameter
      (or ``x_order`` / ``y_order`` depending on orientation).
    - ``suptitle`` applies only to the subplot grid. Individual plots use
      per-subplot titles only.
    - Tick labels on the numeric (metric) axis are formatted with thousands
      separators. The categorical comparison axis is left unformatted.
    """

    # Coerce metrics_comp string to list
    if isinstance(metrics_comp, str):
        metrics_comp = [metrics_comp]

    # Normalize plot_type shorthands before validation
    _plot_type_aliases = {
        "violin": "violinplot",
        "box": "boxplot",
    }
    plot_type = _plot_type_aliases.get(plot_type, plot_type)

    if plot_type not in ("boxplot", "violinplot"):
        raise ValueError(
            f"Invalid `plot_type` '{plot_type}'. "
            "Choose from 'boxplot', 'violinplot', 'box', or 'violin'."
        )

    # Strip layout kwargs before passing to seaborn
    _layout_keys = {"figsize", "subplot_figsize", "individual_figsize"}
    seaborn_kwargs = {k: v for k, v in kwargs.items() if k not in _layout_keys}

    # Resolve caller-supplied palette once, before any loop
    user_palette = seaborn_kwargs.pop("palette", None)

    # Check for valid show_plot values
    if show_plot not in ["individual", "subplots", "both"]:
        raise ValueError(
            "Invalid `show_plot` value selected. Choose from 'individual', "
            "'subplots', or 'both'."
        )

    # Check for valid rotate_plot values
    if not isinstance(rotate_plot, bool):
        raise ValueError(
            "Invalid `rotate_plot` value selected. Choose from 'True' or 'False'."
        )

    # Check for valid individual_figsize values
    if not (
        isinstance(individual_figsize, (tuple, list))
        and len(individual_figsize) == 2
        and all(isinstance(x, (int, float)) for x in individual_figsize)
    ):
        raise ValueError(
            "Invalid `individual_figsize` value. It should be a tuple or list "
            "of two numbers (width, height)."
        )

    # Check for valid subplot_figsize values if specified
    if subplot_figsize is not None and not (
        isinstance(subplot_figsize, (tuple, list))
        and len(subplot_figsize) == 2
        and all(isinstance(x, (int, float)) for x in subplot_figsize)
    ):
        raise ValueError(
            "Invalid `subplot_figsize` value. It should be a tuple or list of "
            "two numbers (width, height)."
        )

    # Validate image_filename usage
    if (
        image_filename is not None
        and not os.path.splitext(image_filename)[1]
        and not (image_path_png or image_path_svg)
    ):
        raise ValueError(
            "``image_filename`` was provided without a file extension and "
            "neither ``image_path_png`` nor ``image_path_svg`` was specified. "
            "Provide an extension (e.g. 'plot.png') or at least one output path."
        )

    total_plots = len(metrics_list) * len(metrics_comp)

    # Ensure at least one plot is possible
    if total_plots == 0:
        raise ValueError(
            "No valid plots to generate. Ensure `metrics_list` and "
            "`metrics_comp` are not empty."
        )

    # Respect whichever of n_rows / n_cols the caller provided
    if n_rows is None and n_cols is None:
        n_cols = max(1, int(np.ceil(np.sqrt(total_plots))))
        n_rows = max(1, int(np.ceil(total_plots / n_cols)))
    elif n_rows is None:
        n_rows = max(1, int(np.ceil(total_plots / n_cols)))
    elif n_cols is None:
        n_cols = max(1, int(np.ceil(total_plots / n_rows)))

    # Set default subplot figure size if not specified
    if subplot_figsize is None:
        subplot_figsize = (5 * n_cols, 5 * n_rows)

    # Map plot_type to the corresponding seaborn function
    plot_function = getattr(sns, plot_type)

    # Resolve the seaborn `order` kwarg from custom_order.
    # For vertical plots the comparison variable is on x; for rotated plots
    # it is on y. Seaborn uses `order` for the categorical axis in both cases.
    order_kwarg = {"order": custom_order} if custom_order is not None else {}

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------
    if show_plot in ["individual", "both"]:
        for met_comp in metrics_comp:
            if user_palette is not None:
                palette = user_palette
            else:
                unique_vals = df[met_comp].value_counts().count()
                palette = _get_palette(unique_vals)

            for met_list in metrics_list:
                fig = plt.figure(figsize=individual_figsize)
                ax = plot_function(
                    x=met_list if rotate_plot else met_comp,
                    y=met_comp if rotate_plot else met_list,
                    data=df,
                    hue=met_comp,
                    palette=palette,
                    dodge=False,
                    **order_kwarg,
                    **seaborn_kwargs,
                )

                title = (
                    f"Distribution of {_get_label(met_list)} "
                    f"by {_get_label(met_comp)}"
                )
                plt.title(
                    "\n".join(textwrap.wrap(title, width=text_wrap)),
                    fontsize=label_fontsize,
                )
                plt.xlabel(
                    "\n".join(
                        textwrap.wrap(
                            (
                                _get_label(met_list)
                                if rotate_plot
                                else _get_label(met_comp)
                            ),
                            width=text_wrap,
                        )
                    ),
                    fontsize=label_fontsize,
                )
                plt.ylabel(
                    "\n".join(
                        textwrap.wrap(
                            (
                                _get_label(met_comp)
                                if rotate_plot
                                else _get_label(met_list)
                            ),
                            width=text_wrap,
                        )
                    ),
                    fontsize=label_fontsize,
                )
                ax.tick_params(axis="x", rotation=xlabel_rot)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                _apply_thousands(ax, axis="x" if rotate_plot else "y")

                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)

                if show_legend and ax.legend_:
                    ax.legend(loc=legend_loc, fontsize=tick_fontsize)
                elif ax.legend_:
                    ax.legend_.remove()

                if image_filename is not None:
                    safe_met_list = (
                        met_list.replace(" ", "_")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("/", "_per_")
                    )
                    if_dir, if_base = os.path.split(image_filename)
                    if_stem, if_ext = os.path.splitext(if_base)
                    per_combo_stem = f"{if_stem}_{safe_met_list}"
                    per_combo_name = (
                        os.path.join(if_dir, f"{per_combo_stem}{if_ext}")
                        if if_ext
                        else per_combo_stem
                    )
                    _save_figure(
                        fig=fig,
                        image_path_png=image_path_png,
                        image_path_svg=image_path_svg,
                        image_filename=per_combo_name,
                    )

                plt.show()

    # ------------------------------------------------------------------
    # Subplot grid
    # ------------------------------------------------------------------
    if show_plot in ["subplots", "both"]:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=subplot_figsize)

        if suptitle is not None:
            fig.suptitle(suptitle, fontsize=label_fontsize, y=suptitle_y)

        if n_rows * n_cols == 1:
            axs = [axs]
        else:
            axs = axs.flatten()

        for i, ax in enumerate(axs):
            if i < total_plots:
                met_comp = metrics_comp[i // len(metrics_list)]
                met_list = metrics_list[i % len(metrics_list)]

                if user_palette is not None:
                    palette = user_palette
                else:
                    unique_vals = df[met_comp].value_counts().count()
                    palette = _get_palette(unique_vals)

                plot_function(
                    x=met_list if rotate_plot else met_comp,
                    y=met_comp if rotate_plot else met_list,
                    data=df,
                    hue=met_comp,
                    ax=ax,
                    palette=palette,
                    dodge=False,
                    **order_kwarg,
                    **seaborn_kwargs,
                )
                title = (
                    f"Distribution of {_get_label(met_list)} "
                    f"by {_get_label(met_comp)}"
                )
                ax.set_title(
                    "\n".join(textwrap.wrap(title, width=text_wrap)),
                    fontsize=label_fontsize,
                )
                ax.set_xlabel(
                    "\n".join(
                        textwrap.wrap(
                            (
                                _get_label(met_list)
                                if rotate_plot
                                else _get_label(met_comp)
                            ),
                            width=text_wrap,
                        )
                    ),
                    fontsize=label_fontsize,
                )
                ax.set_ylabel(
                    "\n".join(
                        textwrap.wrap(
                            (
                                _get_label(met_comp)
                                if rotate_plot
                                else _get_label(met_list)
                            ),
                            width=text_wrap,
                        )
                    ),
                    fontsize=label_fontsize,
                )
                ax.tick_params(axis="x", rotation=xlabel_rot)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                _apply_thousands(ax, axis="x" if rotate_plot else "y")

                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)

                if show_legend and ax.legend_:
                    ax.legend(loc=legend_loc, fontsize=tick_fontsize)
                elif ax.legend_:
                    ax.legend_.remove()

            else:
                ax.set_visible(False)

        plt.tight_layout()

        if image_filename is not None:
            if_dir, if_base = os.path.split(image_filename)
            if_stem, if_ext = os.path.splitext(if_base)
            if if_ext:
                grid_name = image_filename  # new: full path, use as-is
            else:
                grid_name = f"{if_stem}_{plot_type}"  # legacy: unchanged
            _save_figure(
                fig=fig,
                image_path_png=image_path_png,
                image_path_svg=image_path_svg,
                image_filename=grid_name,
            )

        plt.show()


################################################################################
########################## Multi-Purpose Scatter Plots #########################
################################################################################


def scatter_fit_plot(
    df: pd.DataFrame,
    x_vars: Optional[Union[List[str], str]] = None,
    y_vars: Optional[Union[List[str], str]] = None,
    all_vars: Optional[List[str]] = None,
    exclude_combinations: Optional[List[Tuple[str, str]]] = None,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    max_cols: int = 4,
    image_path_png: Optional[str] = None,
    image_path_svg: Optional[str] = None,
    image_filename: Optional[str] = None,
    save_plots: Optional[str] = None,
    show_legend: bool = True,
    legend_loc: str = "best",
    xlabel_rot: int = 0,
    show_plot: str = "subplots",
    rotate_plot: bool = False,
    individual_figsize: Tuple[int, int] = (6, 4),
    subplot_figsize: Optional[Tuple[int, int]] = None,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    text_wrap: int = 50,
    add_best_fit_line: bool = False,
    scatter_color: str = "C0",
    best_fit_linecolor: str = "red",
    best_fit_linestyle: str = "-",
    hue: Optional[str] = None,
    hue_palette: Optional[Union[Dict, List, str]] = None,
    size: Optional[str] = None,
    sizes: Optional[Dict] = None,
    marker: str = "o",
    show_correlation: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    label_names: Optional[Dict[str, str]] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Create and save scatter plots or a grid of scatter subplots for given
    x_vars and y_vars, with an optional best fit line, customizable
    point color, size, markers, and axis limits.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.

    x_vars : list of str or str, optional
        List of variable names to plot on the x-axis. If a string is provided,
        it will be converted into a list with one element.

    y_vars : list of str or str, optional
        List of variable names to plot on the y-axis. If a string is provided,
        it will be converted into a list with one element.

    all_vars : list of str, optional
        If provided, automatically generates scatter plots for all combinations
        of variables in this list, overriding x_vars and y_vars.

    exclude_combinations : list of tuples, optional
        List of (x_var, y_var) combinations to exclude from the plots.

    n_rows : int, optional
        Number of rows in the subplot grid. If not specified, it will be
        calculated based on the number of plots and n_cols.

    n_cols : int, optional
        Number of columns in the subplot grid. If not specified, it will be
        calculated based on the number of plots and max_cols.

    max_cols : int, optional (default=4)
        Maximum number of columns in the subplot grid.

    image_path_png : str, optional
        Directory path to save PNG images of the scatter plots.

    image_path_svg : str, optional
        Directory path to save SVG images of the scatter plots.

    image_filename : str, optional
        Destination for saved plots. May be a bare stem (legacy: requires
        ``image_path_png`` / ``image_path_svg``) or a full path with extension
        (saved verbatim, no directory needed). For individual saves the
        ``{x}_vs_{y}`` name is substituted into the stem so each file stays
        distinct; the subplot grid uses a single name. If None, the auto-
        generated name is used with the directory paths.

    save_plots : str, optional
        Controls which plots to save: "all", "individual", or "subplots".
        If None but an output target (`image_path_png`, `image_path_svg`, or
        `image_filename`) is provided, defaults to "subplots". If None with no
        target, plots are not saved.
        - "all": Saves both individual and subplots.
        - "individual": Saves each scatter plot separately with a progress bar
          (powered by `tqdm`) to track saving progress.
        - "subplots": Saves a single subplot grid of all combinations.

    show_legend : bool, optional (default=True)
        Whether to display the legend on the plots.

    legend_loc : str, default="best"
        Legend placement passed to Matplotlib.

    xlabel_rot : int, optional (default=0)
        Rotation angle for x-axis labels.

    show_plot : str, optional (default="subplots")
        Controls plot display: "individual", "subplots", or "both".

    rotate_plot : bool, optional (default=False)
        Whether to rotate (pivot) the plots.

    individual_figsize : tuple or list, optional (default=(6, 4))
        Width and height of the figure for individual plots.

    subplot_figsize : tuple or list, optional
        Width and height of the figure for subplots.
        If not specified, defaults to a calculated size based on the number of
        rows and columns.

    label_fontsize : int, optional (default=12)
        Font size for axis labels.

    tick_fontsize : int, optional (default=10)
        Font size for axis tick labels.

    text_wrap : int, optional (default=50)
        The maximum width of text (in characters) before wrapping. This applies
        to the plot title, x-axis labels, and y-axis labels.

    add_best_fit_line : bool, optional (default=False)
        Whether to add a best fit line to the scatter plots.

    scatter_color : str, optional (default="C0")
        Color code for the scattered points.

    best_fit_linecolor : str, optional (default="red")
        Color code for the best fit line.

    best_fit_linestyle : str, optional (default="-")
        Linestyle for the best fit line.

    hue : str, optional
        Column name for the grouping variable that will produce points with
        different colors.

    hue_palette : dict, list, or str, optional
        Specifies colors for each hue level. Can be a dictionary mapping hue
        levels to colors, a list of colors, or the name of a seaborn color
        palette.

    size : str, optional
        Column name for the grouping variable that will produce points with
        different sizes.

    sizes : dict, optional
        Dictionary mapping sizes (smallest and largest) to min and max values.

    marker : str, optional (default="o")
        Marker style used for the scatter points.

    show_correlation : bool, optional (default=True)
        Whether to display the Pearson correlation coefficient in the plot title.

    xlim : tuple or list, optional
        Limits for the x-axis as a tuple or list of (min, max).
        If not provided, the limits are determined automatically.

    ylim : tuple or list, optional
        Limits for the y-axis as a tuple or list of (min, max).
        If not provided, the limits are determined automatically.

    label_names : dict, optional
        A dictionary to rename columns for display in the plot titles and labels.

    **kwargs : dict, optional
        Additional keyword arguments to pass to sns.scatterplot.

    Returns:
    --------
    None
        This function does not return any value but generates and optionally
        saves scatter plots for the specified x_vars and y_vars.

    Raises:
    -------
    ValueError
        If `all_vars` is provided and either `x_vars` or `y_vars` is also
        provided.

    ValueError
        If neither `all_vars` nor both `x_vars` and `y_vars` are provided.

    ValueError
        If `hue_palette` is specified without `hue`.

    ValueError
        If `show_plot` is not one of ["individual", "subplots", "both", "combinations"].

    ValueError
        If `save_plots` is not one of [None, "all", "individual", "subplots"].

    ValueError
        If `save_plots` is set (or a save target is implied) without any of
        `image_path_png`, `image_path_svg`, or `image_filename`.

    ValueError
        If `rotate_plot` is not a boolean value.

    ValueError
        If `individual_figsize` is not a tuple or list of two numbers
        (width, height).

    ValueError
        If `subplot_figsize` is provided and is not a tuple or list of two numbers
        (width, height).

    ValueError
        If `exclude_combinations` contains invalid entries (i.e., items that
        are not tuples of exactly two elements).

    ValueError
        If any column names in `exclude_combinations` do not exist in the
        DataFrame.
    """

    # Ensure x_vars and y_vars are lists
    if isinstance(x_vars, str):
        x_vars = [x_vars]
    if isinstance(y_vars, str):
        y_vars = [y_vars]

    # Check for conflicting inputs of variable assignments
    if all_vars is not None and (x_vars is not None or y_vars is not None):
        raise ValueError(
            f"Cannot pass `all_vars` and still choose `x_vars` "
            f"and/or `y_vars`. Must choose either `x_vars` and "
            f"`y_vars` as inputs or `all_vars`."
        )

    # Check if hue_palette is provided without hue
    if hue_palette is not None and hue is None:
        raise ValueError(
            f"Cannot specify `hue_palette` without specifying `hue`. "
            f"Please provide the `hue` parameter or remove `hue_palette`."
        )

    # Generate combinations of x_vars and y_vars or use all_vars
    if all_vars:
        combinations = list(itertools.combinations(all_vars, 2))
    elif x_vars is not None and y_vars is not None:
        combinations = [(x_var, y_var) for x_var in x_vars for y_var in y_vars]
    else:
        raise ValueError(
            f"Either `all_vars` or both `x_vars` and `y_vars` must be provided."
        )

    # Validate exclude_combinations
    if exclude_combinations:
        # Validate exclude_combinations without modifying the original input
        normalized_exclude_combinations = {
            tuple(sorted(pair)) for pair in exclude_combinations
        }

        # Check if all columns in exclude_combinations exist in the DataFrame
        invalid_columns = {
            col
            for pair in normalized_exclude_combinations
            for col in pair
            if col not in df.columns
        }
        if invalid_columns:
            raise ValueError(
                f"Invalid column names in `exclude_combinations`: {invalid_columns}. "
                "Please ensure all columns exist in the DataFrame."
            )

        # Use normalized_exclude_combinations to filter combinations
        combinations = [
            (x, y)
            for (x, y) in combinations
            if tuple(sorted((x, y))) not in normalized_exclude_combinations
        ]

    # Handle show_plot="combinations"
    if show_plot == "combinations":
        return combinations

    # Calculate the number of plots
    num_plots = len(combinations)

    # Ensuring that `n_cols` and `n_rows` never become zero, preventing a
    # division by zero error.
    if n_cols is None or n_cols == 0:
        n_cols = min(num_plots, max_cols) if num_plots > 0 else 1
    if n_rows is None:
        n_rows = max(1, math.ceil(num_plots / n_cols))

    # Set default subplot grid figure size if not specified
    if subplot_figsize is None:
        subplot_figsize = (5 * n_cols, 5 * n_rows)

    # Validate the show_plot input
    valid_show_plot_values = ["individual", "subplots", "both", "combinations"]
    if show_plot not in valid_show_plot_values:
        raise ValueError(f"Invalid `show_plot`. Choose from {valid_show_plot_values}.")

    # Validate the save_plots input
    if save_plots not in [None, "all", "individual", "subplots"]:
        raise ValueError(
            "Invalid `save_plots` value. Choose from 'all', "
            "'individual', 'subplots', or None."
        )

    # If an output target is supplied but save_plots wasn't, default to subplots
    if save_plots is None and (image_path_png or image_path_svg or image_filename):
        save_plots = "subplots"

    # Check if save_plots is set without any output target
    if save_plots and not (image_path_png or image_path_svg or image_filename):
        raise ValueError(
            "To save plots, specify `image_path_png`, `image_path_svg`, "
            "or `image_filename`."
        )

    # Validate the rotate_plot input
    if not isinstance(rotate_plot, bool):
        raise ValueError("Invalid `rotate_plot`. Choose True or False.")

    # Validate the individual_figsize input
    if not (
        isinstance(individual_figsize, (tuple, list))
        and len(individual_figsize) == 2
        and all(isinstance(x, (int, float)) for x in individual_figsize)
    ):
        raise ValueError(
            "Invalid `individual_figsize` value. It should be a tuple or list "
            "of two numbers (width, height)."
        )

    # Validate the subplot_figsize input if specified
    if subplot_figsize is not None and not (
        isinstance(subplot_figsize, (tuple, list))
        and len(subplot_figsize) == 2
        and all(isinstance(x, (int, float)) for x in subplot_figsize)
    ):
        raise ValueError(
            "Invalid `subplot_figsize` value. It should be a tuple or list of two "
            "numbers (width, height)."
        )

    # Create grid for individual or subplots plotting
    _, axes = plt.subplots(n_rows, n_cols, figsize=subplot_figsize)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Render and show individual plots
    if show_plot in ["individual", "both"]:
        for x_var, y_var in combinations:
            plt.figure(figsize=individual_figsize)
            ax = sns.scatterplot(
                x=x_var if not rotate_plot else y_var,
                y=y_var if not rotate_plot else x_var,
                data=df,
                color=scatter_color if hue is None else None,
                hue=hue,
                palette=hue_palette,
                size=size,
                sizes=sizes,
                marker=marker,
                **kwargs,
            )
            if show_legend and ax.legend_:
                ax.legend(loc=legend_loc, fontsize=tick_fontsize)
            elif ax.legend_:
                ax.legend_.remove()

            if add_best_fit_line:
                x_data = df[x_var] if not rotate_plot else df[y_var]
                y_data = df[y_var] if not rotate_plot else df[x_var]
                _add_best_fit(
                    ax=ax,
                    x=x_data,
                    y=y_data,
                    linestyle=best_fit_linestyle,
                    linecolor=best_fit_linecolor,
                    show_legend=show_legend,
                    legend_loc=legend_loc,
                )

            r_value = df[x_var].corr(df[y_var])
            title = (
                f"{_get_label(x_var, label_names)} vs. {_get_label(y_var, label_names)}"
            )
            if show_correlation:
                title += f" ($r$ = {r_value:.2f})"
            plt.title(
                "\n".join(textwrap.wrap(title, width=text_wrap)),
                fontsize=label_fontsize,
            )
            plt.xlabel(
                "\n".join(
                    textwrap.wrap(
                        (
                            _get_label(x_var, label_names)
                            if not rotate_plot
                            else _get_label(y_var, label_names)
                        ),
                        width=text_wrap,
                    )
                ),
                fontsize=label_fontsize,
            )
            plt.ylabel(
                "\n".join(
                    textwrap.wrap(
                        (
                            _get_label(y_var, label_names)
                            if not rotate_plot
                            else _get_label(x_var, label_names)
                        ),
                        width=text_wrap,
                    )
                ),
                fontsize=label_fontsize,
            )
            ax.tick_params(axis="x", rotation=xlabel_rot)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            _apply_thousands(ax)

            # Apply xlim and ylim if provided
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)

            plt.show()

    # Render and show subplots
    if show_plot in ["subplots", "both"]:

        for i, ax in enumerate(axes):
            if i < num_plots:
                x_var, y_var = combinations[i]
                sns.scatterplot(
                    x=x_var if not rotate_plot else y_var,
                    y=y_var if not rotate_plot else x_var,
                    data=df,
                    ax=ax,
                    color=scatter_color,
                    hue=hue,
                    palette=hue_palette,
                    size=size,
                    sizes=sizes,
                    marker=marker,
                    **kwargs,
                )

                if show_legend and ax.legend_:
                    ax.legend(loc=legend_loc, fontsize=tick_fontsize)
                elif ax.legend_:
                    ax.legend_.remove()

                if add_best_fit_line:
                    x_data = df[x_var] if not rotate_plot else df[y_var]
                    y_data = df[y_var] if not rotate_plot else df[x_var]
                    _add_best_fit(
                        ax=ax,
                        x=x_data,
                        y=y_data,
                        linestyle=best_fit_linestyle,
                        linecolor=best_fit_linecolor,
                        show_legend=show_legend,
                        legend_loc=legend_loc,
                    )

                r_value = df[x_var].corr(df[y_var])
                title = f"{_get_label(x_var, label_names)} vs. {_get_label(y_var, label_names)}"
                if show_correlation:
                    title += f" ($r$ = {r_value:.2f})"
                ax.set_title(
                    "\n".join(textwrap.wrap(title, width=text_wrap)),
                    fontsize=label_fontsize,
                )
                ax.set_xlabel(
                    "\n".join(
                        textwrap.wrap(
                            (
                                _get_label(x_var, label_names)
                                if not rotate_plot
                                else _get_label(y_var, label_names)
                            ),
                            width=text_wrap,
                        )
                    ),
                    fontsize=label_fontsize,
                )
                ax.set_ylabel(
                    "\n".join(
                        textwrap.wrap(
                            (
                                _get_label(y_var, label_names)
                                if not rotate_plot
                                else _get_label(x_var, label_names)
                            ),
                            width=text_wrap,
                        )
                    ),
                    fontsize=label_fontsize,
                )
                ax.tick_params(axis="x", rotation=xlabel_rot)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                _apply_thousands(ax)

                # Apply xlim and ylim if provided
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)

            else:
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    # Save individual plots with progress bar
    if save_plots in ["all", "individual"]:
        with tqdm(total=len(combinations), desc="Saving scatter plot(s)") as pbar:
            for x_var, y_var in combinations:
                fig_individual, ax = plt.subplots(
                    figsize=individual_figsize
                )  # Use distinct figure for each plot
                sns.scatterplot(
                    x=x_var if not rotate_plot else y_var,
                    y=y_var if not rotate_plot else x_var,
                    data=df,
                    ax=ax,
                    color=scatter_color if hue is None else None,
                    hue=hue,
                    palette=hue_palette,
                    size=size,
                    sizes=sizes,
                    marker=marker,
                    **kwargs,
                )

                if add_best_fit_line:
                    x_data = df[x_var] if not rotate_plot else df[y_var]
                    y_data = df[y_var] if not rotate_plot else df[x_var]
                    _add_best_fit(
                        ax=ax,
                        x=x_data,
                        y=y_data,
                        linestyle=best_fit_linestyle,
                        linecolor=best_fit_linecolor,
                        show_legend=show_legend,
                        legend_loc=legend_loc,
                    )

                ax.set_title(
                    f"{_get_label(x_var, label_names)} vs. {_get_label(y_var, label_names)}",
                    fontsize=label_fontsize,
                )
                ax.tick_params(axis="x", rotation=xlabel_rot)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                _apply_thousands(ax)

                safe_x_var = x_var.replace(" ", "_").replace("/", "_per_")
                safe_y_var = y_var.replace(" ", "_").replace("/", "_per_")
                _save_figure(
                    fig=fig_individual,
                    image_path_png=image_path_png,
                    image_path_svg=image_path_svg,
                    image_filename=_resolve_save_name(
                        image_filename,
                        f"scatter_{safe_x_var}_vs_{safe_y_var}",
                        multi=True,
                    ),
                )
                plt.close(fig_individual)  # Clear memory
                pbar.update(1)  # Update progress bar

    # Save subplots
    if save_plots in ["all", "subplots"]:
        # Render the subplots
        fig_grid, axes = plt.subplots(n_rows, n_cols, figsize=subplot_figsize)
        # Normalize axes to a list
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for i, ax in enumerate(axes):
            if i < num_plots:
                x_var, y_var = combinations[i]
                sns.scatterplot(
                    x=x_var if not rotate_plot else y_var,
                    y=y_var if not rotate_plot else x_var,
                    data=df,
                    ax=ax,
                    color=scatter_color,
                    hue=hue,
                    palette=hue_palette,
                    size=size,
                    sizes=sizes,
                    marker=marker,
                    **kwargs,
                )

                if add_best_fit_line:
                    x_data = df[x_var] if not rotate_plot else df[y_var]
                    y_data = df[y_var] if not rotate_plot else df[x_var]
                    _add_best_fit(
                        ax=ax,
                        x=x_data,
                        y=y_data,
                        linestyle=best_fit_linestyle,
                        linecolor=best_fit_linecolor,
                        show_legend=show_legend,
                        legend_loc=legend_loc,
                    )

                r_value = df[x_var].corr(df[y_var])
                title = f"{_get_label(x_var, label_names)} vs. {_get_label(y_var, label_names)}"
                if show_correlation:
                    title += f" ($r$ = {r_value:.2f})"
                ax.set_title(
                    "\n".join(textwrap.wrap(title, width=text_wrap)),
                    fontsize=label_fontsize,
                )
                ax.tick_params(axis="x", rotation=xlabel_rot)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                _apply_thousands(ax)
            else:
                ax.axis("off")  # Turn off unused axes

        plt.tight_layout()

        # Save the subplots without a progress bar
        _save_figure(
            fig=fig_grid,
            image_path_png=image_path_png,
            image_path_svg=image_path_svg,
            image_filename=_resolve_save_name(image_filename, "scatter_plots_subplots"),
        )

        plt.close(fig_grid)  # Clear memory


################################################################################
######################### Correlation Matrices #################################
################################################################################


def flex_corr_matrix(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    annot: bool = True,
    cmap: Optional[str] = None,
    save_plots: bool = False,
    image_path_png: Optional[str] = None,
    image_path_svg: Optional[str] = None,
    image_filename: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    title: Optional[str] = None,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    xlabel_rot: int = 45,
    ylabel_rot: int = 0,
    xlabel_alignment: str = "right",
    ylabel_alignment: str = "center_baseline",
    text_wrap: int = 50,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_label: Optional[str] = None,
    triangular: bool = True,
    label_names: Optional[Dict[str, str]] = None,
    cbar_padding: float = 0.8,
    cbar_width_ratio: float = 0.05,
    show_colorbar: bool = True,
    corr_method: str = "pearson",
    show_significance: bool = False,
    significance_level: float = 0.05,
    significance_method: str = "stars",
    significance_legend_x: float = 0.5,
    p_adjust: Optional[str] = None,
    filter_significance: Optional[float] = None,
    corr_threshold: Optional[float] = None,
    return_corr: bool = False,
    return_sig: bool = False,
    show_n: bool = False,
    return_n: bool = False,
    n_format: str = "count",
    show_plot: bool = False,
    **kwargs: Dict[str, Any],
) -> Optional[Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]]:
    """
    Creates a correlation heatmap with extensive customization options, including
    triangular masking, alignment adjustments, title wrapping, dynamic colorbar
    scaling, and optional significance overlays. Users can save the plot in PNG
    and SVG formats.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.

    cols : list of str, optional
        List of column names to include in the correlation matrix. If None, all
        columns are included.

    annot : bool, optional (default=True)
        Whether to annotate the heatmap with correlation coefficients.

    cmap : str or None, optional (default=None)
        The colormap to use for the heatmap. When None, resolves by mode:
        "coolwarm" for the correlation heatmap (diverging, since correlations are
        signed and centered on zero) and "viridis" for the `show_n` heatmap
        (sequential, since sample sizes run from zero upward with no midpoint).
        Pass a name to override in either mode.

    save_plots : bool, optional (default=False)
        Whether to save the heatmap as an image. When True, the plot is saved
        using the title (or "Correlation Matrix" if no title is provided) as
        the filename. For explicit filename control, use `image_filename`
        instead.

    image_path_png : str, optional
        Directory path to save the heatmap as a PNG image.

    image_path_svg : str, optional
        Directory path to save the heatmap as an SVG image.

    image_filename : str, optional
        Base filename (without extension) for saving the figure via
        `_save_figure`. When provided, takes precedence over the
        `save_plots`-derived filename. Requires at least one of
        `image_path_png` or `image_path_svg` to be set.

    figsize : tuple, optional (default=(10, 10))
        Width and height of the heatmap figure.

    title : str, optional
        Title of the heatmap.

    label_fontsize : int, optional (default=12)
        Font size for axis labels and title.

    tick_fontsize : int, optional (default=10)
        Font size for tick labels and colorbar labels.

    xlabel_rot : int, optional (default=45)
        Rotation angle for x-axis labels.

    ylabel_rot : int, optional (default=0)
        Rotation angle for y-axis labels.

    xlabel_alignment : str, optional (default="right")
        Horizontal alignment for x-axis labels (e.g., "center", "right").

    ylabel_alignment : str, optional (default="center_baseline")
        Vertical alignment for y-axis labels (e.g., "center", "top").

    text_wrap : int, optional (default=50)
        Maximum width of the text before wrapping. This applies to the plot
        title, x-axis labels, and y-axis labels, ensuring that long text is
        neatly displayed without overflow or truncation.

    vmin : float or None, optional (default=None)
        Minimum value for the heatmap color scale. When None, resolves by mode: -1
        for the correlation heatmap, 0 for the `show_n` heatmap.

    vmax : float or None, optional (default=None)
        Maximum value for the heatmap color scale. When None, resolves by mode: 1
        for the correlation heatmap; for the `show_n` heatmap, 100 under
        `n_format="percent"` and the largest observed count otherwise.

    cbar_label : str or None, optional (default=None)
        Label for the colorbar. When None, resolves by mode: "Correlation Index"
        for the correlation heatmap, and a sample-size label derived from
        `n_format` for the `show_n` heatmap.

    triangular : bool, optional (default=True)
        Whether to show only the upper triangle of the correlation matrix.

    label_names : dict, optional
        A dictionary to map original column names to custom labels.

    cbar_padding : float, optional (default=0.8)
        Padding between the heatmap and the colorbar.

    cbar_width_ratio : float, optional (default=0.05)
        Relative width of the colorbar compared to the heatmap.

    show_colorbar : bool, optional (default=True)
        Whether to display the colorbar. If False, no colorbar will be shown.

    corr_method : str, optional (default="pearson")
        Method for computing the correlation matrix. Options are "pearson",
        "spearman", or "kendall". The corresponding scipy pairwise function
        is used when computing p-values for significance testing.

    show_significance : bool, optional (default=False)
        If True, overlays significance information on the heatmap cells based
        on pairwise p-values computed using `corr_method`.

    significance_level : float, optional (default=0.05)
        The p-value threshold below which a correlation is considered
        statistically significant. Gates both the star annotations ("stars"
        mode) and the cell blanking ("mask" mode); cells with
        p >= significance_level get no star or are blanked, respectively.

    significance_method : str, optional (default="stars")
        How to display significance on the heatmap. Options are:
        - "stars": appends significance stars to each cell annotation.
          * p < 0.05, ** p < 0.01, *** p < 0.001
        - "mask": blanks out cells where the correlation is not significant
          (p >= significance_level), leaving only significant cells annotated.

    significance_legend_x : float, optional (default=0.5)
        Horizontal position of the significance legend in figure coordinates
        when `show_significance=True` and `significance_method="stars"`.
        0.0 is the left edge, 1.0 is the right edge, and 0.5 centers the
        legend. The vertical position is computed automatically from the
        rendered axes bounding box so it always sits just below the x-axis
        labels regardless of figure size.

    p_adjust : str or None, optional (default=None)
        Multiple-comparison correction applied to the pairwise p-values before
        anything consumes them. Options are:
        - None: no correction. P-values are reported as computed.
        - "bonferroni": controls the family-wise error rate. Each p-value is
          multiplied by the number of tests and clipped at 1.0. Conservative.
        - "fdr_bh": Benjamini-Hochberg step-up procedure. Controls the false
          discovery rate. Less conservative than Bonferroni and usually the
          better choice on a wide matrix.
        A correlation matrix over k variables runs k * (k - 1) / 2 unique
        pairwise tests, so on even a modest matrix the uncorrected stars
        overstate significance. The correction is computed over unique pairs,
        not over every off-diagonal cell, so each pair is counted once. Pairs
        with too few complete observations to test are excluded from the test
        count. The correction runs before `filter_significance`, the stars/mask
        overlay, and `return_sig`, so all of them see adjusted values.

    filter_significance : float or None, optional (default=None)
        If provided, filters the correlation matrix to only include variables
        that have at least one statistically significant correlation with
        another variable at the specified p-value threshold. For example,
        ``filter_significance=0.05`` drops any variable with no pairwise
        p-value below 0.05. Valid values are ``None`` (no filtering), or a
        float such as ``0.05``, ``0.01``, or ``0.001``.
        When set, p-values are computed for filtering, but the significance
        overlay is drawn only if ``show_significance=True``.

    corr_threshold : float or None, optional (default=None)
        If provided, drops any variable whose strongest off-diagonal absolute
        correlation is below this value, so only features that correlate with
        at least one other feature at or above the threshold remain in the
        matrix. This filters *variables*, not individual cells: a surviving
        variable still displays its weaker correlations (including values
        below the threshold) against the other survivors. Must be in [0, 1]
        (e.g. 0.3). Raises if no variable qualifies.

    return_corr : bool, optional (default=False)
        If True, returns the correlation matrix (as plotted, after any
        `corr_threshold` / `filter_significance` filtering) as a DataFrame. By
        default this also suppresses the heatmap; pass `show_plot=True` to
        render it as well. If False, returns None and always plots.

    return_sig : bool, optional (default=False)
        If True, returns the pairwise p-value matrix (computed with
        `corr_method`, corrected by `p_adjust` if set, and subset to the same
        variables as the plotted matrix after any filtering) as a DataFrame.
        The diagonal holds the initialized placeholder value of 1. Like
        `return_corr`, this suppresses the heatmap by default; pass
        `show_plot=True` to render it as well.

    show_n : bool, optional (default=False)
        If True, the heatmap plots the pairwise sample sizes instead of the
        correlation coefficients. This is a mode switch, not an overlay: the cells
        are colored by n, labeled with n, and scaled on a count axis rather than the
        correlation axis. `cmap`, `vmin`, `vmax`, and `cbar_label` all pick up
        sample-size-appropriate defaults automatically in this mode, and can still
        be overridden. The result is a coverage map, showing at a glance which pairs
        rest on most of the data and which rest on a sliver of it, and revealing the
        structure of the missingness itself. Honors `n_format` and `triangular`.
        Mutually exclusive with `show_significance`, since there are no coefficients
        to attach stars to; draw the correlation figure separately.

    return_n : bool, optional (default=False)
        If True, returns the pairwise sample-size matrix as a DataFrame with the
        same shape, index, columns, and symmetry as the correlation matrix. Each
        off-diagonal cell holds the number of rows in which *both* variables in
        that pair are present, which is the number of observations that actually
        went into that cell's correlation coefficient. Because correlations are
        computed pairwise-complete (missing values are dropped per pair, not
        listwise across the whole frame), this count differs from cell to cell,
        and a strong coefficient backed by a handful of rows is indistinguishable
        from one backed by thousands when looking at the heatmap alone. The
        diagonal holds each variable's own non-null count. Like `return_corr`,
        this suppresses the heatmap by default; pass `show_plot=True` to render
        it as well.

    n_format : str, optional (default="count")
        Units for the sample sizes, applied to both the matrix returned by
        `return_n` and the heatmap drawn by `show_n`. Ignored when neither is set.
        Options are:
        - "count": raw row counts as integers.
        - "percent": each count divided by the total number of rows in the frame
          and multiplied by 100, returned as floats. Column filtering
          (`corr_threshold`, `filter_significance`) drops variables but never
          rows, so the denominator is always the full row count and the values
          stay comparable across filtered and unfiltered calls. Values are not
          rounded; round them yourself if you want a tidier display. Under this
          setting the diagonal reads as each variable's completeness, so a
          variable present in every row shows 100.0.

    show_plot : bool, optional (default=False)
        Only has an effect when `return_corr=True`, `return_sig=True`, or
        `return_n=True`. By default, requesting a returned frame suppresses the
        heatmap and returns the data only; set `show_plot=True` to also render
        the plot. When no return flag is set the heatmap always displays and
        this flag is ignored.

    **kwargs : dict, optional
        Additional keyword arguments to pass to `sns.heatmap()`.

    Returns:
    --------
    pandas.DataFrame, tuple of pandas.DataFrame, or None
        - If exactly one of `return_corr`, `return_sig`, or `return_n` is True:
          that single DataFrame.
        - If more than one is True: a tuple containing the requested frames in
          the fixed order `(corr_matrix, pval_matrix, n_matrix)`, with the
          unrequested frames omitted rather than padded with None. Requesting
          `return_corr` and `return_n` therefore yields a two-element tuple
          `(corr_matrix, n_matrix)`.
        - If none are set: None (the heatmap is plotted).

    Raises:
    -------
    ValueError
        If `annot`, `save_plots`, or `triangular` is not a boolean value.
    ValueError
        If `cols` is provided but is not a list of column names.
    ValueError
        If `save_plots` is True but neither `image_path_png` nor `image_path_svg`
        is specified.
    ValueError
        If `image_filename` is provided but neither `image_path_png` nor
        `image_path_svg` is specified.
    ValueError
        If `corr_method` is not one of "pearson", "spearman", or "kendall".
    ValueError
        If `significance_method` is not one of "stars" or "mask".
    ValueError
        If `p_adjust` is not None, "bonferroni", or "fdr_bh".
    ValueError
        If `n_format` is not one of "count" or "percent".
    ValueError
        If `show_n` and `show_significance` are both True.
    ValueError
        If `filter_significance` is not None and is not a positive float.
    ValueError
        If `corr_threshold` is not None and is not a number in [0, 1].

    Notes:
    ------
    - If `triangular=True`, the heatmap will display only the upper triangle
      of the correlation matrix, excluding the diagonal.
    - Custom labels specified in `label_names` will replace the default column
      names in the heatmap's axes.
    - Save formats are determined by the paths provided for PNG and SVG files.
    - The colorbar width dynamically scales based on the grid square size for
      consistent proportions.
    - The `cbar_padding` and `cbar_width_ratio` parameters control the relative
      spacing and size of the colorbar, providing additional layout customization.
    - The `show_colorbar` parameter allows users to optionally exclude the
      colorbar for a cleaner plot.
    - When `show_significance=True`, the diagonal always displays "1.00" to
      reflect the perfect self-correlation regardless of significance method.
    - `corr_threshold` removes whole variables, not cells. A kept variable
      shows all of its correlations against other kept variables, so some
      displayed values may be below the threshold.
    - The returned matrix is the full square matrix; the triangular display
      mask is not applied to it.
    - The p-value matrix returned by `return_sig=True` carries the initialized
      value of 1 on its diagonal, since a variable has no p-value against
      itself.
    - Correlations are computed pairwise-complete: for each pair of variables,
      rows missing either member of the pair are dropped, and the remaining rows
      are used. Rows missing some *third* variable are not dropped. Every cell in
      the matrix can therefore rest on a different number of observations. Set
      `return_n=True` to see those counts.
    - When `p_adjust` is combined with `corr_threshold`, the correction is
      computed only over the variables that survived the threshold, since
      p-values are not computed for variables that were already dropped. The test
      count reflects the tests that were actually run on the reduced matrix.
    - `p_adjust` does not change the star tiers themselves. The tiers remain
      p < 0.05, p < 0.01, p < 0.001 by convention; they are simply evaluated
      against corrected p-values, so fewer cells earn stars.
    - `show_n` and `show_significance` draw different quantities and cannot share a
      figure. The correlation heatmap answers how strong a relationship is; the
      `show_n` heatmap answers how much data stands behind it. Call the function
      twice to get both.
    """

    # Validation: Ensure annot is a boolean
    if not isinstance(annot, bool):
        raise ValueError(
            "Invalid value for `annot`. Please enter either True or False."
        )

    # Validation: Ensure cols is a list if provided
    if cols is not None and not isinstance(cols, list):
        raise ValueError("The `cols` parameter must be a list of column names.")

    # Validation: Ensure save_plots is a boolean
    if not isinstance(save_plots, bool):
        raise ValueError("Invalid `save_plots` value. Enter True or False.")

    # Validation: Ensure triangular is a boolean
    if not isinstance(triangular, bool):
        raise ValueError(
            "Invalid `triangular` value. Please enter either True or False."
        )

    # Validate corr_method
    valid_corr_methods = ["pearson", "spearman", "kendall"]
    if corr_method not in valid_corr_methods:
        raise ValueError(
            f"Invalid `corr_method` '{corr_method}'. "
            f"Choose from {valid_corr_methods}."
        )

    # Validate significance_method
    valid_sig_methods = ["stars", "mask"]
    if significance_method not in valid_sig_methods:
        raise ValueError(
            f"Invalid `significance_method` '{significance_method}'. "
            f"Choose from {valid_sig_methods}."
        )

    # show_n replaces the plotted quantity, so a significance overlay computed from
    # the correlations has nothing to annotate. Fail loudly instead of silently
    # dropping one of them.
    if show_n and show_significance:
        raise ValueError(
            "`show_n` and `show_significance` are mutually exclusive. `show_n` plots "
            "the pairwise sample sizes, not the correlations, so significance stars "
            "have nothing to attach to. Draw two separate figures."
        )

    # Validate n_format
    valid_n_formats = ["count", "percent"]
    if n_format not in valid_n_formats:
        raise ValueError(
            f"Invalid `n_format` '{n_format}'. Choose from {valid_n_formats}."
        )

    # Validate p_adjust
    valid_p_adjust = ["bonferroni", "fdr_bh"]
    if p_adjust is not None and p_adjust not in valid_p_adjust:
        raise ValueError(
            f"Invalid `p_adjust` '{p_adjust}'. "
            f"Choose from {valid_p_adjust} or None."
        )

    # Validate filter_significance
    if filter_significance is not None:
        if not isinstance(filter_significance, float) or filter_significance <= 0:
            raise ValueError(
                "`filter_significance` must be a positive float (e.g. 0.05, 0.01, "
                "0.001) or None."
            )

    # Validate corr_threshold
    if corr_threshold is not None and (
        not isinstance(corr_threshold, (int, float)) or not 0 <= corr_threshold <= 1
    ):
        raise ValueError(
            "`corr_threshold` must be a number in [0, 1] (e.g. 0.3) or None."
        )

    # Validate paths are specified if save_plots is True
    if save_plots and not (image_path_png or image_path_svg):
        raise ValueError(
            f"You must specify `image_path_png` or `image_path_svg` "
            f"when `save_plots` is True."
        )

    # Validate paths are specified if image_filename is provided
    if (
        image_filename is not None
        and not os.path.splitext(image_filename)[1]
        and not (image_path_png or image_path_svg)
    ):
        raise ValueError(
            "You must specify `image_path_png` or `image_path_svg` when "
            "`image_filename` has no file extension."
        )
    # Filter DataFrame if cols are specified
    if cols is not None:
        df = df[cols]

    # Select only numeric columns for correlation
    df_numeric = df.select_dtypes(include="number")

    # Calculate the correlation matrix
    corr_matrix = df_numeric.corr(method=corr_method)

    # Eliminate -0.00 display artifact by zeroing out near-zero values
    corr_matrix = corr_matrix.where(corr_matrix.abs() >= 0.005, 0.0)

    # Drop variables whose strongest off-diagonal |r| is below the threshold
    if corr_threshold is not None:
        _off = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool))
        keep = [c for c in corr_matrix.columns if _off[c].abs().max() >= corr_threshold]
        if not keep:
            raise ValueError(
                f"No variables have an absolute correlation >= {corr_threshold}. "
                "Lower `corr_threshold`."
            )
        df_numeric = df_numeric[keep]
        corr_matrix = df_numeric.corr(method=corr_method)
        corr_matrix = corr_matrix.where(corr_matrix.abs() >= 0.005, 0.0)

    # Compute p-values when any consumer needs them: the stars/mask display,
    # the variable filter, or the (optional) p-value return.
    _need_pvalues = show_significance or filter_significance is not None or return_sig

    # The pairwise sweep is shared. Both the p-values and the sample sizes fall
    # out of the same per-pair dropna, so walk the pairs once and fill whichever
    # matrices were actually asked for.
    _need_pairwise = _need_pvalues or return_n or show_n

    if _need_pairwise:
        from scipy import stats as _stats

        n_cols = len(df_numeric.columns)

        if _need_pvalues:
            pval_matrix = pd.DataFrame(
                np.ones((n_cols, n_cols)),
                index=df_numeric.columns,
                columns=df_numeric.columns,
            )

            _stat_func = {
                "pearson": _stats.pearsonr,
                "spearman": _stats.spearmanr,
                "kendall": _stats.kendalltau,
            }[corr_method]

        # The sample-size matrix is needed for the return value, the plot, or both.
        _need_n = return_n or show_n

        if _need_n:
            # Diagonal seeds with each variable's own non-null count, which is what
            # pairing a column with itself and dropping NaN would yield anyway.
            n_matrix = pd.DataFrame(
                np.diag([int(df_numeric[c].notna().sum()) for c in df_numeric.columns]),
                index=df_numeric.columns,
                columns=df_numeric.columns,
            )

        for i, col_i in enumerate(df_numeric.columns):
            for j, col_j in enumerate(df_numeric.columns):
                if i != j:
                    # Pairwise-complete: only rows where both members survive.
                    valid = df_numeric[[col_i, col_j]].dropna()
                    n_valid = len(valid)

                    if _need_n:
                        n_matrix.loc[col_i, col_j] = n_valid

                    if _need_pvalues:
                        if n_valid >= 3:
                            _, p = _stat_func(valid[col_i], valid[col_j])
                            pval_matrix.loc[col_i, col_j] = p
                        else:
                            pval_matrix.loc[col_i, col_j] = np.nan

        # Correct for multiple comparisons before anything reads the p-values, so
        # the stars, the mask, the variable filter, and the returned frame all
        # agree on what "significant" means.
        if _need_pvalues and p_adjust is not None:
            pval_matrix = _adjust_pvalues(pval_matrix, p_adjust)

        # Filter variables that have no significant correlations
        if filter_significance is not None:
            cols_to_keep = []
            for col_i in df_numeric.columns:
                for col_j in df_numeric.columns:
                    if col_i != col_j:
                        p = pval_matrix.loc[col_i, col_j]
                        if not pd.isna(p) and p < filter_significance:
                            cols_to_keep.append(col_i)
                            break
            df_numeric = df_numeric[cols_to_keep]
            corr_matrix = df_numeric.corr(method=corr_method)
            # Re-apply the -0.00 zeroing the recomputed matrix would otherwise lose.
            corr_matrix = corr_matrix.where(corr_matrix.abs() >= 0.005, 0.0)
            pval_matrix = pval_matrix.loc[cols_to_keep, cols_to_keep]
            if _need_n:
                n_matrix = n_matrix.loc[cols_to_keep, cols_to_keep]

        # Express the counts as a share of the frame. Filtering drops columns,
        # never rows, so the denominator is the full row count regardless of
        # what survived.
        if _need_n and n_format == "percent":
            n_matrix = 100.0 * n_matrix / len(df_numeric)

        def _stars(p):
            if pd.isna(p) or p >= significance_level:
                return ""
            elif p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            else:
                return "*"

    # show_n swaps the plotted quantity, and the two quantities want different scales.
    # Correlations are signed, live on [-1, 1], and have a meaningful midpoint at zero,
    # so they want a diverging colormap. Sample sizes run from zero upward with no
    # midpoint, so they want a sequential one. Rather than carry a parallel set of
    # n-specific parameters, `cmap`, `vmin`, `vmax`, and `cbar_label` default to None
    # and resolve to whichever is right for the active mode. Passing any of them
    # explicitly overrides the default in either mode.
    if show_n:
        plot_matrix = n_matrix
        plot_cmap = cmap if cmap is not None else "viridis"
        plot_vmin = vmin if vmin is not None else 0
        plot_vmax = (
            vmax
            if vmax is not None
            else (100.0 if n_format == "percent" else float(n_matrix.to_numpy().max()))
        )
        plot_cbar_label = cbar_label or (
            "% of rows behind each correlation"
            if n_format == "percent"
            else "Rows behind each correlation"
        )
    else:
        plot_matrix = corr_matrix
        plot_cmap = cmap if cmap is not None else "coolwarm"
        plot_vmin = vmin if vmin is not None else -1
        plot_vmax = vmax if vmax is not None else 1
        plot_cbar_label = cbar_label or "Correlation Index"

    # Generate a mask for the upper triangle, excluding the diagonal
    mask = None
    if triangular:
        mask = np.triu(np.ones_like(plot_matrix, dtype=bool), k=1)

    # Build annotation matrix
    if annot and show_n:
        annot_data = n_matrix.map(
            lambda v: f"{v:.1f}%" if n_format == "percent" else f"{int(v):,}"
        )
        annot_arg = annot_data
        fmt_arg = ""
    elif annot and show_significance:
        if significance_method == "stars":
            # Combine r value with significance stars
            annot_data = corr_matrix.copy().astype(object)
            for i, col_i in enumerate(corr_matrix.index):
                for j, col_j in enumerate(corr_matrix.columns):
                    if i == j:
                        annot_data.loc[col_i, col_j] = "1.00"
                    else:
                        r_val = corr_matrix.loc[col_i, col_j]
                        star = _stars(pval_matrix.loc[col_i, col_j])
                        annot_data.loc[col_i, col_j] = (
                            f"{r_val:.2f}{star}" if star else f"{r_val:.2f}"
                        )
            annot_arg = annot_data
            fmt_arg = ""
        elif significance_method == "mask":
            # Only show annotation for significant cells
            annot_data = corr_matrix.copy().astype(object)
            for i, col_i in enumerate(corr_matrix.index):
                for j, col_j in enumerate(corr_matrix.columns):
                    p = pval_matrix.loc[col_i, col_j]
                    if i == j:
                        annot_data.loc[col_i, col_j] = "1.00"
                    elif not pd.isna(p) and p >= significance_level:
                        annot_data.loc[col_i, col_j] = ""
                    else:
                        annot_data.loc[col_i, col_j] = (
                            f"{corr_matrix.loc[col_i, col_j]:.2f}"
                        )

        annot_arg = annot_data
        fmt_arg = ""
    else:
        annot_arg = annot
        fmt_arg = ".2f"

    # Plot unless the caller is pulling a dataframe and opted out.
    # When no return flag is set, plotting always happens regardless of show_plot.
    if not (return_corr or return_sig or return_n) or show_plot:

        # Set up the matplotlib figure
        fig, ax_heatmap = plt.subplots(figsize=figsize)

        # Draw the heatmap
        heatmap = sns.heatmap(
            plot_matrix,
            mask=mask,
            cmap=plot_cmap,
            annot=annot_arg,
            fmt=fmt_arg,
            square=True,
            linewidths=0.5,
            cbar=False,
            vmin=plot_vmin,
            vmax=plot_vmax,
            annot_kws={"fontsize": label_fontsize},
            ax=ax_heatmap,
            **kwargs,
        )

        # Add the colorbar
        if show_colorbar:
            divider = make_axes_locatable(ax_heatmap)
            cax = divider.append_axes(
                "right", size=f"{cbar_width_ratio*100}%", pad=cbar_padding
            )

            cbar = fig.colorbar(
                heatmap.collections[0],
                cax=cax,
                orientation="vertical",
            )

            cbar.ax.tick_params(labelsize=tick_fontsize)
            cbar.set_label(plot_cbar_label, fontsize=label_fontsize)

            pos_heatmap = ax_heatmap.get_position()
            pos_cax = cax.get_position()
            cax.set_position(
                [
                    pos_cax.x0,
                    pos_heatmap.y0,
                    pos_cax.width,
                    pos_heatmap.height,
                ]
            )

            for spine in cax.spines.values():
                spine.set_visible(False)

        # Set the title if provided
        if title:
            ax_heatmap.set_title(
                "\n".join(textwrap.wrap(title, width=text_wrap)),
                fontsize=label_fontsize,
            )

        # Apply custom labels if label_names is provided
        if label_names:
            heatmap.set_xticklabels(
                [
                    "\n".join(
                        textwrap.wrap(
                            label_names.get(label.get_text(), label.get_text()),
                            width=text_wrap,
                        )
                    )
                    for label in heatmap.get_xticklabels()
                ],
                rotation=xlabel_rot,
                fontsize=tick_fontsize,
                ha=xlabel_alignment,
                rotation_mode="anchor",
            )
            heatmap.set_yticklabels(
                [
                    "\n".join(
                        textwrap.wrap(
                            label_names.get(label.get_text(), label.get_text()),
                            width=text_wrap,
                        )
                    )
                    for label in heatmap.get_yticklabels()
                ],
                rotation=ylabel_rot,
                fontsize=tick_fontsize,
                va=ylabel_alignment,
            )
        else:
            heatmap.set_xticklabels(
                [
                    "\n".join(textwrap.wrap(label.get_text(), width=text_wrap))
                    for label in heatmap.get_xticklabels()
                ],
                rotation=xlabel_rot,
                ha=xlabel_alignment,
                fontsize=tick_fontsize,
                rotation_mode="anchor",
            )
            heatmap.set_yticklabels(
                [
                    "\n".join(textwrap.wrap(label.get_text(), width=text_wrap))
                    for label in heatmap.get_yticklabels()
                ],
                rotation=ylabel_rot,
                va=ylabel_alignment,
                fontsize=tick_fontsize,
            )

        # Adjust layout to prevent overlap
        plt.subplots_adjust(
            left=0.1,
            right=0.9,
            top=0.9,
            bottom=0.1,
            wspace=cbar_padding,
        )

        # Add significance legend note if stars mode is active
        if show_significance and significance_method == "stars":
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            tight_bbox = ax_heatmap.get_tightbbox(renderer)
            fig_height = fig.get_size_inches()[1] * fig.dpi
            legend_y = (tight_bbox.y0 / fig_height) - 0.05

            fig.text(
                significance_legend_x,
                legend_y,
                "* p < 0.05   ** p < 0.01   *** p < 0.001",
                ha="center",
                fontsize=tick_fontsize,
                style="italic",
                transform=fig.transFigure,
            )

        # Save via image_filename / _save_figure (preferred path)
        if image_filename is not None:
            _save_figure(
                fig=fig,
                image_path_png=image_path_png,
                image_path_svg=image_path_svg,
                filename=image_filename,
                image_filename=image_filename,
            )

        # Legacy save_plots path — derives filename from title
        elif save_plots:
            filename_title = title or "Correlation Matrix"
            safe_title = filename_title.replace(" ", "_").replace(":", "").lower()
            _save_figure(
                fig=fig,
                image_path_png=image_path_png,
                image_path_svg=image_path_svg,
                filename=safe_title,
                image_filename=image_filename,
            )

        plt.show()

    # Fixed order: (corr, pval, n). Unrequested frames are omitted, not padded,
    # so a single request unwraps to a bare DataFrame.
    requested = []
    if return_corr:
        requested.append(corr_matrix)
    if return_sig:
        requested.append(pval_matrix)
    if return_n:
        requested.append(n_matrix)

    if len(requested) == 1:
        return requested[0]
    if requested:
        return tuple(requested)


################################################################################
############################## Data Doctor #####################################
################################################################################


def data_doctor(
    df: pd.DataFrame,
    feature_name: str,
    data_fraction: float = 1.0,
    scale_conversion: Optional[str] = None,
    scale_conversion_kws: Optional[Dict] = None,
    apply_cutoff: bool = False,
    lower_cutoff: Optional[float] = None,
    upper_cutoff: Optional[float] = None,
    show_plot: bool = True,
    plot_type: str = "all",
    xlim: Optional[Tuple[float, float]] = None,
    kde_ylim: Optional[Tuple[float, float]] = None,
    hist_ylim: Optional[Tuple[float, float]] = None,
    box_violin_ylim: Optional[Tuple[float, float]] = None,
    image_path_png: Optional[str] = None,
    image_path_svg: Optional[str] = None,
    image_filename: Optional[str] = None,
    apply_as_new_col_to_df: bool = False,
    kde_kws: Optional[Dict] = None,
    hist_kws: Optional[Dict] = None,
    box_violin_kws: Optional[Dict] = None,
    box_violin: str = "boxplot",
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    random_state: Optional[int] = None,
    figsize: Tuple[int, int] = (18, 6),
) -> pd.DataFrame:
    """
    Analyze and transform a specific feature in a DataFrame, with options for
    scaling, applying cutoffs, and visualizing the results. This function also
    allows for the creation of a new column with the transformed data if
    specified.

    Plots are displayed by default. If `image_filename` is provided, the figure
    is saved: with an extension it is written verbatim to that path (no
    directory needed); without one it is combined with `image_path_png` and/or
    `image_path_svg`.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the feature to analyze.

    feature_name : str
        The name of the feature (column) to analyze.

    data_fraction : float, optional (default=1.0)
        Fraction of the data to analyze. Useful for large datasets where a
        representative sample is sufficient. If
        `apply_as_new_col_to_df=True`, the full dataset is always used.

    scale_conversion : str, optional
        Transformation applied to the feature. Supported options include:
            - "abs": absolute value
            - "log": natural logarithm
            - "sqrt": square root
            - "cbrt": cube root
            - "reciprocal": reciprocal transformation
            - "stdrz": z-score standardization
            - "minmax": min–max scaling
            - "robust": robust scaling using median and IQR
            - "maxabs": max-absolute scaling
            - "exp": exponential transformation
            - "logit": logit transform (values strictly between 0 and 1)
            - "arcsinh": inverse hyperbolic sine
            - "square": square of values
            - "boxcox": Box–Cox transform (positive values only)
            - "power": Yeo–Johnson power transform
        If None, no transformation is applied.

    scale_conversion_kws : dict, optional
        Additional keyword arguments passed to the transformation method.
        Examples include:
            - "alpha" or "lmbda" for Box–Cox
            - "quantile_range" for robust scaling

    apply_cutoff : bool, optional (default=False)
        Whether to apply lower and/or upper cutoffs to the feature.

    lower_cutoff : float, optional
        Lower bound applied when `apply_cutoff=True`.

    upper_cutoff : float, optional
        Upper bound applied when `apply_cutoff=True`.

    show_plot : bool, optional (default=True)
        Whether to display plots for the transformed feature.

    plot_type : str or list of str, optional (default="all")
        Plot types to generate. Supported values are:
            - "all": KDE, histogram, and box/violin plot
            - "kde": kernel density estimate
            - "hist": histogram
            - "ecdf": empirical CDF
            - "box_violin": boxplot or violin plot
        When multiple plot types are provided, they are arranged in a single
        figure row.

    figsize : tuple of int, optional (default=(18, 6))
        Figure size in inches.

    xlim : tuple of (float, float), optional
        Limits for the x-axis applied across plots.

    kde_ylim : tuple of (float, float), optional
        Limits for the KDE y-axis.

    hist_ylim : tuple of (float, float), optional
        Limits for the histogram y-axis.

    box_violin_ylim : tuple of (float, float), optional
        Limits for the boxplot or violin plot y-axis.

    image_path_png : str, optional
        Directory path used to save a PNG version of the figure.

    image_path_svg : str, optional
        Directory path used to save an SVG version of the figure.

    image_filename : str, optional
        Filename used when saving the figure. May be a full path with extension
        (saved verbatim, no directory required) or a bare stem combined with
        `image_path_png` / `image_path_svg`. The feature name, transformation,
        plot types, and cutoff flag are encoded into the auto-generated name
        when a stem (or no name) is used. If None, the figure is not saved.

    apply_as_new_col_to_df : bool, optional (default=False)
        Whether to add the transformed feature back to the DataFrame as a new
        column. The column name is derived from the feature name and applied
        transformation.

    kde_kws : dict, optional
        Keyword arguments passed to seaborn.kdeplot.

    hist_kws : dict, optional
        Keyword arguments passed to seaborn.histplot.

    box_violin_kws : dict, optional
        Keyword arguments passed to seaborn.boxplot or seaborn.violinplot.

    box_violin : {"boxplot", "violinplot"}, optional (default="boxplot")
        Specifies the plot style used when `plot_type="box_violin"`.

    label_fontsize : int, optional (default=12)
        Font size used for axis labels and titles.

    tick_fontsize : int, optional (default=10)
        Font size used for axis tick labels.

    random_state : int, optional
        Random seed used when sampling the data.

    Returns
    -------
    None
        The function prints a summary report and generates plots. If
        `apply_as_new_col_to_df=True`, the DataFrame is modified in place.

    Raises
    ------
    ValueError
        If an invalid `scale_conversion` is specified.

    ValueError
        If Box–Cox transformation is applied to non-positive values.

    ValueError
        If an invalid `plot_type` or `box_violin` option is provided.

    ValueError
        If `image_filename` is provided without an extension and neither
        `image_path_png` nor `image_path_svg` is specified.

    ValueError
        If the length of the transformed data does not match the original
        feature length.

    Notes
    -----
    - Saving is filename-driven: plots are written only when
    `image_filename` is provided.
    - A full-path `image_filename` (with extension) is saved verbatim with no
    directory required; a bare stem is combined with the provided directories.
    - Output filenames encode the feature name, transformation, plot types,
    and whether cutoffs were applied.
    - Cutoff values, when used, are displayed beneath the plots for clarity.
    - The histogram count (y) axis uses thousands separators. The x-axis keeps
    scientific notation for values at or above 100,000.
    """

    # Suppress warnings for division by zero, or invalid values in subtract
    np.seterr(divide="ignore", invalid="ignore")

    # If the user specifies apply_as_new_col_to_df, check for valid conditions
    if apply_as_new_col_to_df:
        if scale_conversion is None and not apply_cutoff:
            raise ValueError(
                "When applying a new column with `apply_as_new_col_to_df=True`, "
                "you must specify either a `scale_conversion` or set "
                "`apply_cutoff=True`."
            )

    # If conversion will be applied to a new column, set sample_frac to 1
    if apply_as_new_col_to_df:
        data_fraction = 1  # change the sample fraction value to 100 percent, to

    # Define valid scale conversions
    valid_conversions = [
        "logit",
        "abs",
        "log",
        "sqrt",
        "cbrt",
        "reciprocal",
        "stdrz",
        "minmax",
        "robust",
        "maxabs",
        "exp",
        "arcsinh",
        "square",
        "boxcox",
        "power",
        None,
    ]

    # Check if scale_conversion is valid
    if scale_conversion not in valid_conversions:
        raise ValueError(
            f"Invalid scale_conversion '{scale_conversion}'. "
            f"Valid options are: {valid_conversions[:-1]}"
        )

    new_col_name = feature_name

    # Sample the data once to ensure consistency in transformations
    # Convert data according to scale_conversion selection

    if apply_as_new_col_to_df:
        if data_fraction != 1:
            print(
                f"[info] apply_as_new_col_to_df=True → overriding "
                f"data_fraction={data_fraction} to 1.0"
            )
        data_fraction = 1.0
        sampled_feature = df[feature_name]
    else:
        sampled_feature = df.sample(
            frac=data_fraction,
            random_state=random_state,
        )[feature_name]

    # New column name options when apply_as_new_col_to_df
    if apply_as_new_col_to_df:
        if scale_conversion is None and apply_cutoff:
            new_col_name = feature_name + "_w_cutoff"
        elif scale_conversion is not None:
            new_col_name = feature_name + "_" + scale_conversion

    # Initialize scale_conversion_kws if None
    scale_conversion_kws = scale_conversion_kws or {}

    # Transformation logic
    if scale_conversion == "logit":
        if np.any((sampled_feature <= 0) | (sampled_feature >= 1)):
            raise ValueError(
                "Logit transformation requires values to be between 0 and 1. "
                "Consider using a scaling method such as min-max scaling first."
            )
        from scipy.special import logit

        feature_ = logit(sampled_feature, **scale_conversion_kws)

    elif scale_conversion == "abs":
        feature_ = np.abs(sampled_feature, **scale_conversion_kws)

    elif scale_conversion == "log":
        feature_ = np.log(sampled_feature, **scale_conversion_kws)

    elif scale_conversion == "sqrt":
        feature_ = np.sqrt(sampled_feature, **scale_conversion_kws)

    elif scale_conversion == "cbrt":
        feature_ = np.cbrt(sampled_feature, **scale_conversion_kws)

    elif scale_conversion == "reciprocal":
        feature_ = np.divide(1, sampled_feature, **scale_conversion_kws)

    elif scale_conversion == "stdrz":
        mean = np.mean(sampled_feature)
        std = np.std(sampled_feature)
        feature_ = np.divide(
            sampled_feature - mean,
            std,
            **scale_conversion_kws,
        )

    elif scale_conversion == "minmax":
        min_val = np.min(sampled_feature)
        max_val = np.max(sampled_feature)
        feature_ = np.divide(
            sampled_feature - min_val,
            max_val - min_val,
            **scale_conversion_kws,
        )

    elif scale_conversion == "robust":
        # Extract optional keyword arguments for RobustScaler
        # Default to True (center by median)
        with_centering = scale_conversion_kws.get("with_centering", True)
        # Default to IQR
        quantile_range = scale_conversion_kws.get(
            "quantile_range",
            (25.0, 75.0),
        )

        # Apply RobustScaler with optional kwargs
        scaler = RobustScaler(
            with_centering=with_centering,
            quantile_range=quantile_range,
        )
        feature_ = scaler.fit_transform(sampled_feature.values.reshape(-1, 1)).flatten()
        feature_ = pd.Series(feature_, index=sampled_feature.index)

    elif scale_conversion == "maxabs":
        max_abs = np.max(np.abs(sampled_feature))
        # This directly divides by max abs value
        feature_ = sampled_feature / max_abs

    elif scale_conversion == "exp":
        feature_ = np.exp(
            sampled_feature,
            **scale_conversion_kws,
        )

    elif scale_conversion == "arcsinh":
        feature_ = np.arcsinh(
            sampled_feature,
            **scale_conversion_kws,
        )

    elif scale_conversion == "square":
        feature_ = np.square(
            sampled_feature,
            **scale_conversion_kws,
        )

    elif scale_conversion == "boxcox":
        if np.any(sampled_feature <= 0):
            raise ValueError(
                "Box-Cox transformation requires strictly positive values."
            )

        # Initialize scale_conversion_kws if None
        scale_conversion_kws = scale_conversion_kws or {}

        # Apply Box-Cox transformation
        try:
            if "alpha" in scale_conversion_kws:
                # Apply Box-Cox transformation and get confidence interval
                feature_array, box_cox_lmbda, conf_interval = stats.boxcox(
                    sampled_feature, **scale_conversion_kws
                )
            else:
                # Apply Box-Cox transformation and get single lambda
                feature_array, box_cox_lmbda = stats.boxcox(
                    sampled_feature, **scale_conversion_kws
                )

        except ValueError as ve:
            raise ValueError(f"Error during Box-Cox transformation: {ve}")

        # Ensure feature_array is a 1D array
        feature_array = np.asarray(feature_array).flatten()

        # Check if the length of feature_array matches the sampled feature
        if len(feature_array) != len(sampled_feature):
            raise ValueError(
                f"Length of transformed data ({len(feature_array)}) does not match "
                f"the length of the sampled feature ({len(sampled_feature)})"
            )

        # Convert the feature_array into a pandas Series with the correct index
        feature_ = pd.Series(feature_array, index=sampled_feature.index)

    elif scale_conversion == "power":
        pt = PowerTransformer(**scale_conversion_kws)
        feature_array = pt.fit_transform(
            sampled_feature.values.reshape(-1, 1)
        ).flatten()
        # feature_ = pd.Series(feature_array)  # Do not specify index
        feature_ = pd.Series(feature_array, index=sampled_feature.index)

    else:
        feature_ = sampled_feature.copy()
        if scale_conversion is None:
            scale_conversion = "None"

    # Apply cutoffs if specified
    if apply_cutoff:
        feature_ = feature_.clip(lower=lower_cutoff, upper=upper_cutoff)

    # Ensure feature_ is a pandas Series
    if not isinstance(feature_, pd.Series):
        feature_ = pd.Series(feature_)

    print("DATA DOCTOR SUMMARY REPORT".center(52))

    # ASCII table for statistical data
    print(f"+{'-'*30}+{'-'*20}+")
    print(f"| {'Feature':<28} | {feature_name:<18} |")
    print(f"+{'-'*30}+{'-'*20}+")

    # Header for the statistical section
    print(f"| {'Statistic':<28} | {'Value':<18} |")
    print(f"+{'-'*30}+{'-'*20}+")
    print(f"| {'Min':<28} | {np.min(feature_):>18,.4f} |")
    print(f"| {'Max':<28} | {np.max(feature_):>18,.4f} |")
    print(f"| {'Mean':<28} | {np.mean(feature_):>18,.4f} |")
    print(f"| {'Median':<28} | {np.median(feature_):>18,.4f} |")
    print(f"| {'Std Dev':<28} | {np.std(feature_):>18,.4f} |")

    # Header for the quartiles section
    print(f"+{'-'*30}+{'-'*20}+")
    print(f"| {'Quartile':<28} | {'Value':<18} |")
    print(f"+{'-'*30}+{'-'*20}+")
    print(f"| {'Q1 (25%)':<28} | {np.quantile(feature_, 0.25):>18,.4f} |")
    print(f"| {'Q2 (50% = Median)':<28} | {np.quantile(feature_, 0.50):>18,.4f} |")
    print(f"| {'Q3 (75%)':<28} | {np.quantile(feature_, 0.75):>18,.4f} |")
    print(
        f"| {'IQR':<28} | "
        f"{(np.quantile(feature_, 0.75) - np.quantile(feature_, 0.25)):>18,.4f} |"
    )

    # Calculate the first quartile (25th percentile) of the feature's values.
    Q1 = np.quantile(feature_, 0.25)

    # Calculate the third quartile (75th percentile) of the feature's values.
    Q3 = np.quantile(feature_, 0.75)

    # Compute the Interquartile Range (IQR) by subtracting Q1 from Q3.
    # The IQR represents the middle 50% of the data, providing a measure of
    # variability.
    IQR = Q3 - Q1

    # Calculate the lower bound for outliers using the IQR method.
    # This boundary is set at 1.5 * IQR below the first quartile (Q1).
    # Values below this bound are considered potential outliers.
    outlier_lower_bound = Q1 - 1.5 * IQR

    # Calculate the upper bound for outliers using the IQR method.
    # This boundary is set at 1.5 * IQR above the third quartile (Q3).
    # Values above this bound are considered potential outliers.
    outlier_upper_bound = Q3 + 1.5 * IQR

    # Header for the outlier section
    print(f"+{'-'*30}+{'-'*20}+")
    print(f"| {'Outlier Bound':<28} | {'Value':<18} |")
    print(f"+{'-'*30}+{'-'*20}+")
    print(f"| {'Lower Bound':<28} | {outlier_lower_bound:>18,.4f} |")
    print(f"| {'Upper Bound':<28} | {outlier_upper_bound:>18,.4f} |")
    print(f"+{'-'*30}+{'-'*20}+")

    # Add column to dataframe along with data
    if apply_as_new_col_to_df and data_fraction == 1:
        # Ensure feature_ has the same length as df
        if len(feature_) != len(df):
            raise ValueError(
                "Length of transformed feature does not match length of DataFrame."
            )

        # Assign values directly
        df[new_col_name] = feature_.reindex(df.index)
        print()
        print("New Column Name:", new_col_name)
        # Print lambda if 'lmbda' was specified
        if "lmbda" in scale_conversion_kws:
            print(f"Box-Cox Lambda (provided): {scale_conversion_kws['lmbda']:.4f}")

        # Print the lambda or confidence interval based on the transformation
        elif scale_conversion == "boxcox":
            if "alpha" in scale_conversion_kws:
                # Confidence interval is available, so print it
                if "conf_interval" in locals():
                    print(
                        f"Box-Cox C.I. for Lambda: ({conf_interval[0]:.4f}, "
                        f"{conf_interval[1]:.4f})"
                    )
            else:
                # No alpha, so we print the single lambda value
                if "box_cox_lmbda" in locals():
                    print(f"Box-Cox Lambda: {round(box_cox_lmbda, 4)}")

    elif apply_as_new_col_to_df and data_fraction != 1:
        print("New Column Name:", new_col_name)
        print(
            "NOTE: Column was not added to dataframe as sample_frac is set to "
            + str(data_fraction)
            + " and not to 1, representing 100 percent."
        )

    # Update lower_cutoff and upper_cutoff values to represent any value updates
    # made in steps above...to ensure the xlabel reflects these values

    lower_cutoff = round(np.min(feature_), 4)
    upper_cutoff = round(np.max(feature_), 4)

    # Convert plot_type to a list if it’s a single string
    if isinstance(plot_type, str):
        if plot_type == "all":
            plot_type = ["kde", "hist", "box_violin"]
        else:
            plot_type = [plot_type]
    elif not isinstance(plot_type, (list, tuple)):
        raise ValueError("plot_type must be a string, list, or tuple.")

    # Verify that all plot types are valid
    valid_plot_types = ["kde", "hist", "ecdf", "box_violin"]
    invalid_plots = [ptype for ptype in plot_type if ptype not in valid_plot_types]
    if invalid_plots:
        raise ValueError(
            f"Invalid plot type(s) {invalid_plots}. "
            f"Valid options are: {valid_plot_types}"
        )

    # Determine layout based on the number of specified plot types
    n_plots = len(plot_type)
    n_rows = 1 if n_plots <= 3 else 2
    n_cols = n_plots if n_plots <= 3 else (n_plots + 1) // 2

    # Increase figure size slightly to create more space
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(figsize[0] + 2, figsize[1] + 2), squeeze=False
    )

    # Flatten axes for easy indexing if there are multiple rows/columns
    axes = axes.flatten()

    # Plot based on specified plot types
    if show_plot:
        # Build a finite, plot-ready vector AFTER transforms + optional cutoff
        feature_plot = (
            pd.to_numeric(feature_, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        x_plot = feature_plot.to_numpy(dtype=float)

        # Use user cutoffs if provided, otherwise derive clip from finite data
        if apply_cutoff and (lower_cutoff is not None or upper_cutoff is not None):
            clip_low = (
                float(lower_cutoff)
                if lower_cutoff is not None
                else float(np.min(x_plot)) if x_plot.size else None
            )
            clip_high = (
                float(upper_cutoff)
                if upper_cutoff is not None
                else float(np.max(x_plot)) if x_plot.size else None
            )
        else:
            clip_low = float(np.min(x_plot)) if x_plot.size else None
            clip_high = float(np.max(x_plot)) if x_plot.size else None

        clip = (
            (clip_low, clip_high)
            if (clip_low is not None and clip_high is not None)
            else None
        )

        for i, ptype in enumerate(plot_type):
            ax = axes[i]

            # Check if x-axis values are large (in the hundreds of thousands or above)
            if feature_.max() >= 100_000:
                ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(
                    style="sci", axis="x", scilimits=(5, 5)
                )  # Set only for 100,000+

            if ptype == "kde":
                if x_plot.size == 0:
                    ax.text(
                        0.5,
                        0.5,
                        f"No valid values for {feature_name}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(
                        f"KDE Plot: {feature_name} (Scale: {scale_conversion})",
                        fontsize=label_fontsize,
                        pad=25,
                    )
                else:
                    kws = dict(kde_kws or {})
                    kws.setdefault("cut", 0)
                    kws.setdefault("bw_adjust", 1.2)

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="overflow encountered in vecdot",
                            category=RuntimeWarning,
                        )
                        sns.kdeplot(
                            x=x_plot,
                            ax=ax,
                            clip=clip,
                            warn_singular=False,
                            **kws,
                        )

                    ax.set_title(
                        f"KDE Plot: {feature_name} (Scale: {scale_conversion})",
                        fontsize=label_fontsize,
                        pad=25,
                    )

                if xlim:
                    ax.set_xlim(xlim)

                if kde_ylim:
                    ax.set_ylim(kde_ylim)

            elif ptype == "hist":
                sns.histplot(x=feature_, ax=ax, **(hist_kws or {}))
                ax.set_title(
                    f"Histplot: {feature_name} (Scale: {scale_conversion})",
                    fontsize=label_fontsize,
                    pad=25,  # Increased padding
                )
                ax.set_xlabel(f"{feature_name}", fontsize=label_fontsize)
                ax.set_ylabel("Count", fontsize=label_fontsize)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda v, _: f"{v:,.0f}")
                )
                if xlim:
                    ax.set_xlim(xlim)
                if hist_ylim:
                    ax.set_ylim(hist_ylim)

            elif ptype == "ecdf":
                # Convert to 1D numpy array and drop NaNs
                x_raw = np.asarray(feature_)
                x_raw = x_raw[~np.isnan(x_raw)]

                # Handle empty after dropping NaNs
                if x_raw.size == 0:
                    ax.text(
                        0.5,
                        0.5,
                        f"No valid values for {feature_name}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(
                        f"ECDF: {feature_name} (Scale: {scale_conversion})",
                        fontsize=label_fontsize,
                        pad=25,
                    )
                else:
                    # ECDF
                    x = np.sort(x_raw)
                    n = x.size
                    y = np.arange(1, n + 1) / n

                    # Optional normal CDF overlay (SciPy-free; Python 3.7.4+)
                    mu = float(np.mean(x_raw))
                    sigma = float(np.std(x_raw, ddof=1)) if n > 1 else 0.0

                    x_smooth = np.linspace(x.min(), x.max(), 300)
                    if sigma > 0:
                        # Normal CDF via error function:
                        # Phi(x) = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
                        z = (x_smooth - mu) / (sigma * math.sqrt(2.0))
                        y_smooth = 0.5 * (1.0 + np.vectorize(math.erf)(z))
                    else:
                        # If sigma is 0 (all values equal), the normal CDF isn't meaningful
                        y_smooth = None

                    # Plot ECDF (steps + dots)
                    ax.step(x, y, where="post", color="black", linewidth=0.55)
                    ax.plot(x, y, "o", color="black", markersize=2, alpha=0.6)

                    # Plot normal overlay if possible
                    if y_smooth is not None:
                        ax.plot(x_smooth, y_smooth, linewidth=1.45)

                    # Dashed 0 and 1 lines like R
                    ax.axhline(0, color="grey", linestyle="--", linewidth=0.7)
                    ax.axhline(1, color="grey", linestyle="--", linewidth=0.7)

                    # Labels and styling
                    ax.set_title(
                        f"ECDF: {feature_name} (Scale: {scale_conversion})",
                        fontsize=label_fontsize,
                        pad=25,
                    )
                    ax.set_xlabel(f"{feature_name}", fontsize=label_fontsize)
                    ax.set_ylabel("F(x)", fontsize=label_fontsize)
                    ax.tick_params(axis="both", labelsize=tick_fontsize)

                    ax.set_ylim(-0.02, 1.02)
                    ax.grid(alpha=0.2)

                if xlim:
                    ax.set_xlim(xlim)

            elif ptype == "box_violin":
                # feature_ is already a pd.Series by this point in the function
                x = feature_.copy()

                # Drop non-finite values without coercing the whole series
                x = x.replace([np.inf, -np.inf], np.nan).dropna()

                if x.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"No valid values for {feature_name}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    kind_label = "Boxplot" if box_violin == "boxplot" else "Violinplot"
                    ax.set_title(
                        f"{kind_label}: {feature_name} " f"(Scale: {scale_conversion})",
                        fontsize=label_fontsize,
                        pad=25,
                    )
                else:
                    if box_violin == "boxplot":
                        sns.boxplot(
                            x=x,  # IMPORTANT: use x, not feature_
                            ax=ax,
                            **(box_violin_kws or {}),
                        )
                        ax.set_title(
                            f"Boxplot: {feature_name} (Scale: {scale_conversion})",
                            fontsize=label_fontsize,
                            pad=25,
                        )

                    elif box_violin == "violinplot":
                        bv_kws = dict(box_violin_kws or {})
                        bv_kws.setdefault("cut", 0)
                        bv_kws.setdefault("bw_adjust", 1.2)

                        sns.violinplot(
                            x=x,  # IMPORTANT: use x, not feature_
                            ax=ax,
                            **bv_kws,
                        )
                        ax.set_title(
                            f"Violinplot: {feature_name} (Scale: {scale_conversion})",
                            fontsize=label_fontsize,
                            pad=25,
                        )

                ax.set_xlabel(f"{feature_name}", fontsize=label_fontsize)
                ax.set_ylabel("", fontsize=label_fontsize)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                if xlim:
                    ax.set_xlim(xlim)
                if box_violin_ylim:
                    ax.set_ylim(box_violin_ylim)

    # Display the cutoff text universally
    if plot_type == ["kde", "hist", "box_violin"]:  # When using "all"
        # configuration. Check if we have a 2D grid of axes or a single row
        if n_rows > 1:
            # Use dedicated row for text when in subplot mode with multiple rows
            axes[1, 0].text(
                0,
                0.5,
                f"Lower cutoff: {round(lower_cutoff, 2)} |"
                f"Upper cutoff: {round(upper_cutoff, 2)}",
                fontsize=label_fontsize,
                ha="left",
                va="center",
            )
            axes[1, 0].axis("off")
            axes[1, 1].axis("off")
            axes[1, 2].axis("off")
        else:
            # If only a single row of plots, add text below the plots in that row
            fig.text(
                0.5,
                -0.05,  # Position below the plot
                f"Lower cutoff: {lower_cutoff:,.0f} | "
                f"Upper cutoff: {upper_cutoff:,.0f}",
                ha="center",
                fontsize=label_fontsize,
            )
    else:
        # Display text below the single plot row when using custom configuration
        fig.text(
            0.5,
            -0.05,  # Position below the plot
            f"Lower cutoff: {lower_cutoff:,.0f} | "
            f"Upper cutoff: {upper_cutoff:,.0f}",
            ha="center",
            fontsize=label_fontsize,
        )
    # Apply layout adj. to prevent overlaps without affecting inter-plot spacing
    plt.tight_layout(pad=3.0, w_pad=1.5, h_pad=3.0)

    # Check if the user-specified plot type is valid
    if box_violin not in ["boxplot", "violinplot"]:
        raise ValueError(
            f"Invalid plot type '{box_violin}'. "
            f"Valid options are 'boxplot' or 'violinplot'."
        )

    if (
        image_filename
        and not os.path.splitext(image_filename)[1]
        and not (image_path_png or image_path_svg)
    ):
        raise ValueError(
            "You must provide either 'image_path_png' or 'image_path_svg' "
            "when 'image_filename' has no file extension."
        )

    # ------------------------------------------------------------------
    # Save figure (optional)
    # ------------------------------------------------------------------
    if image_path_png or image_path_svg or image_filename:
        # Adjust plot_type for custom labeling of boxplot or violinplot
        adjusted_plot_type = [
            box_violin if ptype == "box_violin" else ptype for ptype in plot_type
        ]
        plot_type_str = "_".join(adjusted_plot_type)
        cutoff_str = "_cutoff" if apply_cutoff else ""
        auto_name = (
            f"{feature_name}_"
            f"{scale_conversion if scale_conversion else 'original'}_"
            f"{plot_type_str}{cutoff_str}"
        )
        _save_figure(
            fig=fig,
            image_path_png=image_path_png,
            image_path_svg=image_path_svg,
            image_filename=_resolve_save_name(image_filename, auto_name),
        )


#################################################################################
######################### Crosstab Plotting Function ############################
#################################################################################


def outcome_crosstab_plot(
    df: pd.DataFrame,
    list_name: List[str],
    outcome: str,
    bbox_to_anchor: Tuple[float, float] = (0.5, -0.25),
    w_pad: float = 4,
    h_pad: float = 4,
    figsize: Tuple[int, int] = (12, 8),
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    label_0: Optional[str] = None,
    label_1: Optional[str] = None,
    normalize: bool = False,
    show_value_counts: bool = False,
    color_schema: Optional[List[str]] = None,
    save_plots: bool = False,
    image_path_png: Optional[str] = None,
    image_path_svg: Optional[str] = None,
    image_filename: Optional[str] = None,
    string: Optional[str] = None,
) -> None:
    """
    Generates crosstab bar plots visualizing the relationship between an outcome
    variable and multiple categorical variables.

    Color control (via color_schema):
      - None: default two-tone plus extras.
      - list/tuple: applies to all subplots, repeating if needed.
      - dict: map column names to specific color lists.

    Parameters:
    - df: pandas.DataFrame to plot.
    - list_name: list of str, categorical columns to include.
    - outcome: str, name of the outcome column.
    - bbox_to_anchor: tuple, legend anchor coordinates.
    - w_pad, h_pad: floats, padding for tight_layout.
    - figsize: tuple, figure size.
    - label_fontsize: int, font size for axis labels, titles, legend.
    - tick_fontsize: int, font size for tick labels.
    - n_rows, n_cols: ints, optional grid rows/columns; auto-calculated if omitted.
    - label_0, label_1: str, custom x-axis labels for outcome levels.
    - normalize: bool, False for raw counts, True for normalized proportions.
    - show_value_counts: bool, append counts or percentages to legend entries.
    - color_schema: None, list, or dict for color customization.
    - save_plots: bool, save via the legacy `string`/directory scheme if True.
      Ignored when `image_filename` is given.
    - image_path_png, image_path_svg: str, directory paths for saving. Required
      with `save_plots`, or with a bare-stem `image_filename`.
    - image_filename: str, optional destination. With an extension it is saved
      verbatim (no directory needed); without one it is combined with
      `image_path_png` / `image_path_svg`. Takes precedence over `save_plots`.
    - string: base filename (sanitized: spaces to underscores, colons removed,
      lowercased) for the legacy `save_plots` path. Defaults to
      "crosstab_plot".

    Raises:
        - ValueError: if `image_filename` is provided without an extension and
        neither `image_path_png` nor `image_path_svg` is specified.

    Notes:
        - The count (y) axis and legend counts use thousands separators; the
        normalized axis and legend use whole-number percentages.
    """
    n_vars = len(list_name)

    if (
        image_filename is not None
        and not os.path.splitext(image_filename)[1]
        and not (image_path_png or image_path_svg)
    ):
        raise ValueError(
            "You must specify `image_path_png` or `image_path_svg` when "
            "`image_filename` has no file extension."
        )

    # Determine grid size
    if n_rows is None and n_cols is None:
        n_cols = math.ceil(math.sqrt(n_vars))
        n_rows = math.ceil(n_vars / n_cols)
    elif n_rows is None:
        n_rows = math.ceil(n_vars / n_cols)
    elif n_cols is None:
        n_cols = math.ceil(n_vars / n_rows)

    # Default color schemes
    default_colors = ["#00BFC4", "#F8766D"]
    extra_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for item, ax in zip(list_name, axes[:n_vars]):
        # Compute crosstab
        if not normalize:
            ylabel = "Count"
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ctab = pd.crosstab(df[outcome], df[item])
        else:
            ylabel = "Percentage"
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
            )
            ctab = pd.crosstab(df[outcome], df[item], normalize="index")

        # Select colors
        n_cats = len(ctab.columns)
        if isinstance(color_schema, dict) and item in color_schema:
            scheme = color_schema[item]
        elif isinstance(color_schema, (list, tuple)):
            scheme = list(color_schema)
        else:
            scheme = default_colors.copy()
        if len(scheme) < n_cats:
            needed = n_cats - len(scheme)
            pool = extra_colors if color_schema is None else scheme
            scheme = (scheme + pool * math.ceil(needed / len(pool)))[:n_cats]

        # Plot
        ctab.plot(kind="bar", stacked=True, rot=0, ax=ax, color=scheme)

        # Formatting
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.set_xlabel("Outcome", fontsize=label_fontsize)
        ax.tick_params(labelsize=tick_fontsize)
        if label_0 and label_1:
            ax.set_xticklabels([label_0, label_1], fontsize=tick_fontsize)
        ax.set_title(f"Outcome vs. {item}", fontsize=label_fontsize)

        # Legend labels: counts or percentages
        labels = [str(col) for col in ctab.columns]
        if show_value_counts:
            if not normalize:
                counts = df[item].value_counts()
                labels = [
                    f"{lbl} (n={counts.get(col, 0):,})"
                    for lbl, col in zip(labels, ctab.columns)
                ]
            else:
                proportions = df[item].value_counts(normalize=True)
                labels = [
                    f"{lbl} ({proportions.get(col, 0)*100:.0f}%)"
                    for lbl, col in zip(labels, ctab.columns)
                ]

        handles, _ = ax.get_legend_handles_labels()
        leg = ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=bbox_to_anchor,
            ncol=1,
            fontsize=label_fontsize,
        )
        for text in leg.get_texts():
            text.set_fontsize(label_fontsize)

    # Hide extra subplots
    for ax in axes[n_vars:]:
        ax.axis("off")

    plt.tight_layout(w_pad=w_pad, h_pad=h_pad)

    # Save logic
    if image_filename is not None:
        _save_figure(
            fig=fig,
            image_path_png=image_path_png,
            image_path_svg=image_path_svg,
            image_filename=image_filename,
        )

    elif save_plots:
        base = string or "crosstab_plot"
        safe = base.replace(" ", "_").replace(":", "").lower()
        if image_path_png:
            plt.savefig(
                os.path.join(image_path_png, f"{safe}.png"), bbox_inches="tight"
            )
        if image_path_svg:
            plt.savefig(
                os.path.join(image_path_svg, f"{safe}.svg"), bbox_inches="tight"
            )

    plt.show()


################################################################################
###################### Distribution Goodness-of-Fit Plots ######################
################################################################################


def distribution_gof_plots(
    df: pd.DataFrame,
    var: str,
    dist: Union[str, List[str]] = "norm",
    fit_method: str = "MLE",
    plot_types: Union[List[str], str] = ("qq", "cdf"),
    qq_type: str = "theoretical",
    reference_data: Optional[np.ndarray] = None,
    scale: str = "linear",
    tail: str = "both",
    figsize: Tuple[int, int] = (6, 5),
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_reference: bool = True,
    label_fontsize: int = 10,
    tick_fontsize: int = 10,
    legend_loc: str = "best",
    text_wrap: int = 50,
    palette: Optional[Dict[str, str]] = None,
    image_path_png: Optional[str] = None,
    image_path_svg: Optional[str] = None,
    image_filename: Optional[str] = None,
):
    """
    Generate goodness-of-fit (GOF) diagnostic plots for comparing empirical data
    to theoretical or empirical reference distributions.

    This function produces distributional diagnostics to assess how well one or
    more candidate probability distributions fit a given variable. Currently
    supported diagnostics include:

    - Quantile–Quantile (QQ) plots
    - Cumulative Distribution Function (CDF) plots
    - Exceedance (survival) plots via tail control

    Both theoretical (parametric) and empirical reference comparisons are supported
    for QQ plots.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data.

    var : str
        Column name in `df` to evaluate.

    dist : str or list of str, default="norm"
        Distribution name(s) to evaluate. Each entry must correspond to a valid
        continuous distribution in `scipy.stats` (e.g., "norm", "lognorm", "gamma").

    fit_method : {"MLE", "MM"}, default="MLE"
        Parameter estimation method for theoretical distributions:
        - "MLE": maximum likelihood estimation
        - "MM": method of moments

    plot_types : str or list of {"qq", "cdf"}, default=("qq", "cdf")
        Diagnostic plot type(s) to generate:
        - "qq": quantile–quantile plot
        - "cdf": cumulative distribution function or exceedance plot
        (controlled via `tail`)

    qq_type : {"theoretical", "empirical"}, default="theoretical"
        Type of QQ plot to generate:
        - "theoretical": sample quantiles vs fitted theoretical distribution
        - "empirical": sample quantiles vs reference empirical distribution

    reference_data : np.ndarray, optional
        Reference empirical sample used when `qq_type="empirical"`.
        Required for empirical QQ plots and ignored otherwise.
        Must contain at least two observations.

    scale : {"linear", "log"}, default="linear"
        Scale applied to the y-axis of applicable plots.

    tail : {"lower", "upper", "both"}, default="both"
        Tail behavior for CDF-based plots:
        - "lower": plot CDF only
        - "upper": plot exceedance probability (1 - CDF)
        - "both": plot both CDF and exceedance curves

    figsize : tuple of int, default=(6, 5)
        Base figure size for each diagnostic plot. When multiple plot types are
        requested, the total figure width scales accordingly.

    xlim : tuple of (float, float), optional
        Limits for the x-axis. Applied after all distributions are plotted to
        ensure consistent scaling.

    ylim : tuple of (float, float), optional
        Limits for the y-axis. Applied after all distributions are plotted to
        ensure consistent scaling.

    show_reference : bool, default=True
        Whether to draw a reference (identity) line on theoretical QQ plots.
        Ignored for empirical QQ plots.

    label_fontsize : int, default=10
        Font size for axis labels and titles.

    tick_fontsize : int, default=10
        Font size for axis tick labels.

    legend_loc : str, default="best"
        Legend placement passed to Matplotlib.

    text_wrap : int, default=50
        Maximum line width before wrapping titles and labels.

    palette : dict[str, str], optional
        Mapping from distribution name to color. Keys must match entries in `dist`.
        If not provided, default Matplotlib colors are used.

    image_path_png : str, optional
        Directory path for saving PNG output.

    image_path_svg : str, optional
        Directory path for saving SVG output.

    image_filename : str, optional
        Filename used when saving. With an extension it is saved verbatim (no
        directory required); without one it is combined with `image_path_png` /
        `image_path_svg`.


    Returns
    -------
    None
        The function generates GOF diagnostic plots and optionally saves them.

    Raises
    ------
    ValueError
        If invalid distribution names, plot types, or fitting methods are provided.

    ValueError
        If `qq_type="empirical"` is used without valid `reference_data`.

    ValueError
        If `image_filename` is provided without an extension and neither
        `image_path_png` nor `image_path_svg` is specified.

    Notes
    -----
    - Reference lines are drawn only for theoretical QQ plots and only when
    `show_reference=True`.
    - Empirical QQ plots compare matched quantiles between samples and do not
    assume a 1:1 identity line.
    - Axis limits (`xlim`, `ylim`) are applied after plotting all distributions to
    maintain comparability across fits.
    """

    # -------------------------
    # Normalize inputs
    # -------------------------
    if isinstance(dist, str):
        dist = [dist]

    if isinstance(plot_types, str):
        plot_types = [plot_types]

    valid_plot_types = {"qq", "cdf"}
    invalid = set(plot_types) - valid_plot_types
    if invalid:
        raise ValueError(
            f"Invalid plot_types: {invalid}. "
            f"Valid options are {sorted(valid_plot_types)}."
        )

    if qq_type not in {"theoretical", "empirical"}:
        raise ValueError("qq_type must be one of {'theoretical', 'empirical'}")

    if scale not in {"linear", "log"}:
        raise ValueError("scale must be 'linear' or 'log'")

    if tail not in {"both", "lower", "upper"}:
        raise ValueError("tail must be 'both', 'lower', or 'upper'")

    if palette:
        missing = set(dist) - set(palette)
        if missing:
            raise ValueError(
                f"Palette is missing color definitions for: {sorted(missing)}"
            )

    if qq_type == "empirical":
        if reference_data is None:
            raise ValueError("reference_data must be provided when qq_type='empirical'")
        if len(reference_data) < 2:
            raise ValueError("reference_data must contain at least 2 observations")

    if image_filename and not os.path.splitext(image_filename)[1]:
        if not (image_path_png or image_path_svg):
            raise ValueError(
                "You must provide either 'image_path_png' or 'image_path_svg' "
                "when 'image_filename' has no file extension."
            )

    # -------------------------
    # Extract data
    # -------------------------
    data = df[var].dropna().values

    # -------------------------
    # Create figure
    # -------------------------
    n_plots = len(plot_types)
    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(figsize[0] * n_plots, figsize[1]),
    )

    axes = np.atleast_1d(axes)

    # -------------------------
    # Plot diagnostics
    # -------------------------
    for ax, plot_type in zip(axes, plot_types):

        for i, d in enumerate(dist):
            color = palette.get(d) if palette else None
            dist_obj, params = _fit_distribution(data, d, fit_method)

            # Apply axis limits if provided
            if xlim is not None:
                ax.set_xlim(xlim)

            if ylim is not None:
                ax.set_ylim(ylim)

            if plot_type == "qq":
                _qq_plot(
                    ax=ax,
                    data=data,
                    dist_obj=dist_obj,
                    params=params,
                    label=d,
                    scale=scale,
                    label_fontsize=label_fontsize,
                    tick_fontsize=tick_fontsize,
                    qq_type=qq_type,
                    reference_data=reference_data,
                    show_reference=show_reference,
                    color=color,
                )

            elif plot_type == "cdf":
                _cdf_exceedance_plot(
                    ax=ax,
                    data=data,
                    dist_obj=dist_obj,
                    params=params,
                    label=d,
                    scale=scale,
                    tail=tail,
                    label_fontsize=label_fontsize,
                    color=color,
                )

        # Titles and formatting
        title = f"{plot_type.upper()} Plot for {var}"
        ax.set_title(
            "\n".join(textwrap.wrap(title, width=text_wrap)),
            fontsize=label_fontsize,
        )

        ax.tick_params(axis="both", labelsize=tick_fontsize)

        legend = ax.legend(loc=legend_loc)
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(label_fontsize)

    # -------------------------
    # Layout and saving
    # -------------------------
    plt.tight_layout()

    if image_path_png or image_path_svg or image_filename:
        _save_figure(
            fig=fig,
            image_path_png=image_path_png,
            image_path_svg=image_path_svg,
            filename=image_filename,
            image_filename=image_filename,
        )

    plt.show()


################################################################################
######################### Grouped Distributions Function #######################
################################################################################


def grouped_distributions(
    df: pd.DataFrame,
    features: List[str],
    by: str,
    *,
    bins: Union[int, List[Any], np.ndarray] = 30,
    normalize: str = "density",
    plot_style: str = "hist",
    pct_format: bool = True,
    alpha: float = 0.6,
    colors: Optional[Dict[str, str]] = None,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    common_bins: bool = True,
    show_legend: bool = True,
    legend_loc: str = "best",
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_ncols: int = 1,
    reverse_legend: bool = False,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    text_wrap: int = 50,
    figsize: Tuple[int, int] = (10, 6),
    suptitle: Optional[str] = None,
    suptitle_y: float = 1.02,
    xlim: Optional[Union[Tuple[float, float], Dict[str, Tuple[float, float]]]] = None,
    ylim: Optional[Union[Tuple[float, float], Dict[str, Tuple[float, float]]]] = None,
    w_pad: float = 1.0,
    h_pad: float = 1.0,
    bbox_inches: str = "tight",
    dpi: Optional[int] = None,
    image_path_png: Optional[str] = None,
    image_path_svg: Optional[str] = None,
    image_filename: Optional[str] = None,
) -> None:
    """
    Plot conditional distributions of numeric features given a binary variable.

    This function visualizes how selected numeric features are distributed
    conditional on a binary grouping variable (e.g., outcome or class label).
    For each feature, the distributions of the two groups are overlaid on the
    same axis, either as histograms or as filled density curves.

    Supported plot styles:
    - "hist": overlaid histograms using shared bins
    - "density": overlaid kernel density envelopes (always normalized)

    Normalization behavior:
    - normalize="count": raw counts (histograms only)
    - normalize="density": probability density (histograms or density plots)

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the features and grouping variable.

    features : list of str
        List of numeric feature column names to plot.

    by : str
        Name of the binary column used to condition the distributions.
        Must contain exactly two unique, non-null values.

    bins : int or sequence, optional (default=30)
        Number of bins or explicit bin edges for histogram plots.

    normalize : {"density", "count"}, optional (default="density")
        Controls histogram normalization. Ignored for density plots, which
        always use probability density.

    plot_style : {"hist", "density"}, optional (default="hist")
        Plot type to generate:
        - "hist": overlaid histograms
        - "density": overlaid filled KDE curves

    pct_format : bool, optional (default=True)
        If True, formats the y-axis as percentages when plot_style is
        "density" or normalize is "density". Set to False to display
        raw density values instead.

    alpha : float, optional (default=0.6)
        Transparency level for histogram bars or density fills.

    colors : dict, optional
        Mapping from group value to color. Keys must match the two unique
        values in the ``by`` column. If None, a default color scheme is used.

    n_rows : int, optional
        Number of subplot rows. If None, determined automatically.

    n_cols : int, optional
        Number of subplot columns. If None, determined automatically.

    common_bins : bool, optional (default=True)
        If True, both groups share identical histogram bin edges.
        Ignored for density plots.

    show_legend : bool, optional (default=True)
        Whether to display a legend identifying the two groups.

    legend_loc : str, optional (default="best")
        Legend location string passed to Matplotlib. Valid options include
        "best", "upper right", "upper left", "lower left", "lower right",
        "right", "center left", "center right", "lower center",
        "upper center", and "center".

    legend_bbox_to_anchor : tuple of float, optional
        (x, y) anchor point for placing the legend outside the axes.
        Use with legend_loc="upper center" and a negative y value to
        place the legend below the plot (e.g., (0.5, -0.15)).

    legend_ncols : int, optional (default=1)
        Number of columns in the legend. Use legend_ncols=2 when placing
        the legend below a subplot to display labels side by side.

    reverse_legend : bool, optional (default=False)
        If True, reverses the order of legend handles and labels.

    label_fontsize : int, optional (default=12)
        Font size for axis labels, subplot titles, and the figure-level
        suptitle.

    tick_fontsize : int, optional (default=10)
        Font size for tick labels and legend text.

    text_wrap : int, optional (default=50)
        Maximum character width before wrapping subplot titles and axis labels.

    figsize : tuple of (int, int), optional (default=(10, 6))
        Size of the overall figure in inches.

    suptitle : str, optional
        Figure-level title rendered above all subplots via ``fig.suptitle``.
        Font size is controlled by ``label_fontsize``. If None, no
        figure-level title is added.

    suptitle_y : float, optional (default=1.02)
        Vertical position of the figure-level suptitle as a fraction of the
        figure height. Values above 1.0 place the title above the subplot
        grid. Increase to add more space between the title and the plots.

    xlim : tuple of float or dict of {str: tuple of float}, optional
        Limits for the x-axis. Provide a single tuple ``(min, max)`` to apply
        uniformly to all subplots, or a dict keyed by feature name to set
        per-feature limits (e.g., ``{"Age": (0, 25), "BMI": (10, 60)}``).
        Features not present in the dict are left at Matplotlib defaults.

    ylim : tuple of float or dict of {str: tuple of float}, optional
        Limits for the y-axis. Accepts the same format as ``xlim``.

    w_pad : float, optional (default=1.0)
        Width padding between subplots passed to ``tight_layout``.

    h_pad : float, optional (default=1.0)
        Height padding between subplots passed to ``tight_layout``.

    bbox_inches : str, optional (default="tight")
        Bounding box option passed to ``_save_figure``.

    dpi : int, optional
        DPI for raster outputs such as PNG. Passed to ``_save_figure``.

    image_path_png : str, optional
        Directory path to save the PNG image.

    image_path_svg : str, optional
        Directory path to save the SVG image.

    image_filename : str, optional
        Filename for saving the figure. With an extension it is saved verbatim
        (no directory required); without one it is combined with
        ``image_path_png`` / ``image_path_svg``. No files are saved if this is
        not provided.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the ``by`` column is not binary.
        If an invalid ``plot_style`` or ``normalize`` option is provided.
        If ``image_filename`` is provided without an extension and neither
        ``image_path_png`` nor ``image_path_svg`` is specified.
        If the subplot grid is too small for the number of features.

    Warnings
    --------
    UserWarning
        If a feature has fewer than two non-null observations in either
        group, that group's density curve is skipped.

    Notes
    -----
    - All distributions are plotted as overlays on a single axis per feature.
    - ``xlim`` and ``ylim`` accept either a uniform tuple or a per-feature
      dict. Features absent from a dict are left at Matplotlib defaults.
    - ``w_pad`` and ``h_pad`` control subplot spacing via ``tight_layout``.
    - ``dpi`` and ``bbox_inches`` are forwarded to ``_save_figure`` and only
      take effect when ``image_filename`` is provided.
    - To place the legend below a subplot, use
      ``legend_loc="upper center"``, ``legend_bbox_to_anchor=(0.5, -0.15)``,
      and ``legend_ncols=2``.
    - In count mode (``normalize="count"``), the count (y) axis uses thousands
      separators. In density/percentage mode the y-axis shows whole-number
      percentages when ``pct_format=True``.
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if (
        image_filename
        and not os.path.splitext(image_filename)[1]
        and not (image_path_png or image_path_svg)
    ):
        raise ValueError(
            "You must provide either 'image_path_png' or 'image_path_svg' "
            "when 'image_filename' has no file extension."
        )

    if by not in df.columns:
        raise ValueError(f"Column '{by}' not found in DataFrame.")

    unique_vals = df[by].dropna().unique()
    if len(unique_vals) != 2:
        raise ValueError(f"Column '{by}' must be binary. Found values: {unique_vals}")

    if plot_style not in {"hist", "density"}:
        raise ValueError("plot_style must be 'hist' or 'density'.")

    if plot_style == "density" and normalize != "density":
        raise ValueError("Density plots only support normalize='density'.")

    if normalize not in {"density", "count"}:
        raise ValueError("normalize must be 'density' or 'count'.")

    features = list(features)
    n_features = len(features)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    if n_rows is None and n_cols is None:
        n_cols = int(np.ceil(np.sqrt(n_features)))
        n_rows = int(np.ceil(n_features / n_cols))
    elif n_rows is None:
        n_rows = int(np.ceil(n_features / n_cols))
    elif n_cols is None:
        n_cols = int(np.ceil(n_features / n_rows))

    if n_rows * n_cols < n_features:
        raise ValueError(
            f"Grid too small: n_rows*n_cols={n_rows * n_cols}, "
            f"but n_features={n_features}. Increase n_rows or n_cols."
        )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    # ------------------------------------------------------------------
    # Figure-level suptitle
    # ------------------------------------------------------------------
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=label_fontsize, y=suptitle_y)

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------
    if colors is None:
        colors = {
            unique_vals[0]: "#1f77b4",
            unique_vals[1]: "#ff7f0e",
        }

    # ------------------------------------------------------------------
    # Axis limit helper
    # ------------------------------------------------------------------
    def _resolve_lim(lim_param, feature):
        if lim_param is None:
            return None
        if isinstance(lim_param, dict):
            return lim_param.get(feature, None)
        return lim_param

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    for ax, feature in zip(axes, features):
        x0 = df.loc[df[by] == unique_vals[0], feature].dropna().values
        x1 = df.loc[df[by] == unique_vals[1], feature].dropna().values

        # -----------------------
        # Histogram style
        # -----------------------
        if plot_style == "hist":
            if common_bins:
                combined = np.concatenate([x0, x1])
                bin_edges = np.histogram_bin_edges(combined, bins=bins)
            else:
                bin_edges = bins

            density_flag = normalize == "density"

            ax.hist(
                x0,
                bins=bin_edges,
                alpha=alpha,
                color=colors[unique_vals[0]],
                label=str(unique_vals[0]),
                density=density_flag,
            )
            ax.hist(
                x1,
                bins=bin_edges,
                alpha=alpha,
                color=colors[unique_vals[1]],
                label=str(unique_vals[1]),
                density=density_flag,
            )

        # -----------------------
        # Filled density style
        # -----------------------
        else:
            for vals, label in [(x0, unique_vals[0]), (x1, unique_vals[1])]:
                if len(vals) < 2:
                    warnings.warn(
                        f"Feature '{feature}', group '{label}' has fewer than "
                        f"2 non-null observations. Density curve skipped.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

                kde = gaussian_kde(vals)
                x_grid = np.linspace(vals.min(), vals.max(), 500)
                y = kde(x_grid)

                ax.fill_between(
                    x_grid,
                    y,
                    alpha=alpha,
                    color=colors[label],
                    label=str(label),
                )

        # -----------------------
        # Axis labels
        # -----------------------
        ax.set_xlabel(
            "\n".join(textwrap.wrap(feature, width=text_wrap)),
            fontsize=label_fontsize,
        )

        if plot_style == "density" or normalize == "density":
            ax.set_ylabel(
                "Percentage" if pct_format else "Density",
                fontsize=label_fontsize,
            )
        else:
            ax.set_ylabel("Count", fontsize=label_fontsize)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

        ax.tick_params(axis="both", labelsize=tick_fontsize)

        # -----------------------
        # Legend
        # -----------------------
        if show_legend:
            _apply_legend(
                ax=ax,
                labels=None,
                loc=legend_loc,
                fontsize=tick_fontsize,
                reverse=reverse_legend,
                bbox_to_anchor=legend_bbox_to_anchor,
                ncols=legend_ncols,
            )

        # -----------------------
        # Subplot title
        # -----------------------
        ax.set_title(
            "\n".join(textwrap.wrap(f"{feature} by {by}", width=text_wrap)),
            fontsize=label_fontsize,
        )

        # -----------------------
        # Percentage formatter
        # -----------------------
        if pct_format and (plot_style == "density" or normalize == "density"):
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))

        # -----------------------
        # Axis limits
        # -----------------------
        x_lim = _resolve_lim(xlim, feature)
        y_lim = _resolve_lim(ylim, feature)

        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

    # ------------------------------------------------------------------
    # Hide unused axes
    # ------------------------------------------------------------------
    for ax in axes[n_features:]:
        ax.axis("off")

    plt.tight_layout(w_pad=w_pad, h_pad=h_pad)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if image_path_png or image_path_svg or image_filename:
        _save_figure(
            fig=fig,
            image_path_png=image_path_png,
            image_path_svg=image_path_svg,
            filename=image_filename,
            image_filename=image_filename,
            bbox_inches=bbox_inches,
            dpi=dpi,
        )

    plt.show()
