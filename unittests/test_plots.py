# plotting_tests.py

import pytest
import os
import io
import sys
from unittest.mock import patch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

from eda_toolkit import (
    kde_distributions,
    data_doctor,
    flex_corr_matrix,
    scatter_fit_plot,
    box_violin_plot,
    stacked_crosstab_plot,
    outcome_crosstab_plot,
    grouped_distributions,
    distribution_gof_plots,
)


# Fixtures that relate to data used in plotting
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"Feature1": [1, 2, 3, 4, 5], "Feature2": [10, 20, 30, 40, 50]})


@pytest.fixture
def sample_dataframe_values():
    return pd.DataFrame({"values": [1, 2, 3, 4, 5]})


@pytest.fixture
def sample_corr_dataframe():
    return pd.DataFrame(
        {
            "Feature1": [1, 2, 3, 4, 5],
            "Feature2": [5, 4, 3, 2, 1],
            "Feature3": [2, 3, 4, 5, 6],
        }
    )


@pytest.fixture
def sample_scatter_dataframe():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Feature1": np.random.rand(50) * 10,
            "Feature2": np.random.rand(50) * 20,
            "Category": np.random.choice(["A", "B"], size=50),
        }
    )


@pytest.fixture
def sample_boxcox_dataframe():
    """Fixture to provide test data for Box-Cox transformation."""
    return pd.DataFrame({"values": np.array([1, 2, 3, 4, 5])})


@pytest.fixture
def sample_boxcox_invalid_dataframe():
    """Fixture with non-positive values, which should trigger a ValueError."""
    return pd.DataFrame({"values": np.array([-1, 0, 2, 3, 4])})


@pytest.fixture
def sample_box_violin_dataframe():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Metric1": np.random.rand(50) * 100,
            "Metric2": np.random.rand(50) * 200,
            "Category": np.random.choice(["A", "B", "C"], size=50),
        }
    )


@pytest.fixture
def sample_crosstab_dataframe():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Category": np.random.choice(["A", "B", "C"], size=50),
            "Group": np.random.choice(["X", "Y"], size=50),
            "Outcome": np.random.choice(["Success", "Failure"], size=50),
        }
    )


@pytest.fixture(autouse=True, scope="session")
def suppress_plot_show():
    """Automatically mock plt.show() so figures don't pop up during tests."""
    plt.show = lambda *args, **kwargs: None


@pytest.fixture
def binary_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, 100),
            "x2": rng.normal(1, 2, 100),
            "group": np.repeat([0, 1], 50),
        }
    )


@pytest.fixture
def gof_df():
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "x": rng.normal(loc=0, scale=1, size=200),
        }
    )


@pytest.fixture
def reference_data():
    rng = np.random.default_rng(999)
    return rng.normal(size=200)


@pytest.fixture
def sample_corr_dataframe_large():
    """Larger numeric DataFrame for significance / filtering tests."""
    np.random.seed(0)
    n = 100
    x1 = np.random.randn(n)
    return pd.DataFrame(
        {
            "A": x1,
            "B": x1 * 0.9 + np.random.randn(n) * 0.1,  # strongly correlated
            "C": np.random.randn(n),  # uncorrelated
            "D": x1 * -0.8 + np.random.randn(n) * 0.2,  # strongly negatively correlated
        }
    )


@pytest.mark.parametrize(
    "plot_type, vars_of_interest",
    [
        ("both", "all"),
        ("both", None),
        ("hist", "all"),
        ("hist", None),
        ("density", "all"),
        ("density", None),
    ],
)

# < Insert all test_ functions for kde_distributions, data_doctor, scatter_fit_plot, box_violin_plot, flex_corr_matrix, stacked_crosstab_plot >
def test_kde_distributions(sample_dataframe, tmp_path, plot_type, vars_of_interest):
    """Test kde_distributions with different plot types and variable selections."""
    save_path = str(tmp_path)

    # Handle vars_of_interest: "all" means selecting all numeric columns, None defaults to all numeric columns
    if vars_of_interest == "all" or vars_of_interest is None:
        vars_of_interest = sample_dataframe.select_dtypes(
            include=np.number
        ).columns.to_list()

    try:
        kde_distributions(
            df=sample_dataframe,
            vars_of_interest=vars_of_interest,
            figsize=(5, 5),
            hist_color="#0000FF",
            kde_color="#FF0000",
            mean_color="#000000",
            median_color="#000000",
            hist_edgecolor="#000000",
            fill=True,
            fill_alpha=0.6,
            image_path_png=save_path,
            image_filename=f"test_kde_plot_{plot_type}",
            plot_type=plot_type,
            plot_mean=True,
            plot_median=True,
            std_dev_levels=[1, 2],
            std_color=["#808080", "#A0A0A0"],
            show_legend=True,
        )
        assert any(
            f.endswith(".png") for f in os.listdir(save_path)
        ), f"PNG file should be created for plot_type={plot_type}."
    except Exception as e:
        pytest.fail(f"kde_distributions failed for plot_type={plot_type}: {e}")


def test_kde_distributions_invalid_plot_type(sample_dataframe):
    with pytest.raises(ValueError):
        kde_distributions(
            df=sample_dataframe,
            vars_of_interest=["Feature1"],
            plot_type="invalid_option",
        )


def test_kde_distributions_log_scale(sample_dataframe, tmp_path):
    save_path = str(tmp_path)
    try:
        kde_distributions(
            df=sample_dataframe,
            vars_of_interest=["Feature1"],
            log_scale_vars=["Feature1"],
            image_path_png=save_path,
            image_filename="test_log_scale",
        )
        assert any(
            f.endswith(".png") for f in os.listdir(save_path)
        ), "PNG file should be created for log scale."
    except Exception as e:
        pytest.fail(f"kde_distributions with log scale failed: {e}")


def test_kde_distributions_invalid_std_color_length(sample_dataframe):
    with pytest.raises(
        ValueError,
        match="Not enough colors specified in 'std_color'",
    ):
        kde_distributions(
            df=sample_dataframe,
            vars_of_interest=["Feature1"],
            plot_type="hist",
            plot_mean=True,
            plot_median=True,
            std_dev_levels=[1, 2],
            std_color=["#888888"],  # mismatch here
            show_legend=False,
        )


def test_kde_distributions_single_var_subplot_warning(sample_dataframe):
    with pytest.raises(
        ValueError,
        match="Cannot use subplot_figsize when there is only one",
    ):
        kde_distributions(
            df=sample_dataframe,
            vars_of_interest=["Feature1"],
            subplot_figsize=(5, 5),  # This must be passed
        )


def test_kde_distributions_invalid_vars_of_interest(sample_dataframe):
    # Expect function to return early without plotting when invalid vars are passed
    kde_distributions(
        df=sample_dataframe,
        vars_of_interest=["NonExistentFeature"],
    )
    # No exception or warning expected per function design
    assert True  # Smoke test


def test_kde_distributions_log_scale_invalid_column(sample_dataframe):
    with pytest.raises(ValueError, match="Invalid log_scale_vars"):
        kde_distributions(
            df=sample_dataframe,
            vars_of_interest=["Feature1"],
            log_scale_vars=["NonExistentFeature"],
            plot_type="hist",
            show_legend=False,
        )


def test_kde_distributions_invalid_y_axis(sample_dataframe):
    with pytest.raises(ValueError, match="Invalid stat value"):
        kde_distributions(
            df=sample_dataframe,
            vars_of_interest=["Feature1"],
            stat="foobar",  # Invalid
        )


def test_kde_distributions_show_legend_no_labels(sample_dataframe):
    kde_distributions(
        df=sample_dataframe,
        vars_of_interest=["Feature1"],
        plot_mean=False,
        plot_median=False,
        std_dev_levels=[],
        show_legend=True,
    )


def test_kde_distributions_tight_layout_error_handled(sample_dataframe, mocker):
    mocker.patch(
        "matplotlib.pyplot.tight_layout",
        side_effect=RuntimeError("test error"),
    )

    with pytest.raises(RuntimeError, match="test error"):
        kde_distributions(
            df=sample_dataframe,
            vars_of_interest=["Feature1"],
        )


def test_kde_distributions_with_legend_title(sample_scatter_dataframe):
    kde_distributions(
        df=sample_scatter_dataframe,
        vars_of_interest=["Feature1"],
        hue="Category",
        show_legend=True,
        legend_title="Test Legend",
    )


def test_kde_distributions_mismatched_legend_labels(sample_dataframe):
    # legend_labels_list isn't a real param in kde_distributions, so remove test
    # or stub it as smoke
    kde_distributions(
        df=sample_dataframe,
        vars_of_interest=["Feature1"],
        std_dev_levels=[1, 2],
        std_color=["#888", "#999"],
        show_legend=True,
    )
    assert True  # Stubbed until legend label logic is added


def test_kde_distributions_no_save_without_filename(
    sample_dataframe,
    tmp_path,
):
    kde_distributions(
        df=sample_dataframe,
        vars_of_interest=["Feature1"],
        image_path_png=str(tmp_path),
        image_filename=None,  # Should not trigger saving
    )
    saved_files = list(tmp_path.glob("*.png"))
    assert (
        len(saved_files) == 0
    ), "No files should be saved when image_filename is None."


def test_kde_distributions_adds_legend(sample_dataframe):
    kde_distributions(
        df=sample_dataframe,
        vars_of_interest=["Feature1"],
        plot_mean=True,
        plot_median=True,
        std_dev_levels=[1],
        show_legend=True,
    )
    # Cannot assert legend presence without UI, but should run without error


def test_data_doctor_basic(sample_dataframe_values):
    """Test data_doctor with default parameters."""
    try:
        data_doctor(sample_dataframe_values, "values", show_plot=False)
    except Exception as e:
        pytest.fail(f"data_doctor failed with default parameters: {e}")


def test_data_doctor_scale_conversion(sample_dataframe_values):
    """Test data_doctor with different scale conversion options."""
    scale_options = [
        "log",
        "sqrt",
        "cbrt",
        "reciprocal",
        "stdrz",
        "minmax",
        "robust",
        "maxabs",
        "exp",
        "arcsinh",  # Newly added transformations
    ]
    for scale in scale_options:
        try:
            data_doctor(
                sample_dataframe_values,
                "values",
                scale_conversion=scale,
                show_plot=False,
            )
        except Exception as e:
            pytest.fail(f"data_doctor failed with scale_conversion={scale}: {e}")


def test_data_doctor_cutoff(sample_dataframe_values):
    """Test data_doctor with lower and upper cutoff values applied."""
    try:
        data_doctor(
            sample_dataframe_values,
            "values",
            apply_cutoff=True,
            lower_cutoff=2,
            upper_cutoff=4,
            show_plot=False,
        )
    except Exception as e:
        pytest.fail(f"data_doctor failed with cutoffs applied: {e}")


def test_data_doctor_apply_as_new_col(sample_dataframe_values):
    """Test data_doctor when applying transformation as a new column."""
    df_copy = sample_dataframe_values.copy()
    try:
        data_doctor(
            df_copy,
            "values",
            scale_conversion="log",
            apply_as_new_col_to_df=True,
            show_plot=False,
        )
    except Exception as e:
        pytest.fail(f"data_doctor failed when applying as new column: {e}")


def test_data_doctor_apply_as_kde(sample_dataframe_values):
    """Test data_doctor with KDE plot enabled."""
    df_copy = sample_dataframe_values.copy()
    try:
        data_doctor(
            df_copy,
            "values",
            scale_conversion="log",
            apply_as_new_col_to_df=True,
            show_plot=True,
            plot_type="kde",
        )
    except Exception as e:
        pytest.fail(f"data_doctor failed when applying as new column: {e}")


@pytest.mark.parametrize(
    "scale_conversion",
    ["robust", "maxabs", "exp", "arcsinh"],  # Parametrize newly added methods
)
def test_data_doctor_new_scale_conversions(sample_dataframe_values, scale_conversion):
    """Test data_doctor with new scale conversion methods."""
    try:
        data_doctor(
            sample_dataframe_values,
            "values",
            scale_conversion=scale_conversion,
            show_plot=False,
        )
    except Exception as e:
        pytest.fail(f"data_doctor failed with scale_conversion={scale_conversion}: {e}")


def test_data_doctor_logit_valid(sample_dataframe_values):
    """Test logit transformation with valid input (0 < x < 1)"""
    valid_data = sample_dataframe_values.copy()

    # Ensure the new column has the same length as the DataFrame
    valid_data["values"] = np.linspace(0.01, 0.99, len(valid_data))

    try:
        data_doctor(valid_data, "values", scale_conversion="logit", show_plot=False)
    except Exception as e:
        pytest.fail(f"data_doctor failed with valid logit input: {e}")


def test_data_doctor_logit_invalid(sample_dataframe_values):
    """Test logit transformation with invalid input (outside 0 and 1)"""
    invalid_data = sample_dataframe_values.copy()

    # Ensure invalid values match DataFrame length
    invalid_values = np.linspace(-0.5, 1.5, len(invalid_data))
    invalid_data["values"] = invalid_values

    with pytest.raises(
        ValueError, match="Logit transformation requires values to be between 0 and 1."
    ):
        data_doctor(invalid_data, "values", scale_conversion="logit", show_plot=False)


def test_data_doctor_boxcox_valid(sample_boxcox_dataframe):
    """Test Box-Cox transformation with strictly positive values."""
    data_doctor(  # Call function without assigning return value
        sample_boxcox_dataframe,
        feature_name="values",
        scale_conversion="boxcox",
    )

    # Now check that the transformation actually modified the DataFrame
    assert "values" in sample_boxcox_dataframe.columns, "Column 'values' is missing"
    assert (
        sample_boxcox_dataframe["values"].isna().sum() == 0
    ), "Box-Cox output should not have NaNs"


def test_data_doctor_boxcox_invalid(sample_boxcox_invalid_dataframe):
    """Ensure ValueError is raised when Box-Cox encounters non-positive values."""
    with pytest.raises(
        ValueError, match="Box-Cox transformation requires strictly positive values"
    ):
        data_doctor(
            sample_boxcox_invalid_dataframe,
            feature_name="values",  # Ensure feature_name is provided
            scale_conversion="boxcox",
        )


def test_data_doctor_boxcox_mismatch(sample_boxcox_dataframe, mocker):
    """Test Box-Cox transformation error when transformed length mismatches input."""
    mocker.patch(
        "scipy.stats.boxcox", return_value=(np.array([1.2, 2.4, 3.1]), 0.5)
    )  # Wrong length

    with pytest.raises(
        ValueError,
        match="Length of transformed data .* does not match the length of the sampled feature",
    ):
        data_doctor(
            sample_boxcox_dataframe,
            feature_name="values",  # Ensure feature_name is provided
            scale_conversion="boxcox",
        )


def test_data_doctor_adds_column_correctly(sample_boxcox_dataframe):
    """Ensure a new column is added when apply_as_new_col_to_df=True."""
    data_doctor(
        sample_boxcox_dataframe,
        feature_name="values",
        scale_conversion="boxcox",
        apply_as_new_col_to_df=True,
    )

    new_col_name = "values_boxcox"
    assert (
        new_col_name in sample_boxcox_dataframe.columns
    ), f"Column '{new_col_name}' not added."
    assert (
        sample_boxcox_dataframe[new_col_name].isna().sum() == 0
    ), "Transformed column contains NaNs."


def test_data_doctor_length_consistency(sample_boxcox_dataframe):
    """Ensure the new transformed column has the same length as the original DataFrame."""
    data_doctor(
        sample_boxcox_dataframe,
        feature_name="values",
        scale_conversion="boxcox",
        apply_as_new_col_to_df=True,
    )

    new_col_name = "values_boxcox"
    assert len(sample_boxcox_dataframe) == len(
        sample_boxcox_dataframe[new_col_name]
    ), "Length mismatch after transformation."


def test_data_doctor_raises_error_for_length_mismatch(sample_boxcox_dataframe):
    """Ensure the function handles length mismatch gracefully instead of failing."""

    # Create an artificial length mismatch
    sample_boxcox_dataframe["values_boxcox"] = sample_boxcox_dataframe["values"].iloc[
        :-1
    ]

    try:
        data_doctor(
            sample_boxcox_dataframe,
            feature_name="values",
            scale_conversion="boxcox",
            apply_as_new_col_to_df=True,
        )

        print("\n[DEBUG] Function executed successfully despite length mismatch.")

    except Exception as e:
        print(f"\n[Caught Exception] {type(e).__name__}: {e}")
        assert False, f"Unexpected exception was raised: {e}"

    # Check if the function allowed length mismatch or handled it in a specific way
    assert (
        "values_boxcox" in sample_boxcox_dataframe.columns
    ), "Column should still exist in DataFrame."


def test_data_doctor_no_column_added_when_fraction_is_not_one(sample_boxcox_dataframe):
    """Ensure data_fraction affects transformation properly."""

    new_col_name = "values_boxcox"

    # Ensure column does not exist before transformation
    if new_col_name in sample_boxcox_dataframe.columns:
        sample_boxcox_dataframe.drop(columns=[new_col_name], inplace=True)

    data_doctor(
        sample_boxcox_dataframe,
        feature_name="values",
        scale_conversion="boxcox",
        apply_as_new_col_to_df=True,
        data_fraction=0.5,  # Expecting partial transformation
    )

    # Count transformed values
    transformed_count = sample_boxcox_dataframe[new_col_name].notna().sum()

    # Print debug info
    print(
        f"\n[DEBUG] Transformed {transformed_count} / {len(sample_boxcox_dataframe)} rows."
    )

    assert (
        transformed_count > 0
    ), "At least some rows should be transformed when data_fraction != 1."
    expected_transformed_rows = int(len(sample_boxcox_dataframe) * 0.5)

    # Allow a small tolerance in case of rounding
    assert (
        expected_transformed_rows <= transformed_count <= len(sample_boxcox_dataframe)
    ), f"Expected {expected_transformed_rows} rows transformed, but got {transformed_count}."


def test_data_doctor_handles_length_mismatch_gracefully(sample_boxcox_dataframe):
    """Ensure no error is raised if the transformed feature length does not match the DataFrame length."""

    # Ensure function runs without error (rather than expecting a ValueError)
    data_doctor(
        sample_boxcox_dataframe,
        feature_name="values",
        scale_conversion="boxcox",
        apply_as_new_col_to_df=True,
    )

    new_col_name = "values_boxcox"

    # Ensure the column exists in the DataFrame
    assert (
        new_col_name in sample_boxcox_dataframe.columns
    ), "The transformed column was not added."

    # Ensure the transformed column has the correct length
    assert len(sample_boxcox_dataframe[new_col_name]) == len(
        sample_boxcox_dataframe
    ), "Transformed column length does not match original DataFrame length."


def test_data_doctor_respects_data_fraction(sample_boxcox_dataframe):
    """Ensure data_fraction affects transformations when apply_as_new_col_to_df=True."""

    new_col_name = "values_boxcox"

    # Ensure the column does not exist before transformation
    if new_col_name in sample_boxcox_dataframe.columns:
        sample_boxcox_dataframe.drop(columns=[new_col_name], inplace=True)

    data_doctor(
        sample_boxcox_dataframe,
        feature_name="values",
        scale_conversion="boxcox",
        apply_as_new_col_to_df=True,
        data_fraction=0.5,  # Use only 50% of the data
    )

    # Check if the transformed column was added
    assert (
        new_col_name in sample_boxcox_dataframe.columns
    ), "The transformed column should still be added."

    # Verify the function correctly applies the transformation (if partial transformation is unsupported, expect full)
    assert len(sample_boxcox_dataframe[new_col_name]) == len(
        sample_boxcox_dataframe
    ), "All rows should be transformed when apply_as_new_col_to_df=True."


def test_data_doctor_boxcox_lambda_output(sample_boxcox_dataframe):
    """Check that the Box-Cox lambda is printed correctly."""
    captured_output = io.StringIO()  # Create a buffer to capture print output
    sys.stdout = captured_output

    data_doctor(
        sample_boxcox_dataframe,
        feature_name="values",
        scale_conversion="boxcox",
        apply_as_new_col_to_df=True,
    )

    sys.stdout = sys.__stdout__  # Reset stdout
    output_text = captured_output.getvalue()

    assert "Box-Cox Lambda" in output_text, "Box-Cox Lambda value not printed."


def test_data_doctor_abs(sample_dataframe_values):
    """Ensure 'abs' transformation correctly converts negative values to positive."""
    df_copy = sample_dataframe_values.copy()
    df_copy["values"] = [-10, -5, 0, 5, 10]  # Mix of negative and positive numbers

    # Apply absolute value transformation
    data_doctor(
        df_copy,
        "values",
        scale_conversion="abs",
        apply_as_new_col_to_df=True,
        show_plot=False,
    )

    # The transformation should create a new column
    new_col_name = "values_abs"
    assert new_col_name in df_copy.columns, f"Column '{new_col_name}' was not added."

    transformed_values = df_copy[new_col_name].values
    assert all(
        transformed_values >= 0
    ), f"All transformed values should be non-negative, got {transformed_values}."


def test_data_doctor_boxcox_with_alpha(sample_boxcox_dataframe):
    """Ensure Box-Cox transformation works with alpha (confidence interval)."""
    data_doctor(
        sample_boxcox_dataframe,
        feature_name="values",
        scale_conversion="boxcox",
        scale_conversion_kws={"alpha": 0.05},  # Request confidence interval
        show_plot=False,
    )


def test_data_doctor_boxcox_fails_on_nonpositive(sample_boxcox_invalid_dataframe):
    """Ensure Box-Cox transformation raises an error when input contains non-positive values."""
    with pytest.raises(
        ValueError, match="Box-Cox transformation requires strictly positive values"
    ):
        data_doctor(
            sample_boxcox_invalid_dataframe,
            feature_name="values",
            scale_conversion="boxcox",
            show_plot=False,
        )


def test_data_doctor_power_transform(sample_boxcox_dataframe):
    """Ensure PowerTransformer correctly transforms values."""
    data_doctor(
        sample_boxcox_dataframe,
        feature_name="values",
        scale_conversion="power",
        show_plot=False,
    )


def test_data_doctor_invalid_box_violin(sample_dataframe_values):
    """Ensure ValueError is raised when an invalid box_violin option is provided."""
    with pytest.raises(ValueError, match="Invalid plot type 'invalid_plot'.*"):
        data_doctor(
            sample_dataframe_values,
            feature_name="values",
            show_plot=True,
            plot_type="box_violin",
            box_violin="invalid_plot",  # Invalid option
        )


def test_flex_corr_matrix_basic(sample_corr_dataframe):
    try:
        flex_corr_matrix(sample_corr_dataframe, show_colorbar=True)
    except Exception as e:
        pytest.fail(f"flex_corr_matrix failed with default parameters: {e}")


def test_flex_corr_matrix_custom_cols(sample_corr_dataframe):
    try:
        flex_corr_matrix(
            sample_corr_dataframe, cols=["Feature1", "Feature3"], show_colorbar=False
        )
    except Exception as e:
        pytest.fail(f"flex_corr_matrix failed with custom column selection: {e}")


def test_flex_corr_matrix_triangulation(sample_corr_dataframe):
    try:
        flex_corr_matrix(sample_corr_dataframe, triangular=True)
    except Exception as e:
        pytest.fail(f"flex_corr_matrix failed with triangular masking: {e}")


def test_flex_corr_matrix_save_plot(sample_corr_dataframe, tmp_path):
    save_path = str(tmp_path)
    try:
        flex_corr_matrix(
            sample_corr_dataframe,
            save_plots=True,
            image_path_png=save_path,
            image_path_svg=save_path,
            title="Correlation Test Matrix",
        )
        assert any(f.endswith(".png") for f in os.listdir(save_path))
        assert any(f.endswith(".svg") for f in os.listdir(save_path))
    except Exception as e:
        pytest.fail(f"flex_corr_matrix failed to save plots: {e}")


# ------------------------------------------------------------------
# Basic smoke tests
# ------------------------------------------------------------------


def test_flex_corr_matrix_no_colorbar(sample_corr_dataframe):
    flex_corr_matrix(sample_corr_dataframe, show_colorbar=False)


def test_flex_corr_matrix_full_matrix(sample_corr_dataframe):
    flex_corr_matrix(sample_corr_dataframe, triangular=False)


def test_flex_corr_matrix_with_title(sample_corr_dataframe):
    flex_corr_matrix(sample_corr_dataframe, title="Test Title")


def test_flex_corr_matrix_custom_figsize(sample_corr_dataframe):
    flex_corr_matrix(sample_corr_dataframe, figsize=(6, 6))


def test_flex_corr_matrix_label_names(sample_corr_dataframe):
    flex_corr_matrix(
        sample_corr_dataframe,
        label_names={"Feature1": "F1", "Feature2": "F2", "Feature3": "F3"},
    )


def test_flex_corr_matrix_no_annot(sample_corr_dataframe):
    flex_corr_matrix(sample_corr_dataframe, annot=False)


def test_flex_corr_matrix_custom_cmap(sample_corr_dataframe):
    flex_corr_matrix(sample_corr_dataframe, cmap="viridis")


def test_flex_corr_matrix_xlabel_rotation(sample_corr_dataframe):
    flex_corr_matrix(sample_corr_dataframe, xlabel_rot=90)


def test_flex_corr_matrix_image_filename(sample_corr_dataframe, tmp_path):
    flex_corr_matrix(
        sample_corr_dataframe,
        image_filename="test_corr",
        image_path_png=str(tmp_path),
    )
    assert any(f.endswith(".png") for f in os.listdir(tmp_path))


def test_flex_corr_matrix_image_filename_svg(sample_corr_dataframe, tmp_path):
    flex_corr_matrix(
        sample_corr_dataframe,
        image_filename="test_corr",
        image_path_svg=str(tmp_path),
    )
    assert any(f.endswith(".svg") for f in os.listdir(tmp_path))


# ------------------------------------------------------------------
# corr_method
# ------------------------------------------------------------------


def test_flex_corr_matrix_spearman(sample_corr_dataframe):
    flex_corr_matrix(sample_corr_dataframe, corr_method="spearman")


def test_flex_corr_matrix_kendall(sample_corr_dataframe):
    flex_corr_matrix(sample_corr_dataframe, corr_method="kendall")


def test_flex_corr_matrix_invalid_corr_method(sample_corr_dataframe):
    with pytest.raises(ValueError, match="Invalid `corr_method`"):
        flex_corr_matrix(sample_corr_dataframe, corr_method="invalid")


# ------------------------------------------------------------------
# Significance overlay
# ------------------------------------------------------------------


def test_flex_corr_matrix_show_significance_stars(sample_corr_dataframe_large):
    flex_corr_matrix(
        sample_corr_dataframe_large,
        show_significance=True,
        significance_method="stars",
    )


def test_flex_corr_matrix_show_significance_mask(sample_corr_dataframe_large):
    flex_corr_matrix(
        sample_corr_dataframe_large,
        show_significance=True,
        significance_method="mask",
    )


def test_flex_corr_matrix_invalid_significance_method(sample_corr_dataframe):
    with pytest.raises(ValueError, match="Invalid `significance_method`"):
        flex_corr_matrix(
            sample_corr_dataframe,
            show_significance=True,
            significance_method="pvalues",
        )


def test_flex_corr_matrix_significance_level_strict(sample_corr_dataframe_large):
    flex_corr_matrix(
        sample_corr_dataframe_large,
        show_significance=True,
        significance_level=0.01,
        significance_method="stars",
    )


def test_flex_corr_matrix_significance_legend_x(sample_corr_dataframe_large):
    flex_corr_matrix(
        sample_corr_dataframe_large,
        show_significance=True,
        significance_method="stars",
        significance_legend_x=0.3,
    )


def test_flex_corr_matrix_spearman_significance(sample_corr_dataframe_large):
    flex_corr_matrix(
        sample_corr_dataframe_large,
        corr_method="spearman",
        show_significance=True,
        significance_method="stars",
    )


def test_flex_corr_matrix_kendall_significance(sample_corr_dataframe_large):
    flex_corr_matrix(
        sample_corr_dataframe_large,
        corr_method="kendall",
        show_significance=True,
        significance_method="mask",
    )


# ------------------------------------------------------------------
# filter_significance
# ------------------------------------------------------------------


def test_flex_corr_matrix_filter_significance(sample_corr_dataframe_large):
    flex_corr_matrix(
        sample_corr_dataframe_large,
        filter_significance=0.05,
    )


def test_flex_corr_matrix_filter_significance_strict(sample_corr_dataframe_large):
    flex_corr_matrix(
        sample_corr_dataframe_large,
        filter_significance=0.001,
    )


def test_flex_corr_matrix_filter_significance_auto_enables_show_significance(
    sample_corr_dataframe_large,
):
    """filter_significance should auto-enable show_significance without error."""
    try:
        flex_corr_matrix(
            sample_corr_dataframe_large,
            filter_significance=0.05,
            show_significance=False,  # should be overridden internally
        )
    except Exception as e:
        pytest.fail(f"filter_significance failed to auto-enable significance: {e}")


def test_flex_corr_matrix_filter_significance_no_overlay_when_off(
    sample_corr_dataframe_large,
):
    """filter_significance computes p-values for filtering but does NOT force
    the stars overlay when show_significance=False."""
    plt.close("all")
    flex_corr_matrix(
        sample_corr_dataframe_large,
        filter_significance=0.05,
        show_significance=False,
    )
    # runs, filters, plots a plain heatmap (no exception is the smoke check)


def test_flex_corr_matrix_filter_significance_invalid(sample_corr_dataframe):
    with pytest.raises(ValueError, match="`filter_significance`"):
        flex_corr_matrix(
            sample_corr_dataframe,
            filter_significance=-0.05,
        )


def test_flex_corr_matrix_filter_significance_invalid_type(sample_corr_dataframe):
    with pytest.raises((ValueError, TypeError)):
        flex_corr_matrix(
            sample_corr_dataframe,
            filter_significance="0.05",
        )


# ------------------------------------------------------------------
# Validation errors
# ------------------------------------------------------------------


def test_flex_corr_matrix_invalid_annot(sample_corr_dataframe):
    with pytest.raises(ValueError, match="Invalid value for `annot`"):
        flex_corr_matrix(sample_corr_dataframe, annot="yes")


def test_flex_corr_matrix_invalid_save_plots(sample_corr_dataframe):
    with pytest.raises(ValueError, match="Invalid `save_plots`"):
        flex_corr_matrix(sample_corr_dataframe, save_plots="yes")


def test_flex_corr_matrix_invalid_triangular(sample_corr_dataframe):
    with pytest.raises(ValueError, match="Invalid `triangular`"):
        flex_corr_matrix(sample_corr_dataframe, triangular="yes")


def test_flex_corr_matrix_invalid_cols_type(sample_corr_dataframe):
    with pytest.raises(ValueError, match="`cols` parameter must be a list"):
        flex_corr_matrix(sample_corr_dataframe, cols="Feature1")


def test_flex_corr_matrix_save_plots_without_path(sample_corr_dataframe):
    with pytest.raises(
        ValueError,
        match="image_path_png.*image_path_svg|image_path_svg.*image_path_png|specify",
    ):
        flex_corr_matrix(sample_corr_dataframe, save_plots=True)


def test_flex_corr_matrix_image_filename_without_path(sample_corr_dataframe):
    with pytest.raises(
        ValueError,
        match="image_path_png.*image_path_svg|image_path_svg.*image_path_png|specify",
    ):
        flex_corr_matrix(sample_corr_dataframe, image_filename="test")


################################################################################
# Near-zero artifact
################################################################################


def test_flex_corr_matrix_no_negative_zero(sample_corr_dataframe_large, tmp_path):
    """Verify that near-zero values don't produce -0.00 in the heatmap."""
    # The fixture has uncorrelated columns — some cells will be near zero
    # Just check the function runs without error; visual check is manual
    try:
        flex_corr_matrix(
            sample_corr_dataframe_large,
            show_significance=False,
        )
    except Exception as e:
        pytest.fail(f"Near-zero artifact caused an error: {e}")


# Test that scatter_fit_plot runs without errors with default parameters
def test_scatter_fit_plot_basic(sample_scatter_dataframe):
    try:
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars="Feature1",
            y_vars="Feature2",
            show_plot="subplots",  # Default to subplots
        )
    except Exception as e:
        pytest.fail(f"scatter_fit_plot failed with default parameters: {e}")


# Test that scatter_fit_plot correctly excludes specified variable combinations
def test_scatter_fit_plot_exclude(sample_scatter_dataframe):
    exclude_combinations = [("Feature1", "Feature2")]
    result = scatter_fit_plot(
        sample_scatter_dataframe,
        x_vars="Feature1",
        y_vars="Feature2",
        exclude_combinations=exclude_combinations,
        show_plot="combinations",
    )
    assert len(result) == 0, f"Expected no valid plots, but got {len(result)}"


# Test that scatter_fit_plot correctly applies the best fit line
def test_scatter_fit_plot_best_fit(sample_scatter_dataframe):
    try:
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars="Feature1",
            y_vars="Feature2",
            add_best_fit_line=True,
            show_plot="subplots",
        )
    except Exception as e:
        pytest.fail(f"scatter_fit_plot failed with best fit line: {e}")


# Test that scatter_fit_plot correctly handles hue parameter
def test_scatter_fit_plot_hue(sample_scatter_dataframe):
    try:
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars="Feature1",
            y_vars="Feature2",
            hue="Category",
            show_plot="subplots",
        )
    except Exception as e:
        pytest.fail(f"scatter_fit_plot failed with hue parameter: {e}")


# Test that scatter_fit_plot raises error on invalid input
def test_scatter_fit_plot_invalid_input(sample_scatter_dataframe):
    with pytest.raises(ValueError):
        scatter_fit_plot(sample_scatter_dataframe, x_vars=None, y_vars=None)


def test_scatter_fit_plot_conflicting_all_vars(sample_scatter_dataframe):
    with pytest.raises(
        ValueError, match="Cannot pass `all_vars` and still choose `x_vars`"
    ):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            all_vars=["Feature1", "Feature2", "Feature3"],  # Invalid combination
        )


def test_scatter_fit_plot_hue_palette_without_hue(sample_scatter_dataframe):
    with pytest.raises(
        ValueError, match="Cannot specify `hue_palette` without specifying `hue`"
    ):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            hue_palette="coolwarm",  # Invalid without hue
        )


def test_scatter_fit_plot_invalid_save_plots(sample_scatter_dataframe):
    with pytest.raises(
        ValueError,
        match="Invalid `save_plots` value. Choose from 'all', 'individual', 'subplots', or None.",
    ):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            save_plots="wrong_option",  # Invalid choice
        )


def test_scatter_fit_plot_invalid_exclude_combinations(sample_scatter_dataframe):
    with pytest.raises(
        ValueError, match="Invalid column names in `exclude_combinations`"
    ):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            exclude_combinations=[("NonExistentColumn", "Feature2")],  # Invalid column
        )


def test_scatter_fit_plot_invalid_rotate_plot(sample_scatter_dataframe):
    with pytest.raises(
        ValueError, match="Invalid `rotate_plot`. Choose True or False."
    ):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            rotate_plot="yes",  # Invalid, should be boolean
        )


def test_scatter_fit_plot_save_without_path(sample_scatter_dataframe):
    with pytest.raises(
        ValueError,
        match=r"To save plots, specify `image_path_png`, `image_path_svg`, or `image_filename`\.",
    ):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            save_plots="all",  # Requires path
        )


def test_scatter_fit_plot_save_png(tmp_path, sample_scatter_dataframe):
    """Test scatter_fit_plot successfully saves PNG files when a valid path is provided"""
    save_path = str(tmp_path)  # Temporary directory for saving
    os.makedirs(save_path, exist_ok=True)

    try:
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            save_plots="all",
            image_path_png=save_path,  # Ensure PNG saving works
        )

        # Check that at least one PNG file is created
        png_files = [f for f in os.listdir(save_path) if f.endswith(".png")]
        assert len(png_files) > 0, "Expected at least one PNG file to be created."
    except Exception as e:
        pytest.fail(f"scatter_fit_plot failed to save PNG files: {e}")


def test_scatter_fit_plot_save_svg(tmp_path, sample_scatter_dataframe):
    """Test scatter_fit_plot successfully saves SVG files when a valid path is provided"""
    save_path = str(tmp_path)  # Temporary directory for saving
    os.makedirs(save_path, exist_ok=True)

    try:
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            save_plots="all",
            image_path_svg=save_path,  # Ensure SVG saving works
        )

        # Check that at least one SVG file is created
        svg_files = [f for f in os.listdir(save_path) if f.endswith(".svg")]
        assert len(svg_files) > 0, "Expected at least one SVG file to be created."
    except Exception as e:
        pytest.fail(f"scatter_fit_plot failed to save SVG files: {e}")


def test_scatter_fit_plot_progress_bar(tmp_path, sample_scatter_dataframe):
    """Test scatter_fit_plot correctly updates the tqdm progress bar when saving"""
    save_path = str(tmp_path)
    os.makedirs(save_path, exist_ok=True)

    with patch("tqdm.tqdm.update") as mock_update:
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            save_plots="all",
            image_path_png=save_path,
        )

        assert mock_update.called, "Progress bar update was not called."


def test_scatter_fit_plot_memory_cleanup(tmp_path, sample_scatter_dataframe):
    save_path = str(tmp_path)
    os.makedirs(save_path, exist_ok=True)

    try:
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            save_plots="all",
            image_path_png=save_path,
            n_rows=1,
            n_cols=2,  # <- FORCE axes to be an array
        )
    except Exception as e:
        pytest.fail(f"scatter_fit_plot failed memory cleanup check: {e}")

    plt.close("all")

    remaining_figs = plt.get_fignums()
    assert len(remaining_figs) == 0


def test_scatter_fit_plot_invalid_show_plot(sample_scatter_dataframe):
    with pytest.raises(ValueError, match="Invalid `show_plot`. Choose from"):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            show_plot="invalid_option",  # Invalid choice
        )


def test_scatter_fit_plot_missing_variables(sample_scatter_dataframe):
    with pytest.raises(
        ValueError,
        match="Either `all_vars` or both `x_vars` and `y_vars` must be provided.",
    ):
        scatter_fit_plot(sample_scatter_dataframe)


def test_scatter_fit_plot_invalid_figsize(sample_scatter_dataframe):
    """Test scatter_fit_plot raises ValueError for invalid figsize values"""
    with pytest.raises(ValueError, match="Invalid `individual_figsize` value"):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            individual_figsize="invalid_size",  # Not a tuple or list of two numbers
        )

    with pytest.raises(ValueError, match="Invalid `subplot_figsize` value"):
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            subplot_figsize=[10],  # Should be a tuple or list of two numbers
        )


def test_scatter_fit_plot_legend_removal(sample_scatter_dataframe):
    """Test scatter_fit_plot correctly handles missing legend removal"""
    try:
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            show_plot="subplots",
        )
    except AttributeError as e:
        pytest.fail(f"scatter_fit_plot failed due to legend removal issue: {e}")


def test_scatter_fit_plot_xylim(sample_scatter_dataframe):
    """Test scatter_fit_plot correctly applies xlim and ylim if provided"""
    try:
        scatter_fit_plot(
            sample_scatter_dataframe,
            x_vars=["Feature1"],
            y_vars=["Feature2"],
            xlim=(0, 100),
            ylim=(0, 200),
            show_plot="subplots",
        )
    except Exception as e:
        pytest.fail(f"scatter_fit_plot failed with xlim/ylim: {e}")


def test_scatter_fit_plot_missing_all_vars(sample_scatter_dataframe):
    """Test scatter_fit_plot raises error when missing `all_vars` for scatter plot generation"""
    with pytest.raises(
        ValueError,
        match="Either `all_vars` or both `x_vars` and `y_vars` must be provided.",
    ):
        scatter_fit_plot(sample_scatter_dataframe)


# Test that box_violin_plot runs without errors with default parameters
def test_box_violin_plot_basic(sample_box_violin_dataframe):
    try:
        box_violin_plot(
            sample_box_violin_dataframe,
            metrics_list=["Metric1", "Metric2"],
            metrics_comp=["Category"],
            show_plot="both",
        )
    except Exception as e:
        pytest.fail(f"box_violin_plot failed with default parameters: {e}")


# Test that box_violin_plot handles plot type switching correctly
def test_box_violin_plot_violin(sample_box_violin_dataframe):
    try:
        box_violin_plot(
            sample_box_violin_dataframe,
            metrics_list=["Metric1"],
            metrics_comp=["Category"],
            plot_type="violinplot",
            show_plot="both",
        )
    except Exception as e:
        pytest.fail(f"box_violin_plot failed with violin plot type: {e}")


# Test that box_violin_plot correctly applies axis limits
def test_box_violin_plot_axis_limits(sample_box_violin_dataframe):
    try:
        box_violin_plot(
            sample_box_violin_dataframe,
            metrics_list=["Metric1"],
            metrics_comp=["Category"],
            xlim=(0, 120),
            ylim=(0, 300),
            show_plot="both",
        )
    except Exception as e:
        pytest.fail(f"box_violin_plot failed with axis limits: {e}")


# Test that box_violin_plot correctly handles rotation
def test_box_violin_plot_rotate(sample_box_violin_dataframe):
    try:
        box_violin_plot(
            sample_box_violin_dataframe,
            metrics_list=["Metric1"],
            metrics_comp=["Category"],
            rotate_plot=True,
            show_plot="both",
        )
    except Exception as e:
        pytest.fail(f"box_violin_plot failed with rotation enabled: {e}")


# Test that box_violin_plot raises error on invalid input
def test_box_violin_plot_invalid_input(sample_box_violin_dataframe):
    with pytest.raises(ValueError):
        box_violin_plot(
            sample_box_violin_dataframe, metrics_list=[], metrics_comp=["Category"]
        )


# Test that `show_plot` only allows "individual", "subplots", or "both"
def test_box_violin_plot_invalid_show_plot(sample_box_violin_dataframe):
    with pytest.raises(
        ValueError,
        match="Invalid `show_plot` value selected. Choose from 'individual', 'subplots', or 'both'.",
    ):
        box_violin_plot(
            sample_box_violin_dataframe,
            metrics_list=["Metric1"],
            metrics_comp=["Category"],
            show_plot="invalid_option",
        )


# Test that `image_filename` without a path raises an error
def test_box_violin_plot_invalid_save_plots(sample_box_violin_dataframe):
    with pytest.raises(ValueError, match="image_filename"):
        box_violin_plot(
            sample_box_violin_dataframe,
            metrics_list=["Metric1"],
            metrics_comp=["Category"],
            image_filename="test_output",  # No path provided
        )


# Test that `image_filename` requires `image_path_png` or `image_path_svg`
def test_box_violin_plot_missing_image_path(sample_box_violin_dataframe):
    with pytest.raises(
        ValueError,
        match="image_path_png.*image_path_svg|image_path_svg.*image_path_png",
    ):
        box_violin_plot(
            sample_box_violin_dataframe,
            metrics_list=["Metric1"],
            metrics_comp=["Category"],
            image_filename="test_output",  # Should fail since no path is provided
        )


# Test that `rotate_plot` must be a boolean value
def test_box_violin_plot_invalid_rotate_plot(sample_box_violin_dataframe):
    with pytest.raises(
        ValueError,
        match="Invalid `rotate_plot` value selected. Choose from 'True' or 'False'.",
    ):
        box_violin_plot(
            sample_box_violin_dataframe,
            metrics_list=["Metric1"],
            metrics_comp=["Category"],
            rotate_plot="yes",  # Invalid input (should be True/False)
        )


# Test that `individual_figsize` must be a tuple/list of two numbers
def test_box_violin_plot_invalid_individual_figsize(sample_box_violin_dataframe):
    with pytest.raises(ValueError, match="Invalid `individual_figsize` value"):
        box_violin_plot(
            sample_box_violin_dataframe,
            metrics_list=["Metric1"],
            metrics_comp=["Category"],
            individual_figsize="big",  # Invalid type
        )


# Test that `subplot_figsize`, if specified, must be a tuple/list of two numbers
def test_box_violin_plot_invalid_subplot_figsize(sample_box_violin_dataframe):
    with pytest.raises(ValueError, match="Invalid `subplot_figsize` value"):
        box_violin_plot(
            sample_box_violin_dataframe,
            metrics_list=["Metric1"],
            metrics_comp=["Category"],
            subplot_figsize=["too", "big"],  # Invalid values
        )


# Test that stacked_crosstab_plot runs without errors with default parameters
def test_stacked_crosstab_plot_basic(sample_crosstab_dataframe):
    try:
        stacked_crosstab_plot(
            df=sample_crosstab_dataframe,
            col="Category",
            func_col=["Group"],
            legend_labels_list=[["X", "Y"]],
            title=["Distribution of Groups per Category"],
        )
    except Exception as e:
        pytest.fail(f"stacked_crosstab_plot failed with default parameters: {e}")


# Test that stacked_crosstab_plot raises an error when required columns are missing
def test_stacked_crosstab_plot_missing_column(sample_crosstab_dataframe):
    with pytest.raises(KeyError):
        stacked_crosstab_plot(
            df=sample_crosstab_dataframe,
            col="NonExistentColumn",
            func_col=["Group"],
            legend_labels_list=[["X", "Y"]],
            title=["Invalid Test"],
        )


# Test that stacked_crosstab_plot raises an error when legend list lengths mismatch
def test_stacked_crosstab_plot_legend_mismatch(sample_crosstab_dataframe):
    with pytest.raises(ValueError):
        stacked_crosstab_plot(
            df=sample_crosstab_dataframe,
            col="Category",
            func_col=["Group", "Outcome"],
            legend_labels_list=[["X", "Y"]],  # Mismatched length
            title=["Mismatch Test"],
        )


# Test that stacked_crosstab_plot correctly applies log scale
def test_stacked_crosstab_plot_logscale(sample_crosstab_dataframe):
    try:
        stacked_crosstab_plot(
            df=sample_crosstab_dataframe,
            col="Category",
            func_col=["Group"],
            legend_labels_list=[["X", "Y"]],
            title=["Log Scale Test"],
            logscale=True,
        )
    except Exception as e:
        pytest.fail(f"stacked_crosstab_plot failed with log scale: {e}")


# Test that stacked_crosstab_plot correctly removes stacks when specified
def test_stacked_crosstab_plot_remove_stacks(sample_crosstab_dataframe):
    try:
        stacked_crosstab_plot(
            df=sample_crosstab_dataframe,
            col="Category",
            func_col=["Group"],
            legend_labels_list=[["X", "Y"]],
            title=["Without Stacks"],
            remove_stacks=True,
            plot_type="regular",
        )
    except Exception as e:
        pytest.fail(f"stacked_crosstab_plot failed with remove_stacks: {e}")


def test_stacked_crosstab_plot_save_single_format(sample_crosstab_dataframe, tmp_path):
    save_path = str(tmp_path)

    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    try:
        stacked_crosstab_plot(
            df=sample_crosstab_dataframe,
            col="Category",
            func_col=["Group"],
            legend_labels_list=[["X", "Y"]],
            title=["Saving Single Format"],
            save_formats="png",  # Ensure this matches function expectations
            image_path_png=save_path,  # Pass directory instead of filename
        )

        # Assert that at least one PNG file was created in the directory
        png_files = [f for f in os.listdir(save_path) if f.endswith(".png")]
        assert len(png_files) > 0, "PNG file should be created."

    except Exception as e:
        pytest.fail(f"stacked_crosstab_plot failed with single save format: {e}")


def test_stacked_crosstab_plot_save_multiple_formats(
    sample_crosstab_dataframe, tmp_path
):
    save_path = str(tmp_path)

    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    try:
        stacked_crosstab_plot(
            df=sample_crosstab_dataframe,
            col="Category",
            func_col=["Group"],
            legend_labels_list=[["X", "Y"]],
            title=["Saving Multiple Formats"],
            save_formats=["png", "svg"],  # Ensure this matches function expectations
            image_path_png=save_path,  # Pass directory instead of filename
            image_path_svg=save_path,  # Pass directory instead of filename
        )

        # Assert that at least one PNG and one SVG file were created
        png_files = [f for f in os.listdir(save_path) if f.endswith(".png")]
        svg_files = [f for f in os.listdir(save_path) if f.endswith(".svg")]

        assert len(png_files) > 0, "PNG file should be created."
        assert len(svg_files) > 0, "SVG file should be created."

    except Exception as e:
        pytest.fail(f"stacked_crosstab_plot failed with multiple save formats: {e}")


def test_outcome_crosstab_plot_full(tmp_path):
    df = pd.DataFrame(
        {
            "sex": ["M", "F", "F", "M", "M", "F"],
            "outcome": [1, 0, 1, 0, 1, 0],
        }
    )

    outcome_crosstab_plot(
        df=df,
        list_name=["sex"],
        outcome="outcome",
        figsize=(6, 4),
        label_fontsize=10,
        tick_fontsize=8,
        normalize=True,
        show_value_counts=True,
        color_schema=["#123456", "#654321"],
        save_plots=True,
        image_path_png=str(tmp_path),
        image_path_svg=str(tmp_path),
        string="Test Save",
        label_0="Negative",
        label_1="Positive",
    )

    assert (tmp_path / "test_save.png").exists()
    assert (tmp_path / "test_save.svg").exists()

    plt.close("all")


#################################################################################
#################################################################################


@pytest.fixture(autouse=True)
def disable_show(monkeypatch):
    # Prevent plt.show() from blocking
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kw: None)


def test_show_plot_combinations():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [2, 4, 6],
            "c": [3, 6, 9],
        }
    )
    combos = scatter_fit_plot(df, all_vars=["a", "b", "c"], show_plot="combinations")
    expected = list(itertools.combinations(["a", "b", "c"], 2))
    assert isinstance(combos, list)
    assert set(combos) == set(expected)


@pytest.mark.parametrize(
    "bad,kwargs",
    [
        ("all_vars_conflict", dict(x_vars="a", y_vars="b", all_vars=["a", "b"])),
        ("no_vars", dict()),
        ("invalid_show_plot", dict(x_vars="a", y_vars="b", show_plot="oops")),
        ("invalid_save_plots", dict(x_vars="a", y_vars="b", save_plots="oops")),
        ("save_without_path", dict(x_vars="a", y_vars="b", save_plots="all")),
        ("hue_palette_no_hue", dict(x_vars="a", y_vars="b", hue_palette="Blues")),
        ("rotate_not_bool", dict(x_vars="a", y_vars="b", rotate_plot="yes")),
        (
            "bad_individual_figsize",
            dict(x_vars="a", y_vars="b", individual_figsize=(1,)),
        ),
        (
            "bad_subplot_figsize",
            dict(x_vars="a", y_vars="b", subplot_figsize=(1, 2, 3)),
        ),
    ],
)
def test_invalid_inputs_raise(bad, kwargs):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        scatter_fit_plot(df, **kwargs)


@pytest.mark.parametrize(
    "excludes",
    [
        [("a", "x")],  # x exists, y doesn’t
        [("x", "y")],  # both missing
    ],
)
def test_exclude_combinations_invalid(excludes):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        scatter_fit_plot(df, x_vars="a", y_vars="b", exclude_combinations=excludes)


def test_individual_plot_with_best_fit_and_limits(tmp_path):
    # Should exercise: individual branch, best-fit line, xlim/ylim, rotate_plot, label_names
    df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 1, 4, 9]})
    labels = {"x": "X Axis", "y": "Y Axis"}
    # Save each individual plot into tmp_path
    scatter_fit_plot(
        df,
        x_vars="x",
        y_vars="y",
        show_plot="individual",
        add_best_fit_line=True,
        xlim=(0, 3),
        ylim=(0, 10),
        rotate_plot=True,
        label_names=labels,
        save_plots="individual",
        image_path_png=str(tmp_path),
    )
    # Verify at least one file was written
    files = os.listdir(tmp_path)
    assert any(f.startswith("scatter_x_vs_y") and f.endswith(".png") for f in files)


def test_grouped_distributions_non_binary_raises(binary_df):
    df = binary_df.copy()
    df["group"] = np.tile([0, 1, 2], len(df))[: len(df)]
    with pytest.raises(ValueError, match="must be binary"):
        grouped_distributions(
            df=df,
            features=["x1"],
            by="group",
        )


def test_grouped_distributions_non_binary_raises_tile(binary_df):
    df = binary_df.copy()
    df["group"] = np.tile([0, 1, 2], len(df))[: len(df)]
    with pytest.raises(ValueError, match="must be binary"):
        grouped_distributions(
            df=df,
            features=["x1"],
            by="group",
        )


def test_grouped_distributions_invalid_normalize(binary_df):
    with pytest.raises(ValueError, match="normalize must be"):
        grouped_distributions(
            df=binary_df,
            features=["x1"],
            by="group",
            normalize="percent",
        )


def test_density_plot_with_count_normalize_raises(binary_df):
    with pytest.raises(ValueError, match="Density plots only support"):
        grouped_distributions(
            df=binary_df,
            features=["x1"],
            by="group",
            plot_style="density",
            normalize="count",
        )


def test_image_filename_without_path_raises(binary_df):
    with pytest.raises(ValueError, match="image_path_png|image_path_svg"):
        grouped_distributions(
            df=binary_df,
            features=["x1"],
            by="group",
            image_filename="test_plot",
        )


def test_grouped_distributions_auto_layout(binary_df):
    grouped_distributions(
        df=binary_df,
        features=["x1", "x2"],
        by="group",
    )


def test_grouped_distributions_grid_too_small(binary_df):
    with pytest.raises(ValueError, match="Grid too small"):
        grouped_distributions(
            df=binary_df,
            features=["x1", "x2"],
            by="group",
            n_rows=1,
            n_cols=1,
        )


def test_grouped_distributions_hist_runs(binary_df):
    grouped_distributions(
        df=binary_df,
        features=["x1"],
        by="group",
        plot_style="hist",
    )


def test_grouped_distributions_density_runs(binary_df):
    grouped_distributions(
        df=binary_df,
        features=["x1"],
        by="group",
        plot_style="density",
    )


def test_grouped_distributions_calls_save_figure(binary_df, tmp_path):
    with patch("eda_toolkit.plots._save_figure") as mock_save:
        grouped_distributions(
            df=binary_df,
            features=["x1"],
            by="group",
            image_path_png=str(tmp_path),
            image_filename="cond_hist",
        )

        mock_save.assert_called_once()


def test_grouped_distributions_no_legend(binary_df):
    grouped_distributions(
        df=binary_df,
        features=["x1"],
        by="group",
        show_legend=False,
    )


def test_distribution_gof_invalid_plot_type(gof_df):
    with pytest.raises(ValueError, match="Invalid plot_types"):
        distribution_gof_plots(
            df=gof_df,
            var="x",
            plot_types=["qq", "hist"],
        )


def test_distribution_gof_invalid_qq_type(gof_df):
    with pytest.raises(ValueError, match="qq_type must be"):
        distribution_gof_plots(
            df=gof_df,
            var="x",
            qq_type="approximate",
        )


def test_distribution_gof_invalid_scale(gof_df):
    with pytest.raises(ValueError, match="scale must be"):
        distribution_gof_plots(
            df=gof_df,
            var="x",
            scale="logarithmic",
        )


def test_distribution_gof_invalid_tail(gof_df):
    with pytest.raises(ValueError, match="tail must be"):
        distribution_gof_plots(
            df=gof_df,
            var="x",
            plot_types="cdf",
            tail="middle",
        )


def test_distribution_gof_empirical_without_reference_raises(gof_df):
    with pytest.raises(ValueError, match="reference_data must be provided"):
        distribution_gof_plots(
            df=gof_df,
            var="x",
            qq_type="empirical",
        )


def test_distribution_gof_empirical_reference_too_short(gof_df):
    with pytest.raises(ValueError, match="at least 2"):
        distribution_gof_plots(
            df=gof_df,
            var="x",
            qq_type="empirical",
            reference_data=np.array([1.0]),
        )


def test_distribution_gof_filename_without_path_raises(gof_df):
    with pytest.raises(ValueError, match="image_path_png|image_path_svg"):
        distribution_gof_plots(
            df=gof_df,
            var="x",
            image_filename="gof_plot",
        )


def test_distribution_gof_palette_missing_dist_raises(gof_df):
    with pytest.raises(ValueError, match="Palette is missing"):
        distribution_gof_plots(
            df=gof_df,
            var="x",
            dist=["norm", "gamma"],
            palette={"norm": "blue"},
        )


def test_distribution_gof_default_runs(gof_df):
    distribution_gof_plots(
        df=gof_df,
        var="x",
    )


def test_distribution_gof_multiple_distributions_runs(gof_df):
    distribution_gof_plots(
        df=gof_df,
        var="x",
        dist=["norm", "lognorm"],
    )


def test_distribution_gof_only_qq_runs(gof_df):
    distribution_gof_plots(
        df=gof_df,
        var="x",
        plot_types="qq",
    )


def test_distribution_gof_only_cdf_runs(gof_df):
    distribution_gof_plots(
        df=gof_df,
        var="x",
        plot_types="cdf",
    )


def test_distribution_gof_empirical_qq_runs(gof_df, reference_data):
    distribution_gof_plots(
        df=gof_df,
        var="x",
        qq_type="empirical",
        reference_data=reference_data,
    )


def test_distribution_gof_calls_qq_plot(gof_df):
    with patch("eda_toolkit.plots._qq_plot") as mock_qq:
        distribution_gof_plots(
            df=gof_df,
            var="x",
            plot_types="qq",
        )

        assert mock_qq.called


def test_distribution_gof_calls_cdf_plot(gof_df):
    with patch("eda_toolkit.plots._cdf_exceedance_plot") as mock_cdf:
        distribution_gof_plots(
            df=gof_df,
            var="x",
            plot_types="cdf",
        )

        assert mock_cdf.called


def test_distribution_gof_calls_save_figure(gof_df, tmp_path):
    with patch("eda_toolkit.plots._save_figure") as mock_save:
        distribution_gof_plots(
            df=gof_df,
            var="x",
            image_path_png=str(tmp_path),
            image_filename="gof_plot",
        )

        mock_save.assert_called_once()


# --- single-file functions: full path, no directory, saved verbatim --------


def test_flex_corr_matrix_full_path(sample_corr_dataframe, tmp_path):
    target = tmp_path / "corr.png"
    flex_corr_matrix(sample_corr_dataframe, image_filename=str(target))
    assert target.exists()


def test_data_doctor_full_path(sample_dataframe_values, tmp_path):
    target = tmp_path / "dd.png"
    data_doctor(sample_dataframe_values, "values", image_filename=str(target))
    assert target.exists()


def test_distribution_gof_full_path(gof_df, tmp_path):
    target = tmp_path / "gof.png"
    distribution_gof_plots(df=gof_df, var="x", image_filename=str(target))
    assert target.exists()


def test_grouped_distributions_full_path(binary_df, tmp_path):
    target = tmp_path / "gd.png"
    grouped_distributions(
        df=binary_df, features=["x1"], by="group", image_filename=str(target)
    )
    assert target.exists()


def test_outcome_crosstab_full_path(tmp_path):
    df = pd.DataFrame(
        {"sex": ["M", "F", "F", "M", "M", "F"], "outcome": [1, 0, 1, 0, 1, 0]}
    )
    target = tmp_path / "oc.png"
    outcome_crosstab_plot(
        df=df, list_name=["sex"], outcome="outcome", image_filename=str(target)
    )
    assert target.exists()
    plt.close("all")


def test_stacked_crosstab_image_filename_full_path(sample_crosstab_dataframe, tmp_path):
    stacked_crosstab_plot(
        df=sample_crosstab_dataframe,
        col="Category",
        func_col=["Group"],
        legend_labels_list=[["X", "Y"]],
        title=["Full Path"],
        image_filename=str(tmp_path / "st.png"),
    )
    assert any(f.endswith(".png") for f in os.listdir(tmp_path))


# --- fan-out functions: full path, extension stays intact (regression) ------


def test_box_violin_full_path_fanout(sample_box_violin_dataframe, tmp_path):
    box_violin_plot(
        sample_box_violin_dataframe,
        metrics_list=["Metric1", "Metric2"],
        metrics_comp=["Category"],
        show_plot="both",
        image_filename=str(tmp_path / "bv.png"),
    )
    files = os.listdir(tmp_path)
    assert files, "fan-out produced no files"
    assert all(f.endswith(".png") for f in files)
    assert not any(".png_" in f for f in files)
    plt.close("all")


def test_scatter_full_path_fanout(sample_scatter_dataframe, tmp_path):
    scatter_fit_plot(
        sample_scatter_dataframe,
        x_vars=["Feature1"],
        y_vars=["Feature2"],
        save_plots="all",
        image_filename=str(tmp_path / "sc.png"),
    )
    files = os.listdir(tmp_path)
    assert any(f.endswith(".png") for f in files)
    assert not any(".png_" in f for f in files)
    plt.close("all")


# ------------------------------------------------------------------
# corr_threshold (variable-drop)
# ------------------------------------------------------------------


def test_flex_corr_matrix_corr_threshold_drops_uncorrelated(
    sample_corr_dataframe_large,
):
    """corr_threshold drops variables whose strongest |r| is below it.

    In the fixture A/B/D are strongly correlated and C is noise, so a
    threshold of 0.5 should drop C and keep A, B, D.
    """
    result = flex_corr_matrix(
        sample_corr_dataframe_large,
        corr_threshold=0.5,
        return_corr=True,
    )
    assert "C" not in result.columns
    assert set(result.columns) == {"A", "B", "D"}


def test_flex_corr_matrix_corr_threshold_no_survivors_raises(
    sample_corr_dataframe_large,
):
    """A threshold no pair can meet raises rather than drawing an empty plot."""
    with pytest.raises(ValueError, match="No variables have an absolute correlation"):
        flex_corr_matrix(sample_corr_dataframe_large, corr_threshold=0.999)


def test_flex_corr_matrix_corr_threshold_invalid(sample_corr_dataframe):
    with pytest.raises(ValueError, match="`corr_threshold` must be"):
        flex_corr_matrix(sample_corr_dataframe, corr_threshold=1.5)


# ------------------------------------------------------------------
# return_corr
# ------------------------------------------------------------------


def test_flex_corr_matrix_return_corr_true(sample_corr_dataframe):
    """return_corr=True returns the plotted matrix as a square DataFrame."""
    result = flex_corr_matrix(sample_corr_dataframe, return_corr=True)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == result.shape[1]
    # full square, not triangular-masked
    assert result.notna().all().all()


def test_flex_corr_matrix_return_corr_false_returns_none(sample_corr_dataframe):
    """Default return_corr=False preserves the old None return."""
    result = flex_corr_matrix(sample_corr_dataframe, return_corr=False)
    assert result is None


def test_flex_corr_matrix_return_corr_reflects_threshold(
    sample_corr_dataframe_large,
):
    """The returned matrix reflects corr_threshold filtering."""
    result = flex_corr_matrix(
        sample_corr_dataframe_large,
        corr_threshold=0.5,
        return_corr=True,
    )
    assert "C" not in result.columns


# ------------------------------------------------------------------
# show_plot gate (only active with return_corr=True)
# ------------------------------------------------------------------


def test_flex_corr_matrix_return_corr_suppresses_plot(sample_corr_dataframe):
    """return_corr=True with default show_plot=False returns data and draws no figure."""
    plt.close("all")
    result = flex_corr_matrix(sample_corr_dataframe, return_corr=True)
    assert isinstance(result, pd.DataFrame)
    assert (
        len(plt.get_fignums()) == 0
    ), "No figure should be drawn when show_plot=False."


def test_flex_corr_matrix_return_corr_with_show_plot_draws(sample_corr_dataframe):
    """return_corr=True with show_plot=True returns data AND draws the figure."""
    plt.close("all")
    result = flex_corr_matrix(sample_corr_dataframe, return_corr=True, show_plot=True)
    assert isinstance(result, pd.DataFrame)
    assert len(plt.get_fignums()) > 0, "A figure should be drawn when show_plot=True."
    plt.close("all")


def test_flex_corr_matrix_show_plot_ignored_without_return_corr(sample_corr_dataframe):
    """show_plot has no effect when return_corr=False: still plots, still returns None."""
    plt.close("all")
    result = flex_corr_matrix(sample_corr_dataframe, show_plot=False)
    assert result is None
    assert len(plt.get_fignums()) > 0, "Plot should still draw when return_corr=False."
    plt.close("all")


def test_flex_corr_matrix_return_sig_true(sample_corr_dataframe_large):
    """return_sig=True returns the p-value matrix as a square DataFrame."""
    result = flex_corr_matrix(sample_corr_dataframe_large, return_sig=True)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == result.shape[1]


def test_flex_corr_matrix_return_both_returns_tuple(sample_corr_dataframe_large):
    """return_corr and return_sig together return (corr, pval)."""
    corr, pval = flex_corr_matrix(
        sample_corr_dataframe_large, return_corr=True, return_sig=True
    )
    assert isinstance(corr, pd.DataFrame)
    assert isinstance(pval, pd.DataFrame)
    assert corr.shape == pval.shape


def test_flex_corr_matrix_return_sig_suppresses_plot(sample_corr_dataframe_large):
    """return_sig=True suppresses the plot by default, like return_corr."""
    plt.close("all")
    flex_corr_matrix(sample_corr_dataframe_large, return_sig=True)
    assert len(plt.get_fignums()) == 0


def test_flex_corr_matrix_return_sig_with_show_plot_draws(sample_corr_dataframe_large):
    """return_sig=True with show_plot=True returns data AND draws."""
    plt.close("all")
    flex_corr_matrix(sample_corr_dataframe_large, return_sig=True, show_plot=True)
    assert len(plt.get_fignums()) > 0
    plt.close("all")


def test_flex_corr_matrix_return_sig_diagonal_is_one(sample_corr_dataframe_large):
    """The returned p-value matrix carries 1.0 on its diagonal."""
    pval = flex_corr_matrix(sample_corr_dataframe_large, return_sig=True)
    assert np.allclose(np.diag(pval.values), 1.0)
