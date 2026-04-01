"""Tests for CellArraySE slicing and subsetting functionality."""

import pytest
import numpy as np
from summarizedexperiment import SummarizedExperiment

__author__ = "chanjd"
__copyright__ = "chanjd"
__license__ = "MIT"


class TestGetItemBasic:
    """Test __getitem__ bracket notation."""

    def test_slice_both_dimensions(self, sample_cellarr_se):
        result = sample_cellarr_se[0:3, 0:2]
        assert isinstance(result, SummarizedExperiment)
        assert result.shape == (3, 2)

    def test_single_int_indices(self, sample_cellarr_se):
        result = sample_cellarr_se[0, 0]
        assert isinstance(result, SummarizedExperiment)
        assert result.shape == (1, 1)

    def test_negative_indices(self, sample_cellarr_se):
        result = sample_cellarr_se[-1, -1]
        assert isinstance(result, SummarizedExperiment)
        assert result.shape == (1, 1)

    def test_list_of_ints(self, sample_cellarr_se):
        result = sample_cellarr_se[[0, 2, 4], [1, 3]]
        assert isinstance(result, SummarizedExperiment)
        assert result.shape == (3, 2)

    def test_mixed_slice_and_list(self, sample_cellarr_se):
        result = sample_cellarr_se[0:5, [0, 2, 4]]
        assert isinstance(result, SummarizedExperiment)
        assert result.shape == (5, 3)

    def test_full_slice(self, sample_cellarr_se):
        result = sample_cellarr_se[:, :]
        assert result.shape == sample_cellarr_se.shape


class TestGetItemErrors:
    """Test __getitem__ error handling."""

    def test_single_key_raises(self, sample_cellarr_se):
        with pytest.raises(ValueError, match="2-dimensional tuple"):
            sample_cellarr_se[0:5]

    def test_three_keys_raises(self, sample_cellarr_se):
        with pytest.raises(ValueError, match="2-dimensional tuple"):
            sample_cellarr_se[0:5, 0:3, 0:2]

    def test_out_of_bounds_raises(self, sample_cellarr_se):
        with pytest.raises(IndexError, match="out of bounds"):
            sample_cellarr_se[100, 0]

    def test_invalid_key_type_raises(self, sample_cellarr_se):
        with pytest.raises(TypeError):
            sample_cellarr_se[0.5, 0]

    def test_slice_with_step_raises(self, sample_cellarr_se):
        """Step rejection is enforced by CellArraySE before reaching TileDB."""
        with pytest.raises(IndexError, match="[Ss]tep"):
            sample_cellarr_se[0:10:2, 0:5]


class TestSliceMethod:
    """Test the slice() method with full parameter support."""

    def test_slice_with_subsets(self, sample_cellarr_se):
        result = sample_cellarr_se.slice(row_subset=slice(0, 5), col_subset=slice(0, 3))
        assert isinstance(result, SummarizedExperiment)
        assert result.shape == (5, 3)

    def test_slice_none_returns_all(self, sample_cellarr_se):
        result = sample_cellarr_se.slice()
        assert result.shape == sample_cellarr_se.shape

    def test_slice_select_assays(self, sample_cellarr_se):
        result = sample_cellarr_se.slice(row_subset=slice(0, 5), col_subset=slice(0, 3), assays=["counts"])
        assert "counts" in result.assay_names
        assert "tpm" not in result.assay_names

    def test_slice_select_row_columns(self, sample_cellarr_se):
        result = sample_cellarr_se.slice(
            row_subset=slice(0, 5),
            col_subset=slice(0, 3),
            row_columns=["gene_id"],
        )
        assert list(result.row_data.columns) == ["gene_id"]

    def test_slice_select_col_columns(self, sample_cellarr_se):
        result = sample_cellarr_se.slice(
            row_subset=slice(0, 5),
            col_subset=slice(0, 3),
            col_columns=["sample_id", "tissue"],
        )
        assert set(result.column_data.columns) == {"sample_id", "tissue"}

    def test_slice_invalid_assay_raises(self, sample_cellarr_se):
        with pytest.raises(KeyError, match="not found"):
            sample_cellarr_se.slice(assays=["nonexistent"])


class TestSliceWithQuery:
    """Test slice() with TileDB query strings.

    Query filtering is only supported on string-indexed (sparse) CellArrayFrames,
    so these tests use sample_cellarr_se_named rather than sample_cellarr_se.
    """

    def test_row_query_shape(self, sample_cellarr_se_named):
        result = sample_cellarr_se_named.slice(row_query="gene_type == 'protein'")
        assert isinstance(result, SummarizedExperiment)
        assert result.shape == (7, 5)

    def test_row_query_metadata_filtered(self, sample_cellarr_se_named):
        result = sample_cellarr_se_named.slice(row_query="gene_type == 'protein'")
        assert all(v == "protein" for v in result.row_data["gene_type"])

    def test_col_query_shape(self, sample_cellarr_se_named):
        result = sample_cellarr_se_named.slice(col_query="tissue == 'liver'")
        assert result.shape == (10, 2)

    def test_col_query_metadata_filtered(self, sample_cellarr_se_named):
        result = sample_cellarr_se_named.slice(col_query="tissue == 'liver'")
        assert all(v == "liver" for v in result.column_data["tissue"])

    def test_both_queries(self, sample_cellarr_se_named):
        result = sample_cellarr_se_named.slice(
            row_query="gene_type == 'protein'",
            col_query="treatment == 'control'",
        )
        assert result.shape == (7, 3)

    def test_query_with_col_columns(self, sample_cellarr_se_named):
        result = sample_cellarr_se_named.slice(
            row_query="gene_type == 'protein'",
            col_subset=slice(0, 3),
            col_columns=["tissue"],
        )
        assert result.shape[0] == 7
        assert list(result.column_data.columns) == ["tissue"]

    def test_query_and_subset_mutually_exclusive(self, sample_cellarr_se):
        with pytest.raises(ValueError, match="Cannot specify both"):
            sample_cellarr_se.slice(row_subset=slice(0, 5), row_query="gene_type == 'protein'")


class TestSliceDataIntegrity:
    """Test that sliced data matches the correct backing values."""

    def test_assay_data_matches_indices(self, sample_cellarr_se):
        full = sample_cellarr_se.slice()
        full_counts = full.assays["counts"]

        subset = sample_cellarr_se[0:3, 0:2]
        np.testing.assert_array_equal(subset.assays["counts"], full_counts[0:3, 0:2])

    def test_row_metadata_matches_slice(self, sample_cellarr_se):
        """Row metadata must correspond to the sliced rows, not the full frame.
        Expected values come from the fixture in conftest: genes 0-2 are TP53, BRCA1, EGFR.
        """
        subset = sample_cellarr_se[0:3, 0:5]
        assert len(subset.row_data) == 3
        assert list(subset.row_data["gene_name"]) == ["TP53", "BRCA1", "EGFR"]

    def test_col_metadata_matches_slice(self, sample_cellarr_se):
        """Column metadata must correspond to the sliced columns, not the full frame.
        Expected values come from the fixture in conftest: samples 0-1 are liver, kidney.
        """
        subset = sample_cellarr_se[0:10, 0:2]
        assert len(subset.column_data) == 2
        assert list(subset.column_data["tissue"]) == ["liver", "kidney"]

    def test_multiple_assays_sliced_consistently(self, sample_cellarr_se):
        subset = sample_cellarr_se[0:3, 0:2]
        assert subset.assays["counts"].shape == (3, 2)
        assert subset.assays["tpm"].shape == (3, 2)

    def test_sparse_assay_slicing(self, sample_cellarr_se_sparse):
        result = sample_cellarr_se_sparse[0:3, 0:2]
        assert result.shape == (3, 2)
        assert "counts" in result.assay_names

    def test_mixed_assay_slicing(self, sample_cellarr_se_mixed):
        result = sample_cellarr_se_mixed[0:5, 0:3]
        assert result.shape == (5, 3)
        assert "dense_counts" in result.assay_names
        assert "sparse_counts" in result.assay_names


class TestSlicingEdgeCases:
    """Test edge cases in slicing behaviour."""

    def test_negative_index_single(self, sample_cellarr_se):
        """Negative indices must wrap from the end, matching explicit positive equivalents.
        Cross-checked against the positive-index form rather than raw backing data
        since the backing array isn't directly addressable after materialisation.
        """
        last_negative = sample_cellarr_se[-1, -1]
        last_positive = sample_cellarr_se[9, 4]
        np.testing.assert_array_equal(last_negative.assays["counts"], last_positive.assays["counts"])

    def test_negative_index_in_list(self, sample_cellarr_se):
        result = sample_cellarr_se[[-1, -2], [-1]]
        assert result.shape == (2, 1)

    def test_slice_start_only(self, sample_cellarr_se):
        result = sample_cellarr_se[5:, 0:5]
        assert result.shape == (5, 5)

    def test_slice_stop_only(self, sample_cellarr_se):
        result = sample_cellarr_se[:3, :2]
        assert result.shape == (3, 2)

    def test_name_not_found_raises(self, sample_cellarr_se_named):
        with pytest.raises(KeyError, match="not found"):
            sample_cellarr_se_named["nonexistent_gene", 0:5]

    def test_name_not_found_in_list_raises(self, sample_cellarr_se_named):
        with pytest.raises(KeyError, match="not found"):
            sample_cellarr_se_named[["ENSG00000"], ["nonexistent_sample"]]

    def test_index_out_of_bounds_positive(self, sample_cellarr_se):
        with pytest.raises(IndexError, match="out of bounds"):
            sample_cellarr_se[100, 0]

    def test_index_out_of_bounds_negative(self, sample_cellarr_se):
        with pytest.raises(IndexError, match="out of bounds"):
            sample_cellarr_se[-100, 0]

    def test_index_out_of_bounds_in_list(self, sample_cellarr_se):
        with pytest.raises(IndexError, match="out of bounds"):
            sample_cellarr_se[[0, 100], [0]]

    def test_mixed_type_list_raises(self, sample_cellarr_se):
        with pytest.raises(TypeError, match="same type"):
            sample_cellarr_se[[0, "gene1"], [0]]

    def test_unsupported_key_type_raises(self, sample_cellarr_se):
        with pytest.raises(TypeError):
            sample_cellarr_se[0.5, 0]
