"""Tests for CellArraySE core functionality."""

import pytest
import numpy as np
from cellarr_se import CellArraySE

__author__ = "chanjd"
__copyright__ = "chanjd"
__license__ = "MIT"


class TestCellArraySEInit:
    """Test CellArraySE initialization and validation."""

    def test_init_success(self, sample_cellarr_se):
        assert sample_cellarr_se.shape == (10, 5)

    def test_init_empty_assays_raises(self, sample_row_data, sample_col_data):
        with pytest.raises(ValueError, match="cannot be empty"):
            CellArraySE(assays={}, row_data=sample_row_data, col_data=sample_col_data)

    def test_init_invalid_assays_type_raises(self, sample_row_data, sample_col_data):
        with pytest.raises(TypeError, match="must be a dictionary"):
            CellArraySE(assays=[], row_data=sample_row_data, col_data=sample_col_data)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"assays": {"x": "not_a_cellarray"}}, "must be a CellArray"),
            ({"row_data": "not_a_frame"}, "row_data must be a CellArrayFrame"),
            ({"col_data": "not_a_frame"}, "col_data must be a CellArrayFrame"),
        ],
    )
    def test_init_invalid_component_type_raises(
        self, kwargs, match, sample_counts_assay, sample_row_data, sample_col_data
    ):
        base = {"assays": {"counts": sample_counts_assay}, "row_data": sample_row_data, "col_data": sample_col_data}
        base.update(kwargs)
        with pytest.raises(TypeError, match=match):
            CellArraySE(**base)

    def test_init_mismatched_shape_raises(self, temp_dir, sample_row_data, sample_col_data, sample_counts_assay):
        """All assays must share the same shape; a single outlier should fail validation."""
        import os
        import tiledb
        from cellarr_array import DenseCellArray

        uri = os.path.join(temp_dir, "wrong_shape")
        dom = tiledb.Domain(
            tiledb.Dim(name="rows", domain=(0, 4), tile=5, dtype=np.int32),
            tiledb.Dim(name="cols", domain=(0, 4), tile=5, dtype=np.int32),
        )
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=False,
            attrs=[tiledb.Attr(name="data", dtype=np.float64)],
        )
        tiledb.Array.create(uri, schema)
        with tiledb.open(uri, "w") as A:
            A[:] = np.zeros((5, 5))
        wrong_array = DenseCellArray(uri)

        with pytest.raises(ValueError, match="shape"):
            CellArraySE(
                assays={"counts": sample_counts_assay, "wrong": wrong_array},
                row_data=sample_row_data,
                col_data=sample_col_data,
            )


class TestCellArraySEProperties:
    """Test CellArraySE property accessors."""

    def test_shape(self, sample_cellarr_se):
        assert sample_cellarr_se.shape == (10, 5)

    def test_dims(self, sample_cellarr_se):
        assert sample_cellarr_se.dims == (10, 5)

    def test_assay_names(self, sample_cellarr_se):
        assert set(sample_cellarr_se.assay_names) == {"counts", "tpm"}

    def test_row_names_content(self, sample_cellarr_se):
        """Dense frames have no explicit index, so row_names is a 0-based RangeIndex."""
        row_names = sample_cellarr_se.row_names
        assert len(row_names) == 10
        assert list(row_names) == list(range(10))

    def test_col_names_content(self, sample_cellarr_se):
        """Dense frames have no explicit index, so col_names is a 0-based RangeIndex."""
        col_names = sample_cellarr_se.col_names
        assert len(col_names) == 5
        assert list(col_names) == list(range(5))

    def test_row_names_string_indexed(self, sample_cellarr_se_named):
        row_names = sample_cellarr_se_named.row_names
        assert len(row_names) == 10
        assert row_names[0] == "ENSG00000"
        assert row_names[-1] == "ENSG00009"

    def test_col_names_string_indexed(self, sample_cellarr_se_named):
        col_names = sample_cellarr_se_named.col_names
        assert len(col_names) == 5
        assert col_names[0] == "SAMPLE_000"
        assert col_names[-1] == "SAMPLE_004"

    def test_row_columns(self, sample_cellarr_se):
        assert set(sample_cellarr_se.row_columns) == {"gene_id", "gene_name", "gene_type"}

    def test_col_columns(self, sample_cellarr_se):
        assert set(sample_cellarr_se.col_columns) == {"sample_id", "tissue", "treatment"}


class TestCellArraySERepr:
    """Test string representation and display."""

    def test_repr_format(self, sample_cellarr_se):
        assert repr(sample_cellarr_se) == "<CellArraySE: 10x5 | counts, tpm>"

    def test_show_runs(self, sample_cellarr_se, capsys):
        sample_cellarr_se.show()
        out = capsys.readouterr().out
        assert "CellArraySE" in out
        assert "10" in out
        assert "5" in out


class TestCellArraySEAssayIntrospection:
    """Test assay introspection methods."""

    def test_get_assay_type_returns_dtype(self, sample_cellarr_se):
        assert sample_cellarr_se.get_assay_type("counts") == np.float64

    def test_get_assay_type_invalid_name_raises(self, sample_cellarr_se):
        with pytest.raises(KeyError, match="not found"):
            sample_cellarr_se.get_assay_type("nonexistent")

    def test_get_assay_type_sparse(self, sample_cellarr_se_sparse):
        assert sample_cellarr_se_sparse.get_assay_type("counts") == np.float64

    def test_is_sparse_true(self, sample_cellarr_se_sparse):
        assert sample_cellarr_se_sparse.is_sparse("counts") is True

    def test_is_sparse_false(self, sample_cellarr_se):
        assert sample_cellarr_se.is_sparse("counts") is False

    def test_is_sparse_unknown_assay_raises(self, sample_cellarr_se):
        with pytest.raises(KeyError, match="not found"):
            sample_cellarr_se.is_sparse("nonexistent")

    def test_mixed_dense_sparse(self, sample_cellarr_se_mixed):
        assert sample_cellarr_se_mixed.is_sparse("sparse_counts") is True
        assert sample_cellarr_se_mixed.is_sparse("dense_counts") is False


class TestNamedFrames:
    """Test CellArraySE with string-indexed CellArrayFrames."""

    def test_init(self, sample_cellarr_se_named):
        assert sample_cellarr_se_named.shape == (10, 5)

    def test_slice_by_name_list(self, sample_cellarr_se_named):
        result = sample_cellarr_se_named[["ENSG00000", "ENSG00001"], ["SAMPLE_000", "SAMPLE_001"]]
        assert result.shape == (2, 2)
        assert list(result.row_names) == ["ENSG00000", "ENSG00001"]
        assert list(result.column_names) == ["SAMPLE_000", "SAMPLE_001"]

    def test_row_query_filters_metadata(self, sample_cellarr_se_named):
        result = sample_cellarr_se_named.slice(row_query="gene_type == 'protein'")
        assert result.shape[0] == 7
        # Verify the filter actually worked — all returned rows must satisfy it
        assert all(v == "protein" for v in result.row_data["gene_type"])

    def test_col_query_filters_metadata(self, sample_cellarr_se_named):
        result = sample_cellarr_se_named.slice(col_query="tissue == 'liver'")
        assert result.shape[1] == 2
        assert all(v == "liver" for v in result.column_data["tissue"])


class TestSparseFramesIntIndex:
    """Test CellArraySE with sparse CellArrayFrames (integer index).

    Distinct from sparse *assays* (SparseCellArray): here the metadata frames
    themselves are sparse TileDB arrays, but their index is integer rather than
    string. This exercises a different code path in _get_frame_index.
    """

    def test_init(self, sample_cellarr_se_sparse_frames):
        assert sample_cellarr_se_sparse_frames.shape == (10, 5)

    def test_row_names_are_integers(self, sample_cellarr_se_sparse_frames):
        row_names = sample_cellarr_se_sparse_frames.row_names
        assert len(row_names) == 10
        assert row_names[0] == 0

    def test_slicing(self, sample_cellarr_se_sparse_frames):
        result = sample_cellarr_se_sparse_frames[0:5, 0:3]
        assert result.shape == (5, 3)
