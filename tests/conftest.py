"""Shared test fixtures for cellarr_se tests."""

import pytest
import tempfile
import os
import numpy as np
import pandas as pd
import tiledb
from cellarr_array import DenseCellArray, SparseCellArray
from cellarr_frame import CellArrayFrame


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_row_data(temp_dir):
    """Create a CellArrayFrame for row metadata (10 genes)."""
    uri = os.path.join(temp_dir, "row_data")
    df = pd.DataFrame(
        {
            "gene_id": [f"ENSG{i:05d}" for i in range(10)],
            "gene_name": ["TP53", "BRCA1", "EGFR", "MYC", "KRAS", "PIK3CA", "PTEN", "RB1", "APC", "VHL"],
            "gene_type": ["protein"] * 7 + ["lncRNA"] * 3,
        }
    )
    return CellArrayFrame.create(uri, df)


@pytest.fixture
def sample_col_data(temp_dir):
    """Create a CellArrayFrame for column metadata (5 samples)."""
    uri = os.path.join(temp_dir, "col_data")
    df = pd.DataFrame(
        {
            "sample_id": [f"SAMPLE_{i:03d}" for i in range(5)],
            "tissue": ["liver", "kidney", "liver", "brain", "kidney"],
            "treatment": ["control", "treated", "treated", "control", "control"],
        }
    )
    return CellArrayFrame.create(uri, df)


@pytest.fixture
def sample_row_data_named(temp_dir):
    """Create a CellArrayFrame for row metadata with string index (10 genes)."""
    uri = os.path.join(temp_dir, "row_data_named")
    df = pd.DataFrame(
        {
            "gene_name": ["TP53", "BRCA1", "EGFR", "MYC", "KRAS", "PIK3CA", "PTEN", "RB1", "APC", "VHL"],
            "gene_type": ["protein"] * 7 + ["lncRNA"] * 3,
        },
        index=[f"ENSG{i:05d}" for i in range(10)],
    )
    df.index.name = "gene_id"
    return CellArrayFrame.create(uri, df)


@pytest.fixture
def sample_col_data_named(temp_dir):
    """Create a CellArrayFrame for column metadata with string index (5 samples)."""
    uri = os.path.join(temp_dir, "col_data_named")
    df = pd.DataFrame(
        {
            "tissue": ["liver", "kidney", "liver", "brain", "kidney"],
            "treatment": ["control", "treated", "treated", "control", "control"],
        },
        index=[f"SAMPLE_{i:03d}" for i in range(5)],
    )
    df.index.name = "sample_id"
    return CellArrayFrame.create(uri, df)


@pytest.fixture
def sample_row_data_sparse_int(temp_dir):
    """Create a sparse CellArrayFrame with default integer index (10 genes)."""
    uri = os.path.join(temp_dir, "row_data_sparse_int")
    df = pd.DataFrame(
        {
            "gene_id": [f"ENSG{i:05d}" for i in range(10)],
            "gene_name": ["TP53", "BRCA1", "EGFR", "MYC", "KRAS", "PIK3CA", "PTEN", "RB1", "APC", "VHL"],
            "gene_type": ["protein"] * 7 + ["lncRNA"] * 3,
        }
    )
    return CellArrayFrame.create(uri, df, sparse=True)


@pytest.fixture
def sample_col_data_sparse_int(temp_dir):
    """Create a sparse CellArrayFrame with default integer index (5 samples)."""
    uri = os.path.join(temp_dir, "col_data_sparse_int")
    df = pd.DataFrame(
        {
            "sample_id": [f"SAMPLE_{i:03d}" for i in range(5)],
            "tissue": ["liver", "kidney", "liver", "brain", "kidney"],
            "treatment": ["control", "treated", "treated", "control", "control"],
        }
    )
    return CellArrayFrame.create(uri, df, sparse=True)


def _create_dense_array(uri: str, data: np.ndarray) -> DenseCellArray:
    """Helper to create a DenseCellArray from numpy data."""
    n_rows, n_cols = data.shape
    dom = tiledb.Domain(
        tiledb.Dim(name="rows", domain=(0, n_rows - 1), tile=n_rows, dtype=np.int32),
        tiledb.Dim(name="cols", domain=(0, n_cols - 1), tile=n_cols, dtype=np.int32),
    )
    schema = tiledb.ArraySchema(
        domain=dom,
        sparse=False,
        attrs=[tiledb.Attr(name="data", dtype=data.dtype)],
    )
    tiledb.Array.create(uri, schema)

    with tiledb.open(uri, "w") as A:
        A[:] = data

    return DenseCellArray(uri)


def _create_sparse_array(uri: str, data: np.ndarray) -> SparseCellArray:
    """Helper to create a SparseCellArray from numpy data (stores non-zero values)."""
    n_rows, n_cols = data.shape
    dom = tiledb.Domain(
        tiledb.Dim(name="rows", domain=(0, n_rows - 1), tile=n_rows, dtype=np.int32),
        tiledb.Dim(name="cols", domain=(0, n_cols - 1), tile=n_cols, dtype=np.int32),
    )
    schema = tiledb.ArraySchema(
        domain=dom,
        sparse=True,
        attrs=[tiledb.Attr(name="data", dtype=data.dtype)],
    )
    tiledb.Array.create(uri, schema)

    # Write non-zero values
    rows, cols = np.nonzero(data)
    values = data[rows, cols]

    with tiledb.open(uri, "w") as A:
        A[rows, cols] = {"data": values}

    return SparseCellArray(uri)


@pytest.fixture
def sample_counts_assay(temp_dir):
    """Create a DenseCellArray for counts (10 rows x 5 cols)."""
    uri = os.path.join(temp_dir, "counts")
    np.random.seed(42)
    data = np.random.randint(0, 1000, size=(10, 5)).astype(np.float64)
    return _create_dense_array(uri, data)


@pytest.fixture
def sample_tpm_assay(temp_dir):
    """Create a DenseCellArray for TPM values (10 rows x 5 cols)."""
    uri = os.path.join(temp_dir, "tpm")
    np.random.seed(123)
    data = np.random.rand(10, 5).astype(np.float64) * 100
    return _create_dense_array(uri, data)


@pytest.fixture
def sample_sparse_counts_assay(temp_dir):
    """Create a SparseCellArray for counts (10 rows x 5 cols, ~50% sparse)."""
    uri = os.path.join(temp_dir, "sparse_counts")
    np.random.seed(42)
    data = np.random.randint(0, 1000, size=(10, 5)).astype(np.float64)
    # Make ~50% of values zero for sparsity
    mask = np.random.rand(10, 5) < 0.5
    data[mask] = 0
    return _create_sparse_array(uri, data)


@pytest.fixture
def sample_cellarr_se(sample_row_data, sample_col_data, sample_counts_assay, sample_tpm_assay):
    """Create a complete CellArrSE object with dense assays for testing."""
    from cellarr_se import CellArrSE

    return CellArrSE(
        assays={"counts": sample_counts_assay, "tpm": sample_tpm_assay},
        row_data=sample_row_data,
        col_data=sample_col_data,
    )


@pytest.fixture
def sample_cellarr_se_named(sample_row_data_named, sample_col_data_named, sample_counts_assay, sample_tpm_assay):
    """Create a CellArrSE object with string-indexed frames for testing."""
    from cellarr_se import CellArrSE

    return CellArrSE(
        assays={"counts": sample_counts_assay, "tpm": sample_tpm_assay},
        row_data=sample_row_data_named,
        col_data=sample_col_data_named,
    )


@pytest.fixture
def sample_cellarr_se_sparse(sample_row_data, sample_col_data, sample_sparse_counts_assay):
    """Create a CellArrSE object with sparse assay for testing."""
    from cellarr_se import CellArrSE

    return CellArrSE(
        assays={"counts": sample_sparse_counts_assay},
        row_data=sample_row_data,
        col_data=sample_col_data,
    )


@pytest.fixture
def sample_cellarr_se_mixed(sample_row_data, sample_col_data, sample_counts_assay, sample_sparse_counts_assay):
    """Create a CellArrSE object with mixed dense and sparse assays."""
    from cellarr_se import CellArrSE

    return CellArrSE(
        assays={"dense_counts": sample_counts_assay, "sparse_counts": sample_sparse_counts_assay},
        row_data=sample_row_data,
        col_data=sample_col_data,
    )


@pytest.fixture
def sample_cellarr_se_sparse_frames(sample_row_data_sparse_int, sample_col_data_sparse_int, sample_counts_assay):
    """Create a CellArrSE object with sparse frames (integer index) for testing."""
    from cellarr_se import CellArrSE

    return CellArrSE(
        assays={"counts": sample_counts_assay},
        row_data=sample_row_data_sparse_int,
        col_data=sample_col_data_sparse_int,
    )
