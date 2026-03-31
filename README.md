[![PyPI-Server](https://img.shields.io/pypi/v/cellarr-se.svg)](https://pypi.org/project/cellarr-se/)
[![CI](https://github.com/CellArr/cellarr-se/actions/workflows/run-tests.yml/badge.svg)](https://github.com/CellArr/cellarr-se/actions/workflows/run-tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# cellarr-se

`cellarr-se` is a read-only, out-of-core coordinator for TileDB-backed genomic datasets. It wraps the [`cellarr-array`](https://pypi.org/project/cellarr-array/) and [`cellarr-frame`](https://pypi.org/project/cellarr-frame/) primitives into a lazy, [`SummarizedExperiment`](https://pypi.org/project/summarizedexperiment/)-compatible interface, so you can slice large genomics datasets stored on disk without loading them into memory.

Single-cell and bulk RNA-seq datasets frequently exceed available RAM. `cellarr-se` keeps assay matrices and metadata tables on disk as TileDB arrays, performing synchronized lazy slices across all components only when you request them. The result is always a standard in-memory `SummarizedExperiment` object.

## Install

```bash
pip install cellarr-se
```

## Usage

### Construction

`CellArrSE` wraps existing TileDB arrays and frames; it does not create them. Use `cellarr-array` and `cellarr-frame` to build the backing stores first.

```python
from cellarr_se import CellArrSE

se = CellArrSE(
    assays={"counts": my_cell_array, "tpm": my_tpm_array},
    row_data=my_row_frame,   # gene annotations (CellArrayFrame)
    col_data=my_col_frame,   # sample annotations (CellArrayFrame)
)
```

### Inspection

```python
se.shape          # (n_genes, n_samples)
se.assay_names    # ["counts", "tpm"]
se.row_names      # pd.Index of gene identifiers
se.col_names      # pd.Index of sample identifiers
se.row_columns    # list of gene metadata fields
se.col_columns    # list of sample metadata fields

se.show()         # print a summary with the first 5 rows of each metadata table
repr(se)          # <CellArrSE: 20000x500 | counts, tpm>
```

### Slicing

Bracket notation supports integer indices, slices, name strings, and lists:

```python
# Positional slice
subset = se[0:100, 0:50]

# Single element
gene = se[5, 3]

# Lists of indices or names
subset = se[["BRCA1", "TP53"], ["sample_001", "sample_042"]]
```

For attribute-filtered access, use `slice()` with TileDB query strings:

```python
# Filter rows and columns by metadata attributes
subset = se.slice(
    row_query="gene_type == 'protein_coding'",
    col_query="tissue == 'liver'",
)

# Combine query with explicit column selection
subset = se.slice(
    row_query="gene_type == 'protein_coding'",
    col_subset=slice(0, 50),
    assays=["counts"],
    row_columns=["gene_id", "gene_name"],
)
```

Both `se[...]` and `se.slice(...)` return a standard in-memory `SummarizedExperiment`.

### Assay metadata

```python
se.is_sparse("counts")        # True if backed by SparseCellArray
se.get_assay_type("counts")   # numpy dtype of the assay
```
