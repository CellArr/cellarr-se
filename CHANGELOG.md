# Changelog

## Version 0.1.0

Initial public release.

- `CellArrSE`: read-only, out-of-core coordinator wrapping `cellarr-array` assay matrices and `cellarr-frame` row/column metadata tables.
- Lazy, synchronized slicing across all components — data is only loaded when a slice is materialized.
- Bracket notation (`se[rows, cols]`) supporting integer indices, slices, name strings, and lists of either.
- `slice()` method with TileDB query string support for attribute-filtered access on rows and columns.
- Per-dimension control over which assays and metadata columns are included in each slice.
- Slice output is always a standard in-memory `summarizedexperiment.SummarizedExperiment`.
- `show()` for a human-readable summary of the experiment structure and metadata.
- `is_sparse()` and `get_assay_type()` for introspection without loading assay data.
