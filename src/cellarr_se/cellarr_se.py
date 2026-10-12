"""
A logical, read-only coordinator for TileDB-backed multi-dimensional datasets.

This class synchronizes slicing and metadata retrieval across multiple out-of-core
components:

- Assays: A dictionary of `cellarr-array` objects (Dense or Sparse).
- Row Data: An aligned `cellarr-frame` for row-wise annotations.
- Column Data: An aligned `cellarr-frame` for column-wise annotations.

CellArraySE maintains data on disk, performing synchronized "lazy" slices that
return standard in-memory `summarizedexperiment.SummarizedExperiment` objects
only when requested.
"""

from functools import cached_property

import numpy as np
import pandas as pd
from cellarr_array import SparseCellArray
from cellarr_array.core import CellArray
from cellarr_frame import CellArrayFrame
from summarizedexperiment import SummarizedExperiment

__author__ = "chanjd"
__copyright__ = "chanjd"
__license__ = "MIT"


def _get_frame_index(frame: CellArrayFrame) -> pd.Index:
    """Get frame index efficiently without loading column data.

    Sparse frames store explicit index values (integer or string) via frame.index.
    Dense frames only support 0-based integer indices, derived from nonempty_domain.

    Note: only supports single CellArrayFrame objects with 1 dimension for now.
    """
    idx_df = frame.index
    if not idx_df.empty:
        # Sparse frame - decode bytes to strings
        dim_name = frame.index_names[0]
        values = [v.decode() if isinstance(v, bytes) else v for v in idx_df[dim_name]]
        return pd.Index(values, name=dim_name)
    else:
        # Dense frame - derive RangeIndex from nonempty_domain
        with frame.open_array(mode="r") as arr:
            ned = arr.nonempty_domain()
            if ned and ned[0]:
                start, stop = ned[0]
                return pd.RangeIndex(start=start, stop=stop + 1)
            return pd.RangeIndex(start=0, stop=0)


#: Type alias for subset keys accepted by ``__getitem__`` and ``slice()``.
#:
#: Supported types:
#:   - ``int``: Single index (e.g., ``5`` or ``-1`` for last element)
#:   - ``slice``: Range of indices (e.g., ``slice(0, 10)`` or ``0:10`` in brackets)
#:   - ``List[int]``: Multiple indices (e.g., ``[0, 2, 4]``)
#:   - ``str``: Single name matching row_names/col_names
#:   - ``List[str]``: Multiple names (e.g., ``["gene1", "gene2"]``)
#:   - ``None``: Select all elements in that dimension
SubsetKey = int | slice | list[int] | str | list[str] | None


def _validate_input(assays: dict[str, CellArray], row_data: CellArrayFrame, col_data: CellArrayFrame):
    """Validate that assays and metadata frames are compatible.

    Checks:
        - assays is a non-empty dict of CellArray instances
        - row_data and col_data are CellArrayFrame instances
        - All assays have the same shape
        - Assay row count matches row_data row count
        - Assay column count matches col_data row count

    Raises:
        TypeError:
            If inputs are not the expected types.
        ValueError:
            If assays is empty or shapes don't match.
    """

    # Type Enforcement
    if not isinstance(assays, dict):
        raise TypeError(f"Assays must be a dictionary, got {type(assays)}.")
    if not assays:
        raise ValueError("The 'assays' dictionary cannot be empty.")
    for name, arr in assays.items():
        if not isinstance(arr, CellArray):
            raise TypeError(f"Assay '{name}' must be a CellArray instance, got {type(arr)}.")

    if not isinstance(row_data, CellArrayFrame):
        raise TypeError(f"row_data must be a CellArrayFrame instance, got {type(row_data)}.")
    if not isinstance(col_data, CellArrayFrame):
        raise TypeError(f"col_data must be a CellArrayFrame instance, got {type(col_data)}.")

    n_rows = len(_get_frame_index(row_data))
    n_cols = len(_get_frame_index(col_data))
    base_shape = (n_rows, n_cols)

    for name, arr in assays.items():
        if arr.shape != base_shape:
            raise ValueError(f"Assay '{name}' shape {arr.shape} != {base_shape}.")


class CellArraySE:
    def __init__(
        self,
        assays: dict[str, CellArray],
        row_data: CellArrayFrame,
        col_data: CellArrayFrame,
    ):
        """Initialize the SE coordinator with existing TileDB-backed handles.

        Args:
            assays:
                Dictionary mapping assay names to CellArray objects.
                All assays must have the same shape.

            row_data:
                CellArrayFrame containing row metadata.
                Number of rows must match assay row count.

            col_data:
                CellArrayFrame containing column metadata.
                Number of rows must match assay column count.

        Raises:
            ValueError:
                If assays is empty, shapes don't match, or inputs are invalid types.
        """
        _validate_input(assays, row_data, col_data)

        self.assays = assays
        self.row_data = row_data
        self.col_data = col_data

    # --- Shape & Dim Getters ---

    @cached_property
    def shape(self) -> tuple[int, int]:
        """Number of rows and columns as (n_rows, n_cols)."""
        first_assay = next(iter(self.assays.values()))
        return first_assay.shape

    @property
    def dims(self) -> tuple[int, int]:
        """Alias for shape."""
        return self.shape

    # --- Index Accessors ---

    @cached_property
    def row_names(self) -> pd.Index:
        """Index values for rows metadata table."""
        return _get_frame_index(self.row_data)

    @cached_property
    def col_names(self) -> pd.Index:
        """Index values for columns metadata table."""
        return _get_frame_index(self.col_data)

    # --- Metadata Discovery ---

    @cached_property
    def assay_names(self) -> list[str]:
        """Names of available assays."""
        return list(self.assays.keys())

    @cached_property
    def row_columns(self) -> list[str]:
        """Column names of the metadata fields for row_data."""
        return self.row_data.column_names

    @cached_property
    def col_columns(self) -> list[str]:
        """Column names of the metadata fields for column_data."""
        return self.col_data.column_names

    def is_sparse(self, assay_name: str) -> bool:
        """Check if an assay is sparse.

        Args:
            assay_name:
                Name of the assay to check.

        Returns:
            True if the assay is a SparseCellArray, False otherwise.
        """
        if assay_name not in self.assays:
            raise KeyError(f"Assay '{assay_name}' not found.")
        return isinstance(self.assays[assay_name], SparseCellArray)

    def get_assay_type(self, assay_name: str) -> np.dtype:
        """Get the data type of an assay matrix.

        Args:
            assay_name:
                Name of the assay.

        Returns:
            NumPy dtype of the assay's data attribute.

        Raises:
            KeyError:
                If the assay name does not exist.
        """
        if assay_name not in self.assays:
            raise KeyError(f"Assay '{assay_name}' not found.")

        handle = self.assays[assay_name]

        # Open array momentarily to read metadata (Zero-copy)
        with handle.open_array(mode="r") as A:
            return A.schema.attr(0).dtype

    # --- cellarr-se Object Summary ---

    def show(self, n: int = 5):
        """Display a summary of the experiment structure and metadata.

        Args:
            n:
                Number of rows to display from row_data and col_data.
                Defaults to 5.
        """
        print(f"CellArraySE Object | {self.shape[0]} rows x {self.shape[1]} cols")
        print(f"Assays: {', '.join(self.assay_names)}")
        print("\n--- Row Data ---")
        # Use row_names for slicing to support string-indexed frames
        row_subset = list(self.row_names[:n])
        print(self.row_data[row_subset])
        print("\n--- Column Data ---")
        col_subset = list(self.col_names[:n])
        print(self.col_data[col_subset])

    # --- Subsetting/Slicing ---

    def _resolve_key_to_indices(
        self,
        key: SubsetKey,
        names: pd.Index,
        dim_size: int,
    ) -> list[int]:
        """Convert any subset key type to a list of integer positions.

        This normalizes the various ways users can specify subsets (int, slice,
        name, list) into a consistent list of integer indices. These indices
        are then used to slice both metadata frames and assay matrices.

        Args:
            key:
                The subset specification. Can be:
                - None: select all
                - int: single position (supports negative indexing)
                - slice: range of positions (no step support)
                - str: single name lookup
                - List[int]: multiple positions
                - List[str]: multiple name lookups

            names:
                Index of the dimension (row_names or col_names) for name lookups.

            dim_size:
                Size of the dimension for bounds checking and negative index resolution.

        Returns:
            List of integer positions.

        Raises:
            IndexError:
                If position is out of bounds or slice has a step.
            KeyError:
                If name is not found in the index.
            TypeError:
                If key type is not supported.
        """
        if key is None:
            return list(range(dim_size))

        if isinstance(key, int):
            if key < 0:
                key = dim_size + key
            if key < 0 or key >= dim_size:
                raise IndexError(f"Index {key} out of bounds for dimension of size {dim_size}.")
            return [key]

        if isinstance(key, slice):
            if key.step is not None:
                raise IndexError("Slice steps (strides) are not supported.")
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else dim_size
            if start < 0:
                start = dim_size + start
            if stop < 0:
                stop = dim_size + stop
            return list(range(start, stop))

        if isinstance(key, str):
            # Single name lookup
            if key not in names:
                raise KeyError(f"Name '{key}' not found.")
            return [names.get_loc(key)]

        if isinstance(key, list):
            if not key:
                return []
            if not all(isinstance(k, type(key[0])) for k in key):
                raise TypeError("List elements must all be the same type, got mixed types.")
            if isinstance(key[0], int):
                # List of integers
                resolved = []
                for idx in key:
                    if idx < 0:
                        idx = dim_size + idx
                    if idx < 0 or idx >= dim_size:
                        raise IndexError(f"Index {idx} out of bounds for dimension of size {dim_size}.")
                    resolved.append(idx)
                return resolved
            if isinstance(key[0], str):
                # List of names
                resolved = []
                for name in key:
                    if name not in names:
                        raise KeyError(f"Name '{name}' not found.")
                    resolved.append(names.get_loc(name))
                return resolved
            raise TypeError(f"List elements must be int or str, got {type(key[0])}.")

        raise TypeError(f"Unsupported key type: {type(key)}. Expected int, slice, str, List[int], or List[str].")

    def _subset_frame(
        self,
        handle: CellArrayFrame,
        subset: SubsetKey = None,
        query: str | None = None,
        columns: list[str] | None = None,
        names: pd.Index | None = None,
        dim_size: int | None = None,
    ) -> pd.DataFrame:
        """Subset a CellArrayFrame using either positional/name-based subset or TileDB query.

        Args:
            handle: The CellArrayFrame to subset.
            subset: Positional or name-based subset key.
            query: TileDB query condition string.
            columns: Columns to select from the frame.
            names: Row/column names for name-based lookups (required if subset contains strings).
            dim_size: Dimension size for slice resolution.

        Returns:
            Subsetted DataFrame.
        """
        if subset is not None and query is not None:
            raise ValueError("Cannot specify both 'subset' and 'query'. Use one or the other.")

        # Query-based filtering
        if query is not None:
            if columns is not None:
                return handle[query, columns]
            return handle[query]

        # No subset - return all rows
        if subset is None:
            if columns is not None:
                return handle[:, columns]
            return handle[:]

        indices = self._resolve_key_to_indices(subset, names, dim_size)

        # Convert integer indices to names for string-indexed frames
        # Use the passed 'names' parameter (already cached) instead of re-reading
        if len(names) > 0 and isinstance(names[0], str):
            if isinstance(indices, list) and len(indices) > 0 and isinstance(indices[0], (int, np.integer)):
                indices = [names[i] for i in indices]

        # Use bracket notation for subsetting
        if columns is not None:
            return handle[indices, columns]
        return handle[indices]

    def slice(
        self,
        row_subset: SubsetKey = None,
        col_subset: SubsetKey = None,
        row_query: str | None = None,
        col_query: str | None = None,
        assays: list[str] | None = None,
        row_columns: list[str] | None = None,
        col_columns: list[str] | None = None,
    ) -> SummarizedExperiment:
        """Slice the CellArraySE to produce an in-memory SummarizedExperiment.

        This method provides full control over subsetting, including TileDB query
        support. For simple positional/name-based access, use bracket notation
        instead (e.g., ``se[0:10, 0:5]``).

        Args:
            row_subset: Row subset key. Accepted types:

                - ``int``: Single index (e.g., ``5``, ``-1`` for last)
                - ``slice``: Range (e.g., ``slice(0, 10)``)
                - ``List[int]``: Multiple indices (e.g., ``[0, 2, 4]``)
                - ``str``: Single name matching row_names
                - ``List[str]``: Multiple names
                - ``None``: Select all rows (default)

            col_subset: Column subset key. Same types as row_subset.
            row_query: TileDB query string for row filtering (e.g.,
                ``"gene_type == 'protein'"``. Mutually exclusive with row_subset.
            col_query: TileDB query string for column filtering. Mutually exclusive
                with col_subset.
            assays: List of assay names to include. If None, includes all assays.
            row_columns: List of row metadata columns to include. If None, includes all.
            col_columns: List of column metadata columns to include. If None, includes all.

        Returns:
            SummarizedExperiment with the requested subset of data.

        Raises:
            ValueError: If both subset and query are specified for the same dimension.
            KeyError: If assay name or row/column name is not found.
            IndexError: If index is out of bounds.

        Examples:
            Basic positional subsetting::

                subset = se.slice(
                    row_subset=slice(
                        0, 100
                    ),
                    col_subset=slice(
                        0, 50
                    ),
                )

            Select specific assays and metadata columns::

                subset = se.slice(
                    row_subset=[
                        0,
                        1,
                        2,
                    ],
                    col_subset=slice(
                        0, 10
                    ),
                    assays=[
                        "counts"
                    ],
                    row_columns=[
                        "gene_id",
                        "gene_name",
                    ],
                )

            Filter using TileDB query strings::

                # Get protein-coding genes from liver samples
                subset = se.slice(
                    row_query="gene_type == 'protein'",
                    col_query="tissue == 'liver'",
                )

            Combine query with column selection::

                subset = se.slice(
                    row_query="gene_type == 'protein'",
                    col_subset=[
                        0,
                        1,
                        2,
                    ],
                    assays=[
                        "counts",
                        "tpm",
                    ],
                )
        """
        # Validate assay names
        if assays is not None:
            for name in assays:
                if name not in self.assays:
                    raise KeyError(f"Assay '{name}' not found. Available: {self.assay_names}")

        # Validate mutual exclusion before branching
        if row_subset is not None and row_query is not None:
            raise ValueError("Cannot specify both 'subset' and 'query'. Use one or the other.")
        if col_subset is not None and col_query is not None:
            raise ValueError("Cannot specify both 'subset' and 'query'. Use one or the other.")

        # Queries require string-indexed frames — integer-indexed frames have no named
        # attributes to filter on; use positional slicing instead.
        if row_query is not None and (not len(self.row_names) or not isinstance(self.row_names[0], str)):
            raise ValueError("row_query requires a string-indexed row frame.")
        if col_query is not None and (not len(self.col_names) or not isinstance(self.col_names[0], str)):
            raise ValueError("col_query requires a string-indexed column frame.")

        # Resolve assay indices directly — do not derive from the returned DataFrame
        # index, which is reset to 0-based for dense frames and cannot be trusted.
        if row_query is not None:
            # Query path: frame subset first, then convert preserved string index to positions
            df_row = self._subset_frame(
                self.row_data,
                query=row_query,
                columns=row_columns,
                names=self.row_names,
                dim_size=self.shape[0],
            )
            row_indices = [self.row_names.get_loc(name) for name in df_row.index.tolist()]
        else:
            row_indices = self._resolve_key_to_indices(row_subset, self.row_names, self.shape[0])
            df_row = self._subset_frame(
                self.row_data,
                subset=row_subset,
                columns=row_columns,
                names=self.row_names,
                dim_size=self.shape[0],
            )

        if col_query is not None:
            df_col = self._subset_frame(
                self.col_data,
                query=col_query,
                columns=col_columns,
                names=self.col_names,
                dim_size=self.shape[1],
            )
            col_indices = [self.col_names.get_loc(name) for name in df_col.index.tolist()]
        else:
            col_indices = self._resolve_key_to_indices(col_subset, self.col_names, self.shape[1])
            df_col = self._subset_frame(
                self.col_data,
                subset=col_subset,
                columns=col_columns,
                names=self.col_names,
                dim_size=self.shape[1],
            )

        # Determine which assays to include
        assay_names_to_use = assays if assays is not None else self.assay_names

        # Fetch assay data
        sub_assays = {name: self.assays[name][row_indices, col_indices] for name in assay_names_to_use}

        return SummarizedExperiment(
            assays=sub_assays,
            row_data=df_row,
            column_data=df_col,
        )

    def __getitem__(self, key: tuple[SubsetKey, SubsetKey]) -> SummarizedExperiment:
        """Subset using bracket notation: ``se[rows, cols]``.

        This method provides simple positional and name-based subsetting. For
        advanced filtering with TileDB query strings, use :meth:`slice` instead.

        Supported key types (see :data:`SubsetKey`):
            - ``int``: Single index (e.g., ``se[0, 5]``)
            - ``slice``: Range (e.g., ``se[0:10, 0:5]``)
            - ``List[int]``: Multiple indices (e.g., ``se[[0, 1, 2], [3, 4]]``)
            - ``str``: Single name (e.g., ``se["gene1", "sample1"]``)
            - ``List[str]``: Multiple names (e.g., ``se[["gene1", "gene2"], ["s1", "s2"]]``)

        Note:
            Query-based filtering is **not** supported via bracket notation.
            Use ``se.slice(row_query="...", col_query="...")`` for TileDB queries.

        Args:
            key: A 2-tuple of (row_key, col_key).

        Returns:
            SummarizedExperiment with the requested subset.

        Raises:
            ValueError: If key is not a 2-tuple.
            TypeError: If key types are not supported.
            IndexError: If indices are out of bounds.
            KeyError: If names are not found.

        Examples:
            ::

                # Slice by position
                subset = se[
                    0:100, 0:50
                ]

                # Single indices
                single = se[5, 3]

                # List of indices
                subset = se[
                    [0, 2, 4],
                    [1, 3],
                ]

                # For query-based filtering, use slice():
                subset = se.slice(
                    row_query="gene_type == 'protein'"
                )
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(
                "Slicing requires a 2-dimensional tuple (e.g., se[0:10, 0:5]). "
                "For query-based filtering, use se.slice(row_query=..., col_query=...)."
            )

        row_key, col_key = key

        # Validate key types
        valid_types = (int, slice, str, list, type(None))
        if not isinstance(row_key, valid_types):
            raise TypeError(f"Row key must be int, slice, str, or list, got {type(row_key)}.")
        if not isinstance(col_key, valid_types):
            raise TypeError(f"Column key must be int, slice, str, or list, got {type(col_key)}.")

        return self.slice(row_subset=row_key, col_subset=col_key)

    def __repr__(self) -> str:
        """String representation showing shape and assay names."""
        return f"<CellArraySE: {self.shape[0]}x{self.shape[1]} | {', '.join(self.assay_names)}>"
