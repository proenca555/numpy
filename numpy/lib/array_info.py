from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np


def _safe_stat(func, arr: np.ndarray, fallback: Any = None):
    """Call a NumPy function that may raise on empty / non-numeric arrays."""
    try:
        return func(arr) if arr.size > 0 else fallback
    except (TypeError, ValueError, OverflowError):
        return fallback


def array_info(
    arr: Union[np.ndarray, Any],
    *,
    sample_unique_threshold: int = 1_000_000,
    example_values: int = 3,
) -> Dict[str, Any]:
    """
    Return a dictionary summarising structure and basic statistics of *arr*.

    Parameters
    ----------
    arr : array-like
        The data to examine.  Converted to an ``ndarray`` if necessary.
    sample_unique_threshold : int, optional
        For very large arrays, computing the exact number of unique elements
        can be expensive.  If ``arr.size`` exceeds this threshold, a random
        subsample is used instead.
    example_values : int, optional
        How many values (head of the flattened array) to show in the
        ``'example_values'`` field.

    Returns
    -------
    Dict[str, Any]
        A mapping of field names to summary values.
    
    Example
    -------
    >>> import numpy as np, array_info as ai
    >>> a = np.array([[1, 2, np.nan], [4, 5, 6]])
    >>> ai.array_info(a)
    {
        'shape': (2, 3),
        'dtype': 'float64',
        'size': 6,
        'ndim': 2,
        'itemsize': 8,
        'contiguous': True,
        'aligned': True,
        'min': 1.0,
        'max': 6.0,
        'mean': 3.6,
        'std': 1.854049,
        'nan_count': 1,
        'inf_count': 0,
        'none_count': 0,
        'num_unique': 5,
        'example_values': [1.0, 2.0, nan]
    }
    """
    arr = np.asarray(arr)

    info: Dict[str, Any] = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "size": arr.size,
        "ndim": arr.ndim,
        "itemsize": arr.itemsize,
        "contiguous": bool(arr.flags.c_contiguous),
        "aligned": bool(arr.flags.aligned),
    }

    is_bool = arr.dtype == np.bool_
    is_numeric = (
        np.issubdtype(arr.dtype, np.number)
        or np.issubdtype(arr.dtype, np.complexfloating)
        or is_bool
    )

    if is_numeric:
        float_view = arr.astype(float, copy=False) if is_bool else arr
        finite_mask = np.isfinite(float_view)
        finite_vals = float_view[finite_mask]

        if finite_vals.size:
            info.update(
                {
                    "min": _safe_stat(np.nanmin, finite_vals),
                    "max": _safe_stat(np.nanmax, finite_vals),
                    "mean": _safe_stat(np.nanmean, finite_vals),
                    "std": _safe_stat(np.nanstd, finite_vals),
                }
            )
        else:
            info.update({"min": None, "max": None, "mean": None, "std": None})
    else:
        info.update({"min": None, "max": None, "mean": None, "std": None})

    info["nan_count"] = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    info["inf_count"] = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    info["none_count"] = int(np.sum(arr == None)) if arr.dtype == object else 0  

    if arr.size == 0:
        info["num_unique"] = 0
    elif arr.dtype == object:
        info["num_unique"] = "n/a"            
    elif arr.size <= sample_unique_threshold:
        info["num_unique"] = int(len(np.unique(arr)))
    else:
        if arr.size <= sample_unique_threshold:
            try:
                info["num_unique"] = int(len(np.unique(arr)))
            except Exception:  
                info["num_unique"] = "n/a"
        else: 
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(arr.size, size=sample_unique_threshold, replace=False)
            sample = arr.flatten()[sample_idx]
            try:
                info["num_unique"] = f">= {len(np.unique(sample))} (sample)"
            except Exception:
                info["num_unique"] = "n/a (sample)"

    flat = arr.flatten()
    info["example_values"] = flat[:example_values].tolist() if flat.size else []

    return info