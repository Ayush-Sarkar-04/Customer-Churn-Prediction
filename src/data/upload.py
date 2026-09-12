from io import BytesIO
from typing import Dict, Mapping, Any

import pandas as pd

from config import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MAX_FILES,
    MAX_ROWS,
)

from src.data.validator import validate_uploaded_datasets


REQUIRED_DATASETS = {
    "customers",
    "transactions",
    "campaigns",
}


def _get_file_bytes(uploaded_file: Any) -> bytes:
    """
    Read bytes from a Streamlit UploadedFile-like object,
    bytes, or bytearray.
    """
    if isinstance(uploaded_file, bytes):
        return uploaded_file

    if isinstance(uploaded_file, bytearray):
        return bytes(uploaded_file)

    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()

    if hasattr(uploaded_file, "read"):
        uploaded_file.seek(0)
        return uploaded_file.read()

    raise TypeError("Unsupported uploaded file object.")


def _validate_file_size(
    dataset_name: str,
    file_bytes: bytes,
) -> list[str]:
    """Validate the size of one uploaded file."""
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)

        return [
            (
                f"{dataset_name}: file size is {size_mb:.2f} MB; "
                f"maximum allowed size is {MAX_FILE_SIZE_MB} MB."
            )
        ]

    return []


def _read_csv(
    dataset_name: str,
    file_bytes: bytes,
) -> tuple[pd.DataFrame | None, list[str]]:
    """
    Read one CSV safely and enforce the row limit.
    """
    errors = _validate_file_size(dataset_name, file_bytes)

    if errors:
        return None, errors

    try:
        df = pd.read_csv(BytesIO(file_bytes))
    except Exception as exc:
        return None, [
            f"{dataset_name}: unable to read CSV file: {exc}"
        ]

    if len(df) > MAX_ROWS:
        return None, [
            (
                f"{dataset_name}: file contains {len(df):,} rows; "
                f"maximum allowed is {MAX_ROWS:,} rows."
            )
        ]

    return df, []


def load_uploaded_datasets(
    uploaded_files: Mapping[str, Any],
) -> dict:
    """
    Load and validate Customers, Transactions and Campaigns uploads.

    Parameters
    ----------
    uploaded_files:
        Mapping with exactly these dataset names:
        customers, transactions, campaigns.

    Returns
    -------
    dict
        {
            "valid": bool,
            "errors": list[str],
            "data": {
                "customers": DataFrame,
                "transactions": DataFrame,
                "campaigns": DataFrame,
            }
        }

    Data is returned only when the complete upload passes validation.
    """

    errors: list[str] = []

    if not isinstance(uploaded_files, Mapping):
        return {
            "valid": False,
            "errors": ["uploaded_files must be a mapping."],
            "data": None,
        }

    file_count = len(uploaded_files)

    if file_count > MAX_FILES:
        errors.append(
            (
                f"Too many files supplied: {file_count}. "
                f"Maximum allowed is {MAX_FILES}."
            )
        )

    missing_datasets = REQUIRED_DATASETS - set(uploaded_files.keys())

    if missing_datasets:
        errors.extend(
            [
                f"Missing required dataset: {dataset}."
                for dataset in sorted(missing_datasets)
            ]
        )

    unexpected_datasets = set(uploaded_files.keys()) - REQUIRED_DATASETS

    if unexpected_datasets:
        errors.extend(
            [
                f"Unexpected dataset supplied: {dataset}."
                for dataset in sorted(unexpected_datasets)
            ]
        )

    if errors:
        return {
            "valid": False,
            "errors": errors,
            "data": None,
        }

    data: Dict[str, pd.DataFrame] = {}

    for dataset_name in (
        "customers",
        "transactions",
        "campaigns",
    ):
        uploaded_file = uploaded_files[dataset_name]

        if uploaded_file is None:
            errors.append(
                f"{dataset_name}: no file was supplied."
            )
            continue

        try:
            file_bytes = _get_file_bytes(uploaded_file)
        except Exception as exc:
            errors.append(
                f"{dataset_name}: unable to read uploaded file: {exc}"
            )
            continue

        df, read_errors = _read_csv(
            dataset_name,
            file_bytes,
        )

        errors.extend(read_errors)

        if df is not None:
            data[dataset_name] = df

    if errors:
        return {
            "valid": False,
            "errors": errors,
            "data": None,
        }

    # Run the existing project-level validator.
    validation_result = validate_uploaded_datasets(
        data["customers"],
        data["transactions"],
        data["campaigns"],
    )

    if not validation_result.get("valid", False):
        validation_errors = validation_result.get(
            "errors",
            ["Uploaded data failed validation."],
        )

        return {
            "valid": False,
            "errors": validation_errors,
            "data": None,
        }

    return {
        "valid": True,
        "errors": [],
        "data": data,
    }


def upload_result_is_valid(result: dict) -> bool:
    """Convenience helper for application-layer checks."""
    return bool(
        isinstance(result, dict)
        and result.get("valid") is True
        and not result.get("errors")
        and isinstance(result.get("data"), dict)
    )