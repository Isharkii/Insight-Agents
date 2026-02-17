"""
app/api/dependencies.py

Shared FastAPI dependencies for request validation.
"""

from __future__ import annotations

from fastapi import File, HTTPException, UploadFile, status

CSV_CONTENT_TYPES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
}


def get_csv_upload(file: UploadFile = File(...)) -> UploadFile:
    """
    Validate that the uploaded file is a CSV by extension or MIME type.
    """

    filename = (file.filename or "").strip().lower()
    content_type = (file.content_type or "").strip().lower()

    is_csv_filename = filename.endswith(".csv")
    is_csv_content_type = content_type in CSV_CONTENT_TYPES

    if not is_csv_filename and not is_csv_content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are allowed.",
        )

    return file
