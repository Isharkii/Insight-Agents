"""
app/validators/mapping_validator.py

Validation for schema mapping resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class MappingErrorDetail:
    """
    Structured mapping error detail.
    """

    code: str
    message: str
    canonical_field: str | None = None
    source_column: str | None = None
    context: dict[str, Any] | None = None


class SchemaMappingError(ValueError):
    """
    Raised when schema mapping cannot be resolved safely.
    """

    def __init__(self, *, message: str, errors: Sequence[MappingErrorDetail]) -> None:
        super().__init__(message)
        self.message = message
        self.errors = tuple(errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "errors": [
                {
                    "code": error.code,
                    "message": error.message,
                    "canonical_field": error.canonical_field,
                    "source_column": error.source_column,
                    "context": error.context,
                }
                for error in self.errors
            ],
        }


class MappingValidator:
    """
    Validates resolved canonical-to-source mappings.
    """

    def __init__(
        self,
        *,
        required_fields: Sequence[str],
        canonical_fields: Sequence[str],
    ) -> None:
        self._required_fields = tuple(required_fields)
        self._canonical_fields = tuple(canonical_fields)
        self._canonical_set = set(self._canonical_fields)

    def validate(
        self,
        *,
        mapping: dict[str, str],
        source_headers: Sequence[str],
        pre_errors: Sequence[MappingErrorDetail] | None = None,
    ) -> None:
        """
        Validate mapping and raise structured errors if invalid.
        """

        errors: list[MappingErrorDetail] = list(pre_errors or [])
        headers_set = set(source_headers)

        for canonical_field, source_column in mapping.items():
            if canonical_field not in self._canonical_set:
                errors.append(
                    MappingErrorDetail(
                        code="invalid_canonical_field",
                        message="Unknown canonical field in mapping.",
                        canonical_field=canonical_field,
                        source_column=source_column,
                    )
                )
            if source_column not in headers_set:
                errors.append(
                    MappingErrorDetail(
                        code="unknown_source_column",
                        message="Mapped source column does not exist in CSV headers.",
                        canonical_field=canonical_field,
                        source_column=source_column,
                    )
                )

        for required in self._required_fields:
            if required not in mapping:
                errors.append(
                    MappingErrorDetail(
                        code="required_field_unmapped",
                        message="Required canonical field is not mapped.",
                        canonical_field=required,
                        context={"source_headers": list(source_headers)},
                    )
                )

        if errors:
            missing_required = [
                error.canonical_field
                for error in errors
                if error.code == "required_field_unmapped" and error.canonical_field
            ]
            missing_csv = ", ".join(sorted(set(missing_required))) or "unknown"
            raise SchemaMappingError(
                message=f"Schema mapping validation failed. Missing required fields: {missing_csv}.",
                errors=errors,
            )
