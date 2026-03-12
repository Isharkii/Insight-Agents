from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from app.failure_codes import AUTHENTICATION_FAILED
from app.security.errors import SecurityError


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("utf-8"))


def _decode_json_part(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(_b64url_decode(value).decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SecurityError(
            status_code=401,
            code=AUTHENTICATION_FAILED,
            message="Malformed JWT payload.",
        ) from exc
    if not isinstance(payload, dict):
        raise SecurityError(
            status_code=401,
            code=AUTHENTICATION_FAILED,
            message="Invalid JWT payload shape.",
        )
    return payload


def decode_and_verify_hs256_jwt(
    token: str,
    *,
    secret_key: str,
    issuer: str | None = None,
    audience: str | None = None,
    leeway_seconds: int = 30,
) -> dict[str, Any]:
    """
    Decode and verify a compact JWT signed with HS256.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise SecurityError(
            status_code=401,
            code=AUTHENTICATION_FAILED,
            message="Malformed JWT token.",
        )

    header_b64, payload_b64, signature_b64 = parts
    header = _decode_json_part(header_b64)
    payload = _decode_json_part(payload_b64)

    algorithm = str(header.get("alg") or "").upper()
    if algorithm != "HS256":
        raise SecurityError(
            status_code=401,
            code=AUTHENTICATION_FAILED,
            message="Unsupported JWT algorithm.",
            context={"alg": algorithm or None},
        )

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected_sig = hmac.new(
        secret_key.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    try:
        actual_sig = _b64url_decode(signature_b64)
    except Exception as exc:  # noqa: BLE001
        raise SecurityError(
            status_code=401,
            code=AUTHENTICATION_FAILED,
            message="Malformed JWT signature.",
        ) from exc

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise SecurityError(
            status_code=401,
            code=AUTHENTICATION_FAILED,
            message="JWT signature verification failed.",
        )

    now = int(time.time())
    exp = payload.get("exp")
    if exp is not None:
        try:
            exp_i = int(exp)
        except (TypeError, ValueError) as exc:
            raise SecurityError(
                status_code=401,
                code=AUTHENTICATION_FAILED,
                message="Invalid JWT exp claim.",
            ) from exc
        if now > exp_i + leeway_seconds:
            raise SecurityError(
                status_code=401,
                code=AUTHENTICATION_FAILED,
                message="JWT token has expired.",
            )

    nbf = payload.get("nbf")
    if nbf is not None:
        try:
            nbf_i = int(nbf)
        except (TypeError, ValueError) as exc:
            raise SecurityError(
                status_code=401,
                code=AUTHENTICATION_FAILED,
                message="Invalid JWT nbf claim.",
            ) from exc
        if now + leeway_seconds < nbf_i:
            raise SecurityError(
                status_code=401,
                code=AUTHENTICATION_FAILED,
                message="JWT token not active yet.",
            )

    iss = payload.get("iss")
    if issuer and str(iss or "") != issuer:
        raise SecurityError(
            status_code=401,
            code=AUTHENTICATION_FAILED,
            message="JWT issuer mismatch.",
        )

    if audience:
        aud_claim = payload.get("aud")
        if isinstance(aud_claim, list):
            aud_values = {str(item) for item in aud_claim}
        else:
            aud_values = {str(aud_claim)} if aud_claim is not None else set()
        if audience not in aud_values:
            raise SecurityError(
                status_code=401,
                code=AUTHENTICATION_FAILED,
                message="JWT audience mismatch.",
            )

    return payload
