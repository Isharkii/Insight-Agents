from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api.error_handling import register_exception_handlers


def _build_test_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/http-error")
    def http_error() -> None:
        raise HTTPException(status_code=400, detail="Invalid request payload.")

    @app.get("/runtime-error")
    def runtime_error() -> None:
        raise RuntimeError("boom")

    @app.get("/typed/{value}")
    def typed(value: int) -> dict[str, int]:
        return {"value": value}

    return app


def test_http_exception_is_normalized_with_machine_code() -> None:
    client = TestClient(_build_test_app())
    response = client.get("/http-error")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "schema_conflict"
    assert "message" in detail


def test_unhandled_exception_is_internal_failure_with_machine_code() -> None:
    client = TestClient(_build_test_app(), raise_server_exceptions=False)
    response = client.get("/runtime-error")
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["code"] == "internal_failure"
    assert detail["message"] == "Internal server error."


def test_request_validation_error_is_normalized_with_machine_code() -> None:
    client = TestClient(_build_test_app())
    response = client.get("/typed/not-an-int")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "schema_conflict"
    assert detail["message"] == "Request validation failed."
