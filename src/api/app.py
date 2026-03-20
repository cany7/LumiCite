from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes import APIError, api_router
from src.api.schemas import ErrorDetail, ErrorResponse


def create_app() -> FastAPI:
    app = FastAPI(title="RAG API")

    @app.exception_handler(APIError)
    async def handle_api_error(_, exc: APIError) -> JSONResponse:
        payload = ErrorResponse(
            error=ErrorDetail(
                code=exc.code,
                message=exc.message,
                detail=exc.detail,
            )
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_, exc: RequestValidationError) -> JSONResponse:
        payload = ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Request validation failed.",
                detail=str(exc),
            )
        )
        return JSONResponse(status_code=422, content=payload.model_dump())

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(_, exc: StarletteHTTPException) -> JSONResponse:
        status_code = int(exc.status_code)
        detail = str(exc.detail)
        if status_code == 404:
            code = "VALIDATION_ERROR"
            message = "Endpoint not found."
        elif status_code == 405:
            code = "VALIDATION_ERROR"
            message = "Method not allowed."
        else:
            code = "VALIDATION_ERROR" if 400 <= status_code < 500 else "INTERNAL_ERROR"
            message = "HTTP request failed." if 400 <= status_code < 500 else "An internal server error occurred."

        payload = ErrorResponse(
            error=ErrorDetail(
                code=code,
                message=message,
                detail=detail,
            )
        )
        return JSONResponse(status_code=status_code, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_, exc: Exception) -> JSONResponse:
        payload = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                message="An internal server error occurred.",
                detail=str(exc),
            )
        )
        return JSONResponse(status_code=500, content=payload.model_dump())

    app.include_router(api_router)
    return app
