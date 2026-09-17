"""
FastAPI application factory.
"""

import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import cameras_router, health_router, algorithms_router, alarm_events_router, scenes_router


def create_app(cors_origins: Optional[List[str]] = None) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        cors_origins: List of allowed CORS origins

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Maritime Vision AI Service API",
        description="REST API for camera management and AI service control",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    if cors_origins is None:
        cors_origins = ["http://localhost", "http://127.0.0.1"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files (screenshots)
    data_dir = Path(__file__).parent.parent.parent / "data"
    if data_dir.exists():
        app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")

    # Include routers
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(cameras_router, prefix="/api/v1")
    app.include_router(algorithms_router, prefix="/api/v1")
    app.include_router(alarm_events_router, prefix="/api/v1")
    app.include_router(scenes_router, prefix="/api/v1")

    return app
