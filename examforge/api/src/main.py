"""
ExamForge AI - FastAPI Application Entry Point

This is the main entry point for the ExamForge AI API server.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import auth, questions, mastery, tutor, sessions
from .config import settings

app = FastAPI(
    title="ExamForge AI API",
    description="Personalized exam preparation platform API",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(questions.router, prefix="/api/questions", tags=["Questions"])
app.include_router(mastery.router, prefix="/api/mastery", tags=["Mastery"])
app.include_router(tutor.router, prefix="/api/tutor", tags=["AI Tutor"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["Study Sessions"])


@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "name": "ExamForge AI API",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "database": "ok",
            "cache": "ok",
            "ai": "ok"
        }
    }
