"""
ExamForge AI - Authentication Router

Handles user authentication, registration, and token management.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

router = APIRouter()


class UserRegister(BaseModel):
    """User registration request model."""
    email: EmailStr
    password: str
    name: str
    exam_type: str = "jee_main"
    target_date: Optional[str] = None


class UserLogin(BaseModel):
    """User login request model."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """JWT token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    """User data response model."""
    id: str
    email: str
    name: str
    exam_type: str
    target_date: Optional[str]
    created_at: datetime


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister):
    """
    Register a new user.
    
    Creates a new user account and returns JWT tokens.
    """
    # TODO: Implement with Supabase Auth
    # For now, return a placeholder response
    return TokenResponse(
        access_token="placeholder_access_token",
        refresh_token="placeholder_refresh_token",
        expires_in=3600
    )


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    """
    Authenticate user and return tokens.
    """
    # TODO: Implement with Supabase Auth
    return TokenResponse(
        access_token="placeholder_access_token",
        refresh_token="placeholder_refresh_token",
        expires_in=3600
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """
    Refresh access token using refresh token.
    """
    # TODO: Implement token refresh
    return TokenResponse(
        access_token="new_placeholder_access_token",
        refresh_token="new_placeholder_refresh_token",
        expires_in=3600
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user():
    """
    Get current authenticated user's profile.
    """
    # TODO: Implement with auth dependency
    return UserResponse(
        id="placeholder_id",
        email="user@example.com",
        name="Test User",
        exam_type="jee_main",
        target_date="2026-01-15",
        created_at=datetime.now()
    )


@router.post("/logout")
async def logout():
    """
    Logout user and invalidate tokens.
    """
    # TODO: Implement token invalidation
    return {"message": "Successfully logged out"}
