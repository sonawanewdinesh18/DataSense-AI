"""
Utility functions for the Data Analysis RAG Agent.
Handles file type detection, validation, and common helpers.
"""

import os
import base64
import hashlib

import config


def get_file_extension(filename: str) -> str:
    """Extract and return the lowercase file extension without the dot."""
    return os.path.splitext(filename)[1].lower().lstrip(".")


def is_supported_file(filename: str) -> bool:
    """Check if the file type is supported."""
    ext = get_file_extension(filename)
    return ext in config.SUPPORTED_FILE_TYPES


def validate_file_size(file_size_bytes: int) -> bool:
    """Check if file size is within the allowed limit."""
    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    return file_size_bytes <= max_bytes


def generate_file_hash(content: bytes) -> str:
    """Generate an MD5 hash of file content for deduplication."""
    return hashlib.md5(content).hexdigest()


def format_file_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable file size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def sanitize_namespace(name: str) -> str:
    """
    Create a valid Pinecone namespace from a filename.
    Removes special characters and limits length.
    """
    sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return sanitized[:63].lower()


def get_file_icon(filename: str) -> str:
    """Return an emoji icon based on file type."""
    ext = get_file_extension(filename)
    icons = {
        "csv": "📄",
        "xlsx": "📊",
        "xls": "📊",
        "pdf": "📕",
        "txt": "📝",
    }
    return icons.get(ext, "📎")


def check_api_keys() -> dict:
    """
    Verify that required API keys are configured.
    Returns a dict with key names and their status.
    """
    return {
        "Google Gemini API": bool(config.GOOGLE_API_KEY and config.GOOGLE_API_KEY != "your_google_api_key_here"),
        "Pinecone API": bool(config.PINECONE_API_KEY and config.PINECONE_API_KEY != "your_pinecone_api_key_here"),
    }


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to a maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def get_image_base64(image_path: str) -> str:
    """Read a local image and return its base64 data URI."""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = get_file_extension(image_path)
    mime = "image/png" if ext == "png" else f"image/{ext}"
    return f"data:{mime};base64,{encoded}"
