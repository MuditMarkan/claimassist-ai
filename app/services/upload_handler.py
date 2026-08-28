"""
Secure file-upload helpers for policy PDFs and claim evidence.

Handles saving uploaded bytes from FastAPI's UploadFile objects to
disk with safe filenames, size limits, and MIME validation.
"""

import hashlib
import mimetypes
import re
import uuid
from pathlib import Path

from fastapi import UploadFile

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_POLICY_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_EVIDENCE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Allowed MIME types for policy uploads
ALLOWED_POLICY_MIMES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
}

# Allowed MIME types for user evidence uploads (docs + images)
ALLOWED_EVIDENCE_MIMES = {
    "application/pdf",
    "text/plain",
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}

# Allowed extensions mapped to MIME types (fallback if MIME header absent)
ALLOWED_POLICY_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
}

ALLOWED_EVIDENCE_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UploadError(ValueError):
    """Raised for rejected or unsafe uploads."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(original_name: str) -> str:
    """
    Produce a filesystem-safe version of an uploaded filename.

    - Strips path separators and null bytes.
    - Replaces runs of non-alphanumeric chars (except dots/hyphens) with _.
    - Prepends a UUID4 to guarantee uniqueness and prevent overwrites.
    """
    name = Path(original_name).name  # strip directory components
    name = name.replace("\x00", "")
    name = re.sub(r"[^\w.\-]", "_", name)
    name = name.strip("._")
    name = name or "upload"
    return f"{uuid.uuid4().hex}_{name}"


def _detect_mime(
    file: UploadFile,
    allowed_extensions: dict[str, str],
) -> str:
    """
    Resolve the MIME type for an upload.

    Uses the browser-supplied content_type first; falls back to
    extension sniffing when the browser sends application/octet-stream.
    """
    content_type = (file.content_type or "").lower().split(";")[0].strip()

    if content_type and content_type != "application/octet-stream":
        return content_type

    # Fallback: sniff from extension
    ext = Path(file.filename or "").suffix.casefold()
    if ext in allowed_extensions:
        return allowed_extensions[ext]

    guessed, _ = mimetypes.guess_type(file.filename or "")
    return guessed or "application/octet-stream"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def save_policy_upload(
    file: UploadFile,
    destination_directory: Path,
) -> tuple[str, str, int, str]:
    """
    Validate and persist an uploaded policy document.

    Returns:
        (safe_filename, stored_path, file_size, sha256)
    Raises:
        UploadError for rejected files.
    """
    if not file.filename:
        raise UploadError("No filename was provided.")

    ext = Path(file.filename).suffix.casefold()
    if ext not in ALLOWED_POLICY_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_POLICY_EXTENSIONS))
        raise UploadError(
            f"Policy files must be one of: {allowed}"
        )

    data = await file.read()

    if not data:
        raise UploadError("The uploaded file is empty.")

    if len(data) > MAX_POLICY_SIZE_BYTES:
        raise UploadError(
            "Policy file exceeds the 10 MB size limit."
        )

    mime = _detect_mime(file, ALLOWED_POLICY_EXTENSIONS)
    if mime not in ALLOWED_POLICY_MIMES:
        raise UploadError(
            f"Unsupported policy file type: {mime}"
        )

    safe_name = _safe_filename(file.filename)
    destination_directory.mkdir(parents=True, exist_ok=True)
    dest_path = destination_directory / safe_name

    dest_path.write_bytes(data)

    return (
        safe_name,
        str(dest_path.resolve()),
        len(data),
        _sha256_bytes(data),
    )


async def save_evidence_upload(
    file: UploadFile,
    destination_directory: Path,
) -> tuple[str, str, str, int, str]:
    """
    Validate and persist an uploaded evidence document or image.

    Returns:
        (safe_filename, stored_path, mime_type, file_size, sha256)
    Raises:
        UploadError for rejected files.
    """
    if not file.filename:
        raise UploadError("No filename was provided.")

    ext = Path(file.filename).suffix.casefold()
    if ext not in ALLOWED_EVIDENCE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EVIDENCE_EXTENSIONS))
        raise UploadError(
            f"Evidence files must be one of: {allowed}"
        )

    data = await file.read()

    if not data:
        raise UploadError("The uploaded file is empty.")

    if len(data) > MAX_EVIDENCE_SIZE_BYTES:
        raise UploadError(
            "Evidence file exceeds the 10 MB size limit."
        )

    mime = _detect_mime(file, ALLOWED_EVIDENCE_EXTENSIONS)
    if mime not in ALLOWED_EVIDENCE_MIMES:
        raise UploadError(
            f"Unsupported evidence file type: {mime}"
        )

    safe_name = _safe_filename(file.filename)
    destination_directory.mkdir(parents=True, exist_ok=True)
    dest_path = destination_directory / safe_name

    dest_path.write_bytes(data)

    return (
        safe_name,
        str(dest_path.resolve()),
        mime,
        len(data),
        _sha256_bytes(data),
    )
