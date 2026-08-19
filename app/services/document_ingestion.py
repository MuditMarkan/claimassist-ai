import argparse
import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError


# ---------------------------------------------------------------------------
# Safety limits
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PDF_PAGES = 100
MAX_EXTRACTED_CHARACTERS = 2_000_000

ALLOWED_EXTENSIONS = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DocumentIngestionError(ValueError):
    """Raised when a document cannot be ingested safely."""


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DocumentPage(BaseModel):
    """One traceable page or text section."""

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text: str


class IngestedDocument(BaseModel):
    """Validated result returned by document ingestion."""

    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(min_length=1)
    media_type: str = Field(min_length=1)

    # SHA-256 of normalized textual content for TXT/MD,
    # or original PDF bytes for PDF.
    sha256: str = Field(min_length=64, max_length=64)

    file_size: int = Field(gt=0)
    page_count: int = Field(ge=1)
    total_characters: int = Field(ge=1)

    pages: list[DocumentPage]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def calculate_sha256(file_path: Path) -> str:
    """
    Calculate SHA-256 from the original file bytes.

    This function is primarily used for PDFs where the exact uploaded
    document bytes are important.
    """

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def calculate_text_sha256(text: str) -> str:
    """
    Calculate SHA-256 from normalized UTF-8 text.

    This avoids platform-specific newline differences such as:
        Windows: \\r\\n
        Linux:   \\n
    """

    normalized_text = clean_extracted_text(text)

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_extracted_text(text: str) -> str:
    """
    Normalize extracted text without changing its meaning.

    Normalizations:
    - Remove NULL characters.
    - Convert Windows line endings to Unix line endings.
    - Convert old Mac line endings to Unix line endings.
    - Remove leading/trailing whitespace.
    """

    return (
        text
        .replace("\x00", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

def validate_file(file_path: Path) -> tuple[int, str]:
    """
    Validate file existence, extension, and size.
    """

    if not file_path.exists():
        raise DocumentIngestionError(
            f"Document does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise DocumentIngestionError(
            "The supplied path is not a file."
        )

    extension = file_path.suffix.casefold()

    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(
            sorted(ALLOWED_EXTENSIONS)
        )

        raise DocumentIngestionError(
            f"Unsupported document extension. "
            f"Allowed: {allowed}"
        )

    file_size = file_path.stat().st_size

    if file_size == 0:
        raise DocumentIngestionError(
            "The document is empty."
        )

    if file_size > MAX_FILE_SIZE_BYTES:
        raise DocumentIngestionError(
            "The document exceeds the 10 MB file-size limit."
        )

    return file_size, ALLOWED_EXTENSIONS[extension]


# ---------------------------------------------------------------------------
# Text / Markdown ingestion
# ---------------------------------------------------------------------------

def extract_text_file(
    file_path: Path,
) -> list[DocumentPage]:
    """
    Read a UTF-8 text or Markdown document.
    """

    try:
        text = file_path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError as error:
        raise DocumentIngestionError(
            "Text documents must use UTF-8 encoding."
        ) from error

    cleaned_text = clean_extracted_text(text)

    if not cleaned_text:
        raise DocumentIngestionError(
            "The document contains no usable text."
        )

    if len(cleaned_text) > MAX_EXTRACTED_CHARACTERS:
        raise DocumentIngestionError(
            "The extracted text exceeds the safety limit."
        )

    return [
        DocumentPage(
            page_number=1,
            text=cleaned_text,
        )
    ]


# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------

def extract_pdf_pages(
    file_path: Path,
) -> list[DocumentPage]:
    """
    Extract traceable text from every PDF page.
    """

    try:
        reader = PdfReader(
            str(file_path),
            strict=False,
        )

    except PdfReadError as error:
        raise DocumentIngestionError(
            "The PDF is malformed or unreadable."
        ) from error

    except Exception as error:
        raise DocumentIngestionError(
            "The PDF could not be opened safely."
        ) from error

    if reader.is_encrypted:
        raise DocumentIngestionError(
            "Password-protected PDFs are not supported."
        )

    if len(reader.pages) == 0:
        raise DocumentIngestionError(
            "The PDF contains no pages."
        )

    if len(reader.pages) > MAX_PDF_PAGES:
        raise DocumentIngestionError(
            f"The PDF exceeds the "
            f"{MAX_PDF_PAGES}-page limit."
        )

    pages: list[DocumentPage] = []

    total_characters = 0

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        try:
            extracted_text = (
                page.extract_text() or ""
            )

        except Exception as error:
            raise DocumentIngestionError(
                f"Text extraction failed on "
                f"page {page_number}."
            ) from error

        cleaned_text = clean_extracted_text(
            extracted_text
        )

        total_characters += len(cleaned_text)

        if total_characters > MAX_EXTRACTED_CHARACTERS:
            raise DocumentIngestionError(
                "The extracted PDF text exceeds "
                "the safety limit."
            )

        pages.append(
            DocumentPage(
                page_number=page_number,
                text=cleaned_text,
            )
        )

    # A PDF may contain pages but no extractable text.
    # This normally means it is scanned/image-only.
    if not any(page.text for page in pages):
        raise DocumentIngestionError(
            "The PDF has no extractable text. "
            "It may require OCR or the later "
            "ColQwen visual-retrieval module."
        )

    return pages


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest_document(
    file_path: Path,
) -> IngestedDocument:
    """
    Validate and extract a policy document.

    Supported:
        .txt
        .md
        .pdf
    """

    file_path = file_path.resolve()

    file_size, media_type = validate_file(
        file_path
    )

    extension = file_path.suffix.casefold()

    # ---------------------------------------------------------------
    # PDF
    # ---------------------------------------------------------------

    if extension == ".pdf":
        pages = extract_pdf_pages(
            file_path
        )

        # For PDFs, preserve the exact uploaded-file hash.
        document_hash = calculate_sha256(
            file_path
        )

    # ---------------------------------------------------------------
    # TXT / Markdown
    # ---------------------------------------------------------------

    else:
        pages = extract_text_file(
            file_path
        )

        # For text documents, hash the normalized
        # extracted content rather than platform-specific
        # raw file bytes.
        normalized_text = "\n".join(
            page.text
            for page in pages
        )

        document_hash = calculate_text_sha256(
            normalized_text
        )

    # ---------------------------------------------------------------
    # Calculate total characters
    # ---------------------------------------------------------------

    total_characters = sum(
        len(page.text)
        for page in pages
    )

    # ---------------------------------------------------------------
    # Return validated document
    # ---------------------------------------------------------------

    return IngestedDocument(
        source_name=file_path.name,
        media_type=media_type,
        sha256=document_hash,
        file_size=file_size,
        page_count=len(pages),
        total_characters=total_characters,
        pages=pages,
    )


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Safely inspect a ClaimAssist "
            "policy document."
        )
    )

    parser.add_argument(
        "document",
        type=Path,
        help=(
            "Path to a TXT, Markdown, "
            "or PDF policy."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run document ingestion from the command line.
    """

    args = parse_arguments()

    document = ingest_document(
        args.document
    )

    print(
        "Document ingested successfully"
    )

    print(
        f"Source: {document.source_name}"
    )

    print(
        f"Media type: {document.media_type}"
    )

    print(
        f"SHA-256: {document.sha256}"
    )

    print(
        f"File size: {document.file_size} bytes"
    )

    print(
        f"Pages: {document.page_count}"
    )

    print(
        f"Characters: {document.total_characters}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()