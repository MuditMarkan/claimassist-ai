"""
ClaimAssist v1.1 — FastAPI application.

Policy-first flow
-----------------
1.  Company uploads policy PDF  → POST /policies/upload
2.  AI extracts policy facts    → POST /policies/{id}/extract   (auto after upload)
3.  Reviewer verifies fields    → GET  /policies/{id}/verify
4.  Reviewer activates policy   → POST /policies/{id}/activate
5.  Claim submission unlocked   → POST /claims/analyze

Routes
------
GET  /                              Home (claim form — disabled if no active policy)
GET  /policies                      Policy library
POST /policies/upload               Upload + extract policy PDF
GET  /policies/{id}/verify          Verification page for extracted facts
POST /policies/{id}/activate        Mark policy ACTIVE_AND_VERIFIED
POST /claims/analyze                Submit claim (blocked without active policy)
GET  /claims                        Claim list
GET  /claims/{id}                   Claim detail / result

GET  /api/health                    Liveness check
GET  /api/policies                  JSON policy list
"""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.database import (
    DATABASE_PATH,
    EVIDENCE_DIRECTORY,
    POLICIES_DIRECTORY,
    initialize_database,
)
from app.rag.vector_store import CHROMA_DIRECTORY
from app.repositories import (
    DEFAULT_ORG_ID,
    activate_policy_version,
    attach_claim_document,
    create_claim,
    create_policy_document,
    create_policy_version,
    delete_policy_version,
    get_active_policy_version,
    get_claim,
    get_latest_decision,
    get_policy_extraction,
    get_policy_version,
    list_claims,
    list_policy_versions,
    resolve_policy_version_for_date,
    save_decision,
    save_policy_extraction,
    update_claim_status,
    update_policy_index_status,
    update_policy_version_status,
)
from app.services.comparison_engine import compare_claim_to_policy
from app.services.document_ingestion import ingest_document
from app.services.policy_extractor import extract_policy_facts, pages_to_text
from app.services.upload_handler import (
    UploadError,
    save_evidence_upload,
    save_policy_upload,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

APP_DIRECTORY        = Path(__file__).resolve().parent
TEMPLATES_DIRECTORY  = APP_DIRECTORY / "templates"
STATIC_DIRECTORY     = APP_DIRECTORY / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIRECTORY))


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    initialize_database(DATABASE_PATH)
    yield


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_application() -> FastAPI:
    application = FastAPI(
        title="ClaimAssist",
        description="Evidence-grounded insurance claim review — v1.1",
        version="1.1.0",
        lifespan=lifespan,
    )
    application.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIRECTORY)),
        name="static",
    )
    return application


app = create_application()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_active_policy() -> bool:
    return get_active_policy_version(DEFAULT_ORG_ID) is not None


def _render(request: Request, template: str, ctx: dict, status: int = 200) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name=template,
        context=ctx,
        status_code=status,
    )


# ===========================================================================
# HOME
# ===========================================================================

@app.get("/", response_class=HTMLResponse, name="home_page")
async def home_page(request: Request) -> HTMLResponse:
    active_policy = get_active_policy_version(DEFAULT_ORG_ID)
    recent_claims = list_claims(DEFAULT_ORG_ID)[:5]
    return _render(request, "index.html", {
        "active_policy":  active_policy,
        "recent_claims":  recent_claims,
        "error":          None,
        "success":        None,
    })


# ===========================================================================
# POLICY LIBRARY
# ===========================================================================

@app.get("/policies", response_class=HTMLResponse, name="policies_page")
async def policies_page(request: Request) -> HTMLResponse:
    versions = list_policy_versions(DEFAULT_ORG_ID)
    return _render(request, "policies.html", {
        "versions": versions,
        "error":    None,
        "success":  None,
    })


# ---------------------------------------------------------------------------
# Upload policy PDF → save file → ingest → index → extract facts
# ---------------------------------------------------------------------------

@app.post("/policies/upload", response_class=HTMLResponse, name="upload_policy")
async def upload_policy(
    request: Request,
    policy_name:    str        = Form(..., min_length=1, max_length=200),
    version_label:  str        = Form(default="v1", max_length=50),
    policy_file:    UploadFile = File(...),
) -> HTMLResponse:
    # ── Save file to disk ──
    try:
        safe_name, stored_path, file_size, sha256 = \
            await save_policy_upload(policy_file, POLICIES_DIRECTORY)
    except UploadError as err:
        return _render(request, "policies.html", {
            "versions": list_policy_versions(DEFAULT_ORG_ID),
            "error":    str(err),
            "success":  None,
        }, 422)

    # ── Ingest PDF to get page count ──
    try:
        ingested = await asyncio.to_thread(
            lambda: ingest_document(Path(stored_path))
        )
        page_count = ingested.page_count
    except Exception:
        page_count = 0

    # ── Database records ──
    doc_id = create_policy_document(
        original_filename=policy_file.filename or safe_name,
        stored_path=stored_path,
        sha256=sha256,
        file_size=file_size,
        page_count=page_count,
    )
    version_id = create_policy_version(
        policy_document_id=doc_id,
        policy_name=policy_name.strip(),
        version_label=version_label.strip(),
    )

    # ── Index into ChromaDB ──
    update_policy_version_status(version_id, "EXTRACTION_IN_PROGRESS")
    update_policy_index_status(version_id, "INDEXING")

    index_ok    = False
    index_msg   = ""
    try:
        from app.rag.vector_store import index_policy_document as _idx
        await asyncio.to_thread(
            lambda: _idx(Path(stored_path), persist_directory=CHROMA_DIRECTORY)
        )
        update_policy_index_status(version_id, "INDEXED")
        index_ok  = True
    except Exception as idx_err:
        update_policy_index_status(version_id, "FAILED")
        index_msg = f" (indexing failed: {type(idx_err).__name__}: {idx_err})"

    # ── Extract policy facts with AI ──
    extraction_ok  = False
    extraction_msg = ""
    try:
        ingested2 = await asyncio.to_thread(
            lambda: ingest_document(Path(stored_path))
        )
        policy_text_content = pages_to_text(
            [p.model_dump() for p in ingested2.pages]
        )
        await _run_extraction(version_id, policy_text_content)
        extraction_ok = True
    except Exception as ext_err:
        update_policy_version_status(version_id, "FAILED_SAFE")
        extraction_msg = f" Extraction failed: {type(ext_err).__name__}: {ext_err}"

    if extraction_ok:
        success = (
            f'"{policy_name}" uploaded and extracted successfully{index_msg}. '
            f"Please verify the extracted facts before activating."
        )
    else:
        success = (
            f'"{policy_name}" uploaded{index_msg}.{extraction_msg} '
            f"You can retry extraction from the policy detail page."
        )

    return _render(request, "policies.html", {
        "versions": list_policy_versions(DEFAULT_ORG_ID),
        "error":    None,
        "success":  success,
    })


# ---------------------------------------------------------------------------
# Re-extract policy facts (retry after FAILED_SAFE)
# ---------------------------------------------------------------------------

@app.post("/policies/{version_id}/extract",
          response_class=HTMLResponse, name="reextract_policy")
async def reextract_policy(
    request: Request,
    version_id: int,
) -> HTMLResponse:
    try:
        version = get_policy_version(version_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Policy version not found.")

    stored_path = version.get("stored_path")
    if not stored_path or not Path(stored_path).exists():
        return _render(request, "policy_detail.html", {
            "version":    version,
            "extraction": None,
            "error":      "Stored policy file not found on disk.",
            "success":    None,
        }, 422)

    update_policy_version_status(version_id, "EXTRACTION_IN_PROGRESS")

    try:
        ingested = await asyncio.to_thread(
            lambda: ingest_document(Path(stored_path))
        )
        policy_text = pages_to_text([p.model_dump() for p in ingested.pages])
        await _run_extraction(version_id, policy_text)
    except Exception as ext_err:
        update_policy_version_status(version_id, "FAILED_SAFE")
        version = get_policy_version(version_id)
        return _render(request, "policy_detail.html", {
            "version":    version,
            "extraction": get_policy_extraction(version_id),
            "error":      f"Extraction failed: {type(ext_err).__name__}: {ext_err}",
            "success":    None,
        }, 500)

    version    = get_policy_version(version_id)
    extraction = get_policy_extraction(version_id)
    return _render(request, "policy_detail.html", {
        "version":    version,
        "extraction": extraction,
        "error":      None,
        "success":    "Policy facts extracted successfully. Review and activate below.",
    })


# ---------------------------------------------------------------------------
# Verify extracted policy facts
# ---------------------------------------------------------------------------

@app.get("/policies/{version_id}/verify",
         response_class=HTMLResponse, name="verify_policy_page")
async def verify_policy_page(
    request: Request,
    version_id: int,
) -> HTMLResponse:
    try:
        version    = get_policy_version(version_id)
        extraction = get_policy_extraction(version_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Policy version not found.")

    return _render(request, "policy_detail.html", {
        "version":    version,
        "extraction": extraction,
        "error":      None,
        "success":    None,
    })


# ---------------------------------------------------------------------------
# Activate policy version
# ---------------------------------------------------------------------------

@app.post("/policies/{version_id}/activate",
          response_class=HTMLResponse, name="activate_policy")
async def activate_policy(
    request:        Request,
    version_id:     int,
    effective_from: str = Form(default=""),
    effective_to:   str = Form(default=""),
) -> HTMLResponse:
    try:
        version = get_policy_version(version_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Policy version not found.")

    extraction = get_policy_extraction(version_id)
    if not extraction:
        return _render(request, "policy_detail.html", {
            "version":    version,
            "extraction": extraction,
            "error":      "Cannot activate: policy facts have not been extracted yet.",
            "success":    None,
        }, 422)

    activate_policy_version(
        version_id,
        effective_from=effective_from.strip() or None,
        effective_to=effective_to.strip()     or None,
    )

    return _render(request, "policies.html", {
        "versions": list_policy_versions(DEFAULT_ORG_ID),
        "error":    None,
        "success":  (
            f'Policy "{version["policy_name"]}" is now ACTIVE_AND_VERIFIED. '
            f"Claims can now be submitted."
        ),
    })


# ---------------------------------------------------------------------------
# Internal helper — run extraction and persist for any policy version
# ---------------------------------------------------------------------------

async def _run_extraction(version_id: int, policy_text: str) -> None:
    """Extract policy facts from text and save to DB. Raises on failure."""
    facts = await asyncio.to_thread(
        lambda: extract_policy_facts(policy_text)
    )
    save_policy_extraction(
        policy_version_id=version_id,
        policy_number=facts.policy_number,
        named_insured=facts.named_insured,
        effective_from=facts.effective_from,
        effective_to=facts.effective_to,
        vehicle_year=facts.vehicle_year,
        vehicle_make_model=facts.vehicle_make_model,
        vehicle_vin=facts.vehicle_vin,
        vehicle_plate=facts.vehicle_plate,
        coverage_types=facts.coverage_types,
        coverage_limit=facts.coverage_limit,
        deductible=facts.deductible,
        valuation_basis=facts.valuation_basis,
        exclusions=facts.exclusions,
        required_documents=facts.required_documents,
        territory=facts.territory,
        permitted_use=facts.permitted_use,
        raw_extraction_json=facts.model_dump_json(),
    )
    update_policy_version_status(version_id, "NEEDS_VERIFICATION")


# ---------------------------------------------------------------------------
# Upload policy as plain text (paste instead of file)
# ---------------------------------------------------------------------------

@app.post("/policies/upload-text", response_class=HTMLResponse, name="upload_policy_text")
async def upload_policy_text(
    request:       Request,
    policy_name:   str = Form(..., min_length=1, max_length=200),
    version_label: str = Form(default="v1", max_length=50),
    policy_text:   str = Form(..., min_length=50, max_length=100_000),
) -> HTMLResponse:
    """
    Accept policy content pasted as plain text.
    Saves a .txt file to disk, indexes it, and runs AI extraction.
    """
    import hashlib, uuid as _uuid

    cleaned_text = policy_text.strip()
    sha256       = hashlib.sha256(cleaned_text.encode()).hexdigest()
    safe_name    = f"{_uuid.uuid4().hex}_{policy_name[:40].replace(' ','_')}.txt"
    stored_path  = POLICIES_DIRECTORY / safe_name

    POLICIES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stored_path.write_text(cleaned_text, encoding="utf-8")
    file_size = stored_path.stat().st_size

    # DB records
    doc_id = create_policy_document(
        original_filename=f"{policy_name}.txt",
        stored_path=str(stored_path.resolve()),
        sha256=sha256,
        file_size=file_size,
        page_count=1,
    )
    version_id = create_policy_version(
        policy_document_id=doc_id,
        policy_name=policy_name.strip(),
        version_label=version_label.strip(),
    )

    # Index into ChromaDB
    update_policy_version_status(version_id, "EXTRACTION_IN_PROGRESS")
    update_policy_index_status(version_id, "INDEXING")
    index_msg = ""
    try:
        from app.rag.vector_store import index_policy_document as _idx
        await asyncio.to_thread(
            lambda: _idx(stored_path, persist_directory=CHROMA_DIRECTORY)
        )
        update_policy_index_status(version_id, "INDEXED")
    except Exception as idx_err:
        update_policy_index_status(version_id, "FAILED")
        index_msg = f" (indexing failed: {type(idx_err).__name__})"

    # AI extraction
    extraction_ok = False
    extraction_msg = ""
    try:
        await _run_extraction(version_id, cleaned_text)
        extraction_ok = True
    except Exception as ext_err:
        update_policy_version_status(version_id, "FAILED_SAFE")
        extraction_msg = f" Extraction failed: {type(ext_err).__name__}: {ext_err}"

    if extraction_ok:
        success = (
            f'"{policy_name}" saved and extracted successfully{index_msg}. '
            "Please verify the extracted facts before activating."
        )
    else:
        success = (
            f'"{policy_name}" saved{index_msg}.{extraction_msg} '
            "You can retry extraction from the policy detail page."
        )

    return _render(request, "policies.html", {
        "versions": list_policy_versions(DEFAULT_ORG_ID),
        "error":    None,
        "success":  success,
    })


# ---------------------------------------------------------------------------
# Delete a policy version
# ---------------------------------------------------------------------------

@app.post("/policies/{version_id}/delete",
          response_class=HTMLResponse, name="delete_policy")
async def delete_policy(
    request: Request,
    version_id: int,
) -> HTMLResponse:
    try:
        version = get_policy_version(version_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Policy version not found.")

    try:
        delete_policy_version(version_id)
    except ValueError as err:
        return _render(request, "policies.html", {
            "versions": list_policy_versions(DEFAULT_ORG_ID),
            "error":    str(err),
            "success":  None,
        }, 422)

    return _render(request, "policies.html", {
        "versions": list_policy_versions(DEFAULT_ORG_ID),
        "error":    None,
        "success":  f'Policy version "{version["policy_name"]}" ({version["version_label"]}) was deleted.',
    })


# ===========================================================================
# CLAIM SUBMISSION
# ===========================================================================

@app.post("/claims/analyze", response_class=HTMLResponse, name="analyze_claim")
async def analyze_claim(
    request:        Request,
    claim_text:     str        = Form(..., min_length=1, max_length=20_000),
    loss_date:      str        = Form(default=""),
    claimant_name:  str        = Form(default=""),
    driver_name:    str        = Form(default=""),
    vehicle_vin:    str        = Form(default=""),
    evidence_files: list[UploadFile] = File(default=[]),
) -> HTMLResponse:

    # ── Policy-first gate ─────────────────────────────────────────────────
    active_policy = get_active_policy_version(DEFAULT_ORG_ID)
    if not active_policy:
        return _render(request, "index.html", {
            "active_policy": None,
            "recent_claims": list_claims(DEFAULT_ORG_ID)[:5],
            "error": (
                "No active policy found. "
                "Upload and verify a policy before submitting a claim."
            ),
            "success": None,
        }, 422)

    # ── Resolve policy version for loss date ─────────────────────────────
    loss_date_clean = loss_date.strip()
    if loss_date_clean:
        policy_version = resolve_policy_version_for_date(
            loss_date_clean, DEFAULT_ORG_ID
        )
        if not policy_version:
            return _render(request, "index.html", {
                "active_policy": active_policy,
                "recent_claims": list_claims(DEFAULT_ORG_ID)[:5],
                "error": (
                    f"No active policy version covers the loss date "
                    f"{loss_date_clean}. Check the policy effective dates."
                ),
                "success": None,
            }, 422)
    else:
        policy_version = active_policy

    # ── Save evidence files ───────────────────────────────────────────────
    uploaded_names: list[str] = []
    saved_evidence: list[tuple] = []
    real_files = [f for f in evidence_files if f.filename]
    for upload in real_files:
        try:
            result = await save_evidence_upload(upload, EVIDENCE_DIRECTORY)
            saved_evidence.append(result)
            uploaded_names.append(upload.filename or "")
        except UploadError as err:
            return _render(request, "index.html", {
                "active_policy": active_policy,
                "recent_claims": list_claims(DEFAULT_ORG_ID)[:5],
                "error": f"File rejected ({upload.filename}): {err}",
                "success": None,
            }, 422)

    # ── Create claim record ───────────────────────────────────────────────
    claim_id = create_claim(
        raw_claim_text=claim_text,
        loss_date=loss_date_clean or None,
        policy_version_id=policy_version["id"],
    )

    # Attach uploaded evidence files
    for safe_name, stored_path, mime, size, sha in saved_evidence:
        attach_claim_document(
            claim_id=claim_id,
            original_filename=safe_name,
            stored_path=stored_path,
            mime_type=mime,
            file_size=size,
            sha256=sha,
        )

    # ── Extract claim facts with AI ───────────────────────────────────────
    update_claim_status(claim_id, "EXTRACTING")
    try:
        from app.claim_analyzer import extract_claim_facts as _ecf
        claim_facts = await asyncio.to_thread(
            lambda: _ecf(claim_text)
        )
        # Merge form fields into extracted facts (user-supplied overrides)
        if claimant_name.strip() and not claim_facts.claimant_name:
            claim_facts = claim_facts.model_copy(
                update={"claimant_name": claimant_name.strip()}
            )
        if driver_name.strip() and not claim_facts.driver_name:
            claim_facts = claim_facts.model_copy(
                update={"driver_name": driver_name.strip()}
            )
        if vehicle_vin.strip() and not claim_facts.vehicle_vin:
            claim_facts = claim_facts.model_copy(
                update={"vehicle_vin": vehicle_vin.strip()}
            )
        if loss_date_clean and not claim_facts.loss_date:
            claim_facts = claim_facts.model_copy(
                update={"loss_date": loss_date_clean}
            )
    except Exception as ext_err:
        update_claim_status(claim_id, "FAILED_SAFE")
        return _render(request, "index.html", {
            "active_policy": active_policy,
            "recent_claims": list_claims(DEFAULT_ORG_ID)[:5],
            "error": (
                f"Claim extraction failed: {type(ext_err).__name__}. "
                "Check that Ollama is running and the model is available."
            ),
            "success": None,
        }, 500)

    # ── Run deterministic comparison ──────────────────────────────────────
    update_claim_status(claim_id, "COMPARING")
    extraction = get_policy_extraction(policy_version["id"])

    result = compare_claim_to_policy(
        policy=policy_version,
        policy_extraction=extraction,
        claim_facts=claim_facts,
    )

    # ── Persist decision ──────────────────────────────────────────────────
    decision_id = save_decision(
        claim_id=claim_id,
        policy_version_id=policy_version["id"],
        outcome=result.outcome,
        eligible_gross=result.eligible_gross,
        deductible_applied=result.deductible_applied,
        estimated_net=result.estimated_net,
        reasons=result.reasons,
        missing_documents=result.missing_documents,
        passed_checks=result.passed_checks,
        failed_checks=result.failed_checks,
        citations=result.citations,
    )

    decision = get_latest_decision(claim_id)

    return _render(request, "claim_result.html", {
        "claim_id":       claim_id,
        "claim_facts":    claim_facts.model_dump(),
        "policy_version": dict(policy_version),
        "result":         result.model_dump(),
        "decision":       decision,
        "uploaded_files": uploaded_names,
    })


# ===========================================================================
# CLAIM LIST & DETAIL
# ===========================================================================

@app.get("/claims", response_class=HTMLResponse, name="claims_list")
async def claims_list(request: Request) -> HTMLResponse:
    claims = list_claims(DEFAULT_ORG_ID)
    return _render(request, "claims_list.html", {"claims": claims})


@app.get("/claims/{claim_id}", response_class=HTMLResponse, name="claim_detail")
async def claim_detail(request: Request, claim_id: int) -> HTMLResponse:
    try:
        claim    = get_claim(claim_id)
        decision = get_latest_decision(claim_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Claim not found.")

    policy_version = None
    extraction     = None
    if claim.get("policy_version_id"):
        try:
            policy_version = get_policy_version(claim["policy_version_id"])
            extraction     = get_policy_extraction(claim["policy_version_id"])
        except LookupError:
            pass

    return _render(request, "claim_result.html", {
        "claim_id":       claim_id,
        "claim_facts":    {},
        "policy_version": dict(policy_version) if policy_version else {},
        "result":         {},
        "decision":       decision,
        "uploaded_files": [],
    })


# ===========================================================================
# JSON API
# ===========================================================================

@app.get("/api/policies", name="api_list_policies")
async def api_list_policies() -> JSONResponse:
    return JSONResponse(content=list_policy_versions(DEFAULT_ORG_ID))


@app.get("/api/health", name="health_check")
async def health_check() -> dict[str, Any]:
    active = get_active_policy_version(DEFAULT_ORG_ID)
    return {
        "status":          "ok",
        "application":     "ClaimAssist",
        "version":         "1.1.0",
        "database_ready":  DATABASE_PATH.exists(),
        "chroma_ready":    CHROMA_DIRECTORY.exists(),
        "active_policy":   active["policy_name"] if active else None,
        "policy_gate_open": active is not None,
    }
