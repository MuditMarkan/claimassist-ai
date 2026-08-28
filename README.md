# ClaimAssist

An evidence-based, AI-assisted insurance claim review application.
Built entirely with a local stack — no cloud APIs, no paid services,
everything runs on your own machine.

> **Important:** ClaimAssist produces a *preliminary assessment only.*
> It is not a final coverage determination, approval, denial, settlement,
> or payment authorization. Every result must be reviewed and authorized
> by a qualified human adjuster before any action is taken.

---

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [How It Works](#2-how-it-works)
3. [Technology Stack](#3-technology-stack)
4. [Project Structure](#4-project-structure)
5. [Installation](#5-installation)
6. [First-Run Walkthrough](#6-first-run-walkthrough)
7. [Running the App](#7-running-the-app)
8. [Running Tests](#8-running-tests)
9. [API Reference](#9-api-reference)
10. [Configuration](#10-configuration)
11. [Known Limitations](#11-known-limitations)
12. [Production Gate](#12-production-gate)

---

## 1. What It Does

ClaimAssist follows a strict **policy-first** workflow:

```
Company uploads policy PDF
    ↓  AI reads it and extracts structured facts
    ↓  Reviewer confirms the extracted fields
    ↓  Policy becomes ACTIVE_AND_VERIFIED
    ↓
User submits a claim PDF + supporting documents + photos
    ↓  AI reads the claim and extracts structured facts
    ↓  Python rules compare claim facts against policy facts
    ↓  Preliminary calculation:  eligible = min(claim, limit) − deductible
    ↓
Result:  LIKELY_COVERED  /  LIKELY_NOT_COVERED  /  REVIEW_REQUIRED
         with full citations, rule check list, and calculation breakdown
    ↓
Human adjuster reviews and records the authorized final decision
```

Claim submission is **blocked** until a verified policy exists.
This is enforced in both the UI (disabled button) and the API (HTTP 422).

---

## 2. How It Works

### The AI's role — extraction only

The language model (`qwen3.5:4b` via Ollama) does exactly one thing:
reads unstructured PDF text and converts it into a structured JSON object.

| Step | Who does it | Tool |
|---|---|---|
| Read policy PDF and extract facts | AI (Qwen) | Ollama + Pydantic |
| Read claim PDF and extract facts | AI (Qwen) | Ollama + Pydantic |
| Index policy into vector database | Embeddings | nomic-embed-text + ChromaDB |
| Retrieve relevant policy passages | Semantic search | ChromaDB + LangChain |
| Compare claim facts to policy facts | Pure Python | comparison_engine.py |
| Calculate eligible amount | Pure Python | `min(claim, limit) − deductible` |
| Make the preliminary decision | Pure Python | 16 deterministic rules |
| Authorize the final decision | Human adjuster | Review queue |

The AI **never** makes a coverage decision. Every decision comes from
deterministic Python rules that produce the same output every time
given the same inputs.

### Why this matters

- **Auditable** — every decision has a named Python rule behind it
- **Reliable** — rules don't hallucinate or change between runs
- **Safe** — the model cannot approve or deny a claim even if prompted to
- **Explainable** — every result shows exactly which rules passed and failed

---

## 3. Technology Stack

### Web Framework
| Tool | Version | Purpose |
|---|---|---|
| **FastAPI** | `>=0.115` | HTTP routes, form handling, request validation |
| **Uvicorn** | `>=0.34` | ASGI server that runs FastAPI |
| **Jinja2** | bundled with FastAPI | HTML template rendering |
| **python-multipart** | `>=0.0.20` | Parses multipart form data for file uploads |

### Data Validation
| Tool | Version | Purpose |
|---|---|---|
| **Pydantic v2** | `>=2.11` | All data models, schema validation, structured LLM output |

Pydantic is used extensively throughout the app — every LLM response is
validated against a strict schema before it touches any business logic.

### AI — Language Model Inference
| Tool | Version | Purpose |
|---|---|---|
| **Ollama** | local service | Runs language models locally, no internet required |
| **qwen3.5:4b** | model | Extracts structured facts from policy and claim PDFs |
| **ollama** (Python SDK) | `>=0.5` | Python client that calls the local Ollama service |

The model is called with `temperature=0` and `num_predict=4096` to produce
deterministic, complete JSON responses.

### AI — Embeddings & Vector Search (RAG)
| Tool | Version | Purpose |
|---|---|---|
| **nomic-embed-text** | local model | Converts policy text chunks into embedding vectors |
| **ChromaDB** | bundled with langchain-chroma | Persistent local vector database |
| **langchain-chroma** | `>=0.2` | Python interface between LangChain and ChromaDB |
| **langchain-ollama** | `>=0.3` | LangChain integration for Ollama embeddings |
| **langchain-text-splitters** | `>=0.3` | Splits policy documents into overlapping chunks |

**How RAG works in this app:**

1. When a policy is uploaded, it is split into 1,000-character chunks
   with 150-character overlap using `RecursiveCharacterTextSplitter`
2. Each chunk is embedded with `nomic-embed-text` and stored in ChromaDB
   with metadata: `chunk_id` (SHA-256), `source_name`, `page_number`, `citation`
3. When a claim is submitted, the claim text is used as a search query
4. ChromaDB returns the most semantically similar policy passages
5. Those passages are included as evidence context in the extraction prompt

### AI — Workflow Orchestration
| Tool | Version | Purpose |
|---|---|---|
| **LangGraph** | `>=1.2` | Stateful multi-step workflow with conditional branching |
| **langgraph-checkpoint-sqlite** | `>=3.1` | Persists workflow state to SQLite for pause/resume |

LangGraph orchestrates the claim analysis as a directed graph:

```
retrieve_evidence_node
    ↓ (evidence found)       ↓ (no evidence)
extract_facts_node        human_review_node
    ↓ (extraction ok)        ↑
validate_evidence_node ───┘ (validation fails)
    ↓ (all checks pass)
apply_rules_node
    ↓
END
```

If a claim reaches `human_review_node`, LangGraph calls `interrupt()`
which pauses execution and saves a checkpoint. The adjuster submits
a response through the UI and the workflow resumes from the exact
same point without repeating any completed steps.

### Database
| Tool | Version | Purpose |
|---|---|---|
| **SQLite** | bundled with Python | Stores all application data |
| **WAL mode** | SQLite pragma | Allows concurrent reads while writing |

Database tables:
- `organizations` — tenant (seeded with one default org for MVP)
- `policy_documents` — immutable uploaded file record with SHA-256
- `policy_versions` — versioned policy with full lifecycle status
- `policy_extractions` — AI-extracted structured facts per version
- `claims` — claim submissions linked to a policy version
- `claim_documents` — uploaded evidence files (versioned)
- `decisions` — comparison results with financials and rule checks
- `human_reviews` — open adjuster review queue
- `workflow_runs` — LangGraph thread tracking
- `audit_events` — append-only log of every state change
- `evaluation_runs` — evaluation harness results

### Document Processing
| Tool | Version | Purpose |
|---|---|---|
| **pypdf** | `>=6.14` | Extracts text from PDF files page by page |
| **hashlib** | Python stdlib | SHA-256 content hashing for every uploaded file |

Safety limits enforced on every upload:
- Max file size: 10 MB
- Max PDF pages: 100
- Max extracted characters: 2,000,000
- Encrypted PDFs: rejected
- Scanned/image PDFs: rejected with clear error (OCR not yet implemented)

### Testing
| Tool | Version | Purpose |
|---|---|---|
| **pytest** | `>=8.3` | Test runner for unit and integration tests |
| **FastAPI TestClient** | bundled | HTTP-level testing without running a server |

---

## 4. Project Structure

```
claimassist-ai/
│
├── app/
│   ├── main.py                     ← FastAPI app, all routes
│   ├── database.py                 ← SQLite schema, connection helpers
│   ├── repositories.py             ← All database read/write functions
│   ├── business_rules.py           ← Shared data models, normalization helpers
│   ├── claim_analyzer.py           ← LLM extraction: claim facts from claim PDF
│   ├── pipeline.py                 ← Legacy non-RAG pipeline (evaluation only)
│   ├── rag_pipeline.py             ← Legacy RAG pipeline (evaluation only)
│   │
│   ├── services/
│   │   ├── policy_extractor.py     ← LLM extraction: policy facts from policy PDF
│   │   ├── comparison_engine.py    ← 16 deterministic comparison rules + §10 calc
│   │   ├── document_ingestion.py   ← Safe PDF/TXT ingestion with limits
│   │   ├── upload_handler.py       ← Secure file upload: MIME check, safe filenames
│   │   └── evidence_validator.py   ← Deterministic evidence reference verification
│   │
│   ├── rag/
│   │   ├── chunker.py              ← Split policy docs into chunks with metadata
│   │   ├── vector_store.py         ← ChromaDB interface: index + search
│   │   └── retriever.py            ← Build evidence bundles from search results
│   │
│   ├── workflow/
│   │   ├── claim_graph.py          ← LangGraph non-resumable graph (5 nodes)
│   │   └── resumable_graph.py      ← LangGraph resumable graph with SQLite checkpoints
│   │
│   ├── templates/
│   │   ├── index.html              ← Claim submission form
│   │   ├── policies.html           ← Policy library + upload (PDF or text)
│   │   ├── policy_detail.html      ← Extracted facts verification + activate
│   │   ├── claim_result.html       ← Decision, calculation, rule checks
│   │   └── claims_list.html        ← Claim history table
│   │
│   └── static/
│       └── styles.css              ← Full CSS design system
│
├── data/
│   ├── claimassist.db              ← SQLite database (auto-created)
│   ├── chroma/                     ← ChromaDB vector store (auto-created)
│   ├── uploads/
│   │   ├── policies/               ← Uploaded policy files
│   │   └── evidence/               ← Uploaded claim evidence files
│   └── policies/
│       └── sample_policy.txt       ← Realistic sample policy for testing
│
├── evaluation/
│   ├── run_evaluation.py           ← Synthetic test harness
│   └── synthetic_claims.json       ← 10 test cases
│
├── tests/                          ← pytest test suite
│
├── requirements.txt
├── IMPROVEMENTS.md                 ← Roadmap and AI enhancement suggestions
└── README.md                       ← This file
```

---

## 5. Installation

### Requirements

- **Python 3.11 or 3.12** (Python 3.10 will also work)
- **Ollama** installed and running — [https://ollama.com](https://ollama.com)
- **Git**
- Windows, macOS, or Linux

### Step 1 — Clone the repository

```bash
git clone <your-repo-url>
cd claimassist-ai
```

### Step 2 — Create a virtual environment

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

If PowerShell blocks the activation script, run this once first:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Pull the required Ollama models

Make sure Ollama is running (`ollama serve` in a separate terminal),
then pull the two models the app uses:

```bash
# Language model — used for policy and claim fact extraction
ollama pull qwen3.5:4b

# Embedding model — used for indexing policy chunks into ChromaDB
ollama pull nomic-embed-text
```

Both models run locally. No API key or internet connection is needed
after the initial download.

---

## 6. First-Run Walkthrough

Start the app, then follow these steps in your browser.

### Step 1 — Upload a policy

Go to **[http://localhost:8000/policies](http://localhost:8000/policies)**

You have two options:

**Option A — Upload a PDF**
- Click "Upload PDF" tab
- Enter a policy name (e.g. `Standard Auto Policy`)
- Select your policy PDF
- Click **Upload & Extract**

**Option B — Paste text**
- Click "Paste Text" tab
- Enter a policy name
- Paste the full policy text
- Click **Save & Extract**

A sample policy is available at `data/policies/sample_policy.txt`
(Northstar Demo Insurance — fictional test fixture).

The AI will extract: policy number, named insured, effective dates,
vehicle details, coverage types, limits, deductibles, required
documents, exclusions, and territory.

This takes **30–90 seconds** depending on your hardware.

### Step 2 — Verify and activate

After extraction, click **Verify** next to the uploaded policy.

You will see all AI-extracted fields. Check them against the source
document. If they are correct, set the effective dates and click
**Verify & Activate**.

The policy status changes to **ACTIVE AND VERIFIED**.

### Step 3 — Submit a claim

Go to **[http://localhost:8000](http://localhost:8000)**

The green banner confirms which policy is active.

Fill in:
- **Claim document** — paste or describe the claim (include the claim
  amount, loss description, and list of submitted documents)
- **Date of loss** — must fall within the policy effective period
- **Claimant name**, **driver name**, **VIN** — optional but improve accuracy
- **Supporting files** — upload PDFs, photos, estimates (optional)

Click **Analyze Claim**. This takes **15–40 seconds**.

### Step 4 — Review the result

You will see:

- The preliminary outcome: **Likely Covered**, **Likely Not Covered**,
  or **Review Required**
- The preliminary calculation: eligible gross, deductible, estimated net
- A full rule check list showing every rule that passed or failed
- Missing required documents (if any)
- Policy citations used

---

## 7. Running the App

Always activate the virtual environment first.

```powershell
# Windows
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

```bash
# macOS / Linux
source .venv/bin/activate
uvicorn app.main:app --reload
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

The `--reload` flag restarts the server automatically when you edit
source files. Remove it in production.

### Verifying the app is running

```bash
curl http://localhost:8000/api/health
```

Expected response:

```json
{
  "status": "ok",
  "application": "ClaimAssist",
  "version": "1.1.0",
  "database_ready": true,
  "chroma_ready": true,
  "active_policy": "Standard Auto Policy",
  "policy_gate_open": true
}
```

---

## 8. Running Tests

```bash
pytest tests/ -v
```

Run a specific test file:

```bash
pytest tests/test_business_rules.py -v
```

Run the evaluation harness (requires Ollama to be running):

```bash
python -m evaluation.run_evaluation
```

Results are saved to `evaluation/results/latest.json`.

---

## 9. API Reference

All routes return HTML unless noted.

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Claim submission form |
| `POST` | `/claims/analyze` | Submit a claim for analysis |
| `GET` | `/claims` | Claim history list |
| `GET` | `/claims/{id}` | Claim detail and result |
| `GET` | `/policies` | Policy library |
| `POST` | `/policies/upload` | Upload a policy PDF or text file |
| `POST` | `/policies/upload-text` | Submit policy as pasted text |
| `GET` | `/policies/{id}/verify` | View extracted facts and activate |
| `POST` | `/policies/{id}/activate` | Mark policy ACTIVE_AND_VERIFIED |
| `POST` | `/policies/{id}/extract` | Retry AI extraction after failure |
| `POST` | `/policies/{id}/delete` | Delete a non-active policy version |
| `GET` | `/api/health` | JSON liveness check |
| `GET` | `/api/policies` | JSON list of all policy versions |

### Policy lifecycle states

```
UPLOADED
  → EXTRACTION_IN_PROGRESS
  → NEEDS_VERIFICATION       ← reviewer must confirm facts
  → ACTIVE_AND_VERIFIED      ← claims can now be submitted
  → SUPERSEDED               ← replaced by a newer version
  → RETIRED                  ← manually retired
  → FAILED_SAFE              ← extraction or validation failed
```

### Claim states

```
DRAFT → SUBMITTED → VALIDATING → EXTRACTING
  → RETRIEVING_POLICY_EVIDENCE → COMPARING
  → LIKELY_COVERED / LIKELY_NOT_COVERED / REVIEW_REQUIRED
  → HUMAN_REVIEWED → CLOSED
  → FAILED_SAFE / FAILED
```

---

## 10. Configuration

All paths are relative to the project root directory.
The app creates all required directories on first startup.

| Path | Purpose |
|---|---|
| `data/claimassist.db` | SQLite database |
| `data/chroma/` | ChromaDB vector store |
| `data/uploads/policies/` | Uploaded policy files |
| `data/uploads/evidence/` | Uploaded claim evidence |
| `data/langgraph_checkpoints.sqlite` | LangGraph workflow checkpoints |

### Changing the Ollama model

Edit `app/services/policy_extractor.py` and `app/claim_analyzer.py`:

```python
MODEL_NAME = "qwen3.5:4b"   # change this to any Ollama model you have pulled
```

Any instruction-following model that supports structured JSON output
will work. Larger models (e.g. `qwen2.5:14b`) produce more accurate
extractions at the cost of higher RAM and slower inference.

### Changing the embedding model

Edit `app/rag/vector_store.py`:

```python
EMBEDDING_MODEL = "nomic-embed-text:latest"
```

If you change the embedding model after indexing policies, you must
delete `data/chroma/` and re-index all policies, because vectors
from different models are not compatible.

---

## 11. Known Limitations

**Scanned / image PDFs are not supported.**
If your policy PDF is a scanned image rather than a text-based PDF,
`pypdf` will extract no text and the upload will fail. You must use
a text-based PDF or paste the text directly using the "Paste Text"
option. OCR support is planned for a future version.

**Single organization only.**
The MVP uses one hard-coded default organization. Authentication and
multi-tenancy are not implemented. Do not expose this application to
the internet — it has no login system.

**Extraction can fail on complex PDFs.**
If the policy PDF has unusual formatting, columns, tables, or very
long exclusion sections, the AI extraction may truncate or miss fields.
Use the "Retry Extraction" button on the verify page. If it still fails,
use the "Paste Text" option to provide clean text directly.

**Inference is slow without a GPU.**
On a CPU-only machine, policy extraction takes 60–120 seconds and claim
extraction takes 20–60 seconds. A machine with at least 8 GB RAM is
recommended. A dedicated GPU reduces this to 5–15 seconds.

**No background processing.**
Extraction runs synchronously in the HTTP request. The browser will
appear to hang while Ollama is working. Do not close the tab.

**No document viewer.**
Uploaded PDFs and images are stored on disk but cannot be viewed through
the app interface in this version.

**SQLite only.**
The app uses SQLite. It is suitable for a single-user local setup.
For multi-user or production deployment, migrate to PostgreSQL.

---

## 12. Production Gate

**This application is an MVP for local testing with synthetic data only.**

Before using ClaimAssist with real policies, real claims, or real
claimant data, the following must be completed:

- [ ] Authentication and role-based access control
- [ ] Organization isolation verified by security testing
- [ ] HTTPS with valid TLS certificate
- [ ] Secure session management and CSRF protection
- [ ] Encryption at rest for uploaded files and database
- [ ] Secrets management (`.env` file, never committed to git)
- [ ] Production database (PostgreSQL recommended)
- [ ] Object storage with access controls for uploaded files
- [ ] Malware scanning on all uploaded files
- [ ] Rate limiting and abuse protection
- [ ] Backup and point-in-time recovery
- [ ] Retention and deletion policy
- [ ] Privacy impact assessment
- [ ] Legal and insurance-domain compliance review
- [ ] Incident response and monitoring
- [ ] Load testing
- [ ] Formal security penetration test

**MVP completion alone is not production authorization.**

---

## Disclaimer

ClaimAssist is a software development project for learning and
demonstration purposes. It is not a licensed insurance product,
does not constitute legal or financial advice, and must not be
used to make real insurance decisions without appropriate
professional oversight, compliance review, and authorization.

---

*ClaimAssist v1.1 — Built with Python, FastAPI, LangGraph, LangChain, ChromaDB, Ollama, Pydantic, and SQLite.*
