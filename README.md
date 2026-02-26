# PriorAuth Agent (Amazon Nova Hackathon)

PriorAuth Agent is the first system that lets a clinician speak a patient case and have AI navigate the payer's actual web portal to submit the authorization — no API integration needed. Voice intake feeds clinical context through Amazon Nova for ICD-10/CPT coding and medical necessity justification, then Playwright (or Nova Act) fills and submits the payer portal form automatically.

## System Architecture

Judge-facing architecture documentation (components, sequence, Nova/Bedrock integration, and observability):

- [System Architecture](docs/system-architecture.md)

## What is implemented now

- Voice intake parser that extracts patient and clinical context from transcript text.
- **Nova-powered** clinical reasoning agent (ON by default) that maps ICD-10/CPT codes via `amazon.nova-lite-v1:0` with heuristic fallback.
- **Nova-powered** medical necessity justification with extended thinking (ON by default), falling back to template prose.
- Payer policy retrieval agent with Bedrock Knowledge Base RAG (`BEDROCK_KB_ID`) and local JSON fallback.
- **Playwright browser automation** (default) that launches a visible Chromium browser and fills the mock payer portal field-by-field; Nova Act path available when API key is present.
- Strands-backed orchestration wrapper for runtime execution with legacy fallback.
- **Nova 2 Sonic gateway** (`POST /api/transcribe`) — sends audio to Sonic and pipes transcript into the pipeline; falls back to `mock_transcript` text when `USE_NOVA_SONIC` is not set.
- End-to-end orchestrator with trace steps:
  - Voice Intake
  - Eligibility Verification
  - Clinical Coding (Nova)
  - Knowledge Retrieval (Bedrock KB or local)
  - Medical Necessity Analysis (Nova + extended thinking)
  - Form Population
  - Human Review
  - Portal Submission (Playwright / Nova Act / HTTP)

## Quickstart (Nova path — recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 2. Configure AWS credentials with Bedrock access
export AWS_REGION=us-east-1    # or your Bedrock region

# 3. Start mock payer portal
python portal/app.py

# 4. Run full Nova-backed workflow with Playwright browser submission
python main.py --auto-approve
```

Nova coding and justification are **on by default**. If Bedrock credentials are unavailable, the agent automatically falls back to the local heuristic mapper and template justification — no env var change needed.

## Bedrock KB configuration

You can enable Bedrock Knowledge Base retrieval in either of these ways:

1. File-based config (recommended for local development):
```bash
cp knowledge_base/kb_config.example.json knowledge_base/kb_config.json
# edit knowledge_base/kb_config.json and set knowledge_base_id
```

2. Environment variable:
```bash
export BEDROCK_KB_ID=<your-kb-id>
```

If you need to provision a KB first, run:
```bash
python knowledge_base/create_bedrock_kb.py
```

For quick local/offline runs, no AWS setup is required. The retrieval agent automatically falls back to local policy JSON when KB config is missing or unavailable.

## Run locally

1. Install dependencies:
```bash
pip install -r requirements.txt
playwright install chromium
```

2. Start mock payer portal:
```bash
python portal/app.py
```

3. Run orchestrator (auto-approve for full submission):
```bash
python main.py --auto-approve
```

4. Check submitted requests:
```bash
curl http://127.0.0.1:5000/requests
```

5. Open live execution dashboard:
```bash
open http://127.0.0.1:5000/dashboard
```

The dashboard runs workflows asynchronously and streams live step-by-step updates via Server-Sent Events.

### Dashboard API

- `POST /api/runs`: queue a workflow run (returns `202`)
- `GET /api/runs`: list recent run summaries
- `GET /api/runs/<run_id>`: fetch full run state/result
- `GET /api/runs/<run_id>/events`: stream live run events (SSE)
- `POST /api/runs/<run_id>/approve`: approve and resume a run blocked at human review
- `POST /api/transcribe`: Nova 2 Sonic speech-to-text gateway (see below)

## Human-in-the-loop mode

Run without `--auto-approve` to stop at the approval gate:
```bash
python main.py
```

The workflow returns `next_action: "human_review_required"` and includes a review snapshot.

## Nova 2 Sonic voice gateway

The `/api/transcribe` endpoint bridges audio input to the PA pipeline.

**Demo mode** (no Sonic key required):
```bash
curl -s -X POST http://127.0.0.1:5000/api/transcribe \
  -H "Content-Type: application/json" \
  -d '{"mock_transcript": "I need a prior auth for Jane Doe..."}'
```

**Real Sonic mode** (requires Bedrock access to `amazon.nova-sonic-v1:0`):

```bash
export USE_NOVA_SONIC=1
# POST audio_b64 (base64-encoded WAV bytes) to /api/transcribe
# The returned transcript feeds directly into POST /api/runs
```

## Tests

```bash
python -m unittest -v test_nova.py
python -m unittest -v test_portal_api.py
```

### Optional live Nova coding integration test

When AWS credentials and Bedrock model access are configured:

```bash
RUN_BEDROCK_INTEGRATION_TESTS=1 \
python -m unittest -v test_nova.TestPriorAuthPipeline.test_reasoning_agent_nova_integration
```

### Nova-backed coding and justification

Both are **on by default** (`USE_NOVA_REASONING=1`, `USE_NOVA_JUSTIFICATION=1`). To disable and use only local heuristics:

```bash
export USE_NOVA_REASONING=0
python main.py --auto-approve
```

The fallback chain is: Nova → heuristic coding, Nova → template justification. Each step logs the source used (`nova`, `nova-guardrailed`, `heuristic_fallback`, etc.) in the workflow output.

### Enable Bedrock Knowledge Base retrieval

To use real RAG retrieval instead of local policy scoring:

```bash
export BEDROCK_KB_ID=<your-kb-id>
python main.py --auto-approve
```

If `knowledge_base/kb_config.json` exists (from `create_bedrock_kb.py`), the retrieval agent auto-loads `knowledge_base_id` when `BEDROCK_KB_ID` is not set.

If KB retrieval fails at runtime, the system automatically falls back to local policy retrieval and marks `retrieval_source` as `local_fallback` in workflow output.

Current implementation detail: the KB provisioning script uses Titan Text Embeddings v2 (`amazon.titan-embed-text-v2:0`) to stay aligned with the Bedrock KB supported embedding model list.

### Browser automation mode

Control how the agent submits to the payer portal with `USE_BROWSER_AUTOMATION`:

```bash
# Playwright — launches visible Chromium, fills form field-by-field (default)
python main.py --auto-approve

# Nova Act — real Nova Act browser agent (requires nova-act package + API key)
USE_BROWSER_AUTOMATION=nova_act python main.py --auto-approve

# HTTP adapter — fast, for tests and CI only
USE_BROWSER_AUTOMATION=0 python main.py --auto-approve
```

### Environment variables reference

| Variable | Default | Description |
| ---- | ---- | ---- |
| `USE_BROWSER_AUTOMATION` | `playwright` | Browser mode: `playwright` (default), `nova_act`, or `0` (HTTP adapter) |
| `BEDROCK_KB_ID` | _(auto from kb_config.json)_ | Bedrock Knowledge Base ID for RAG retrieval |
| `USE_NOVA_REASONING` | `1` | Enable Nova-backed ICD-10/CPT coding (falls back to heuristic) |
| `NOVA_REASONING_MODEL_ID` | `amazon.nova-lite-v1:0` | Model for clinical coding |
| `USE_NOVA_JUSTIFICATION` | `1` | Enable Nova-backed justification generation (follows USE_NOVA_REASONING) |
| `NOVA_JUSTIFICATION_MODEL_ID` | `amazon.nova-lite-v1:0` | Model for justification prose |
| `USE_NOVA_EXTENDED_THINKING` | `1` | Enable extended thinking for justification |
| `USE_NOVA_SONIC` | `0` | Enable real Bedrock Nova Sonic calls in /api/transcribe |
| `ENABLE_OTEL_CONSOLE` | `1` | Enable console OpenTelemetry span export for demo tracing |
| `ORCHESTRATOR_MODE` | `strands` | Runtime orchestration mode: `strands` or `legacy` |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock API calls |

## Current boundaries

- Voice intake uses a regex transcript parser. In production this node would be backed by Nova 2 Sonic bidirectional streaming with real audio capture.
- The Bedrock KB provisioning script uses `maxTokens: 300` chunks, which is appropriate for the short demo policy documents but would need tuning for longer clinical guidelines in production.
- `_normalize_kb_text` in the retrieval agent handles the 3 demo policy docs correctly; deeply nested or non-standard clinical guideline numbering would need additional parsing rules.
- Facility name extraction uses a greedy regex suited to the demo transcript; edge cases like "therapy at home" may produce unexpected results in free-form dictation.

## Next integration steps for hackathon depth

1. Replace transcript parser with Nova 2 Sonic Bidi streaming + async tool calls.
2. Add metadata filters + richer source attribution for Bedrock Knowledge Base retrieval.
3. Replace HTTP form submission adapter with Nova Act browser action sequence + screenshot approval.
4. Add OpenTelemetry spans per graph node and export to CloudWatch.
5. Add Bedrock Guardrails for transcript and justification safety.
