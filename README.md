# PriorAuth Agent (Amazon Nova Hackathon)

Voice-first prior authorization workflow that converts clinician speech into a structured PA submission package with a human review gate before portal submission.

## System Architecture

Judge-facing architecture documentation (components, sequence, Nova/Bedrock integration, and observability):

- [System Architecture](docs/system-architecture.md)

## What is implemented now

- Voice intake parser that extracts patient and clinical context from transcript text.
- Clinical reasoning agent that maps ICD-10/CPT codes and performs policy-criteria checks.
- Payer policy retrieval agent with Bedrock Knowledge Base RAG (`BEDROCK_KB_ID`) and local JSON fallback.
- Browser automation adapter that submits to the local mock payer portal (`portal/app.py`).
- Strands-backed orchestration wrapper for runtime execution with legacy fallback.
- End-to-end orchestrator with trace steps:
  - Voice Intake
  - Eligibility Verification
  - Clinical Coding
  - Knowledge Retrieval
  - Medical Necessity Analysis
  - Form Population
  - Human Review
  - Portal Submission

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

## Human-in-the-loop mode

Run without `--auto-approve` to stop at the approval gate:
```bash
python main.py
```

The workflow returns `next_action: "human_review_required"` and includes a review snapshot.

## Tests

```bash
python -m unittest -v test_nova.py
python -m unittest -v test_portal_api.py
```

### Optional live Nova coding integration test

When AWS credentials and Bedrock model access are configured:

```bash
RUN_BEDROCK_INTEGRATION_TESTS=1 USE_NOVA_REASONING=1 \
python -m unittest -v test_nova.TestPriorAuthPipeline.test_reasoning_agent_nova_integration
```

### Enable Nova-backed coding in runtime

By default, code mapping uses the local heuristic mapper. To use Nova 2 Lite:

```bash
export USE_NOVA_REASONING=1
export NOVA_REASONING_MODEL_ID=amazon.nova-lite-v1:0
python main.py --auto-approve
```

If the model call fails and `require_model_success` is not enabled, the agent falls back to the heuristic mapper.

### Enable Bedrock Knowledge Base retrieval

To use real RAG retrieval instead of local policy scoring:

```bash
export BEDROCK_KB_ID=<your-kb-id>
python main.py --auto-approve
```

If `knowledge_base/kb_config.json` exists (from `create_bedrock_kb.py`), the retrieval agent auto-loads `knowledge_base_id` when `BEDROCK_KB_ID` is not set.

If KB retrieval fails at runtime, the system automatically falls back to local policy retrieval and marks `retrieval_source` as `local_fallback` in workflow output.

Current implementation detail: the KB provisioning script uses Titan Text Embeddings v2 (`amazon.titan-embed-text-v2:0`) to stay aligned with the Bedrock KB supported embedding model list.

### Enable Nova-backed medical justification

To generate prior-auth justification prose with Nova:

```bash
export USE_NOVA_REASONING=1
export USE_NOVA_JUSTIFICATION=1
export NOVA_JUSTIFICATION_MODEL_ID=amazon.nova-lite-v1:0
python main.py
```

`USE_NOVA_EXTENDED_THINKING=1` is enabled by default for the justification call and automatically falls back to standard inference if unsupported.
The workflow output includes `necessity.extended_thinking_used` so demos can show whether the reasoning path actually executed.

### Browser automation mode

Control how the agent submits to the payer portal with `USE_BROWSER_AUTOMATION`:

```bash
# HTTP adapter (default — fast, for tests and CI)
python main.py --auto-approve

# Playwright — launches visible Chromium, fills form field-by-field (visual demo)
USE_BROWSER_AUTOMATION=playwright python main.py --auto-approve

# Nova Act — real Nova Act browser agent (requires nova-act package + API key)
USE_BROWSER_AUTOMATION=nova_act python main.py --auto-approve
```

### Environment variables reference

| Variable | Default | Description |
| ---- | ---- | ---- |
| `USE_BROWSER_AUTOMATION` | `0` | Browser mode: `playwright`, `nova_act`, or `0` (HTTP adapter) |
| `BEDROCK_KB_ID` | _(auto from kb_config.json)_ | Bedrock Knowledge Base ID for RAG retrieval |
| `USE_NOVA_REASONING` | `0` | Enable Nova-backed ICD-10/CPT coding |
| `NOVA_REASONING_MODEL_ID` | `amazon.nova-lite-v1:0` | Model for clinical coding |
| `USE_NOVA_JUSTIFICATION` | `0` | Enable Nova-backed justification generation |
| `NOVA_JUSTIFICATION_MODEL_ID` | `amazon.nova-lite-v1:0` | Model for justification prose |
| `USE_NOVA_EXTENDED_THINKING` | `1` | Enable extended thinking for justification |
| `ENABLE_OTEL_CONSOLE` | `1` | Enable console OpenTelemetry span export for demo tracing |
| `ORCHESTRATOR_MODE` | `strands` | Runtime orchestration mode: `strands` or `legacy` |

## Next integration steps for hackathon depth

1. Replace transcript parser with Nova 2 Sonic Bidi streaming + async tool calls.
2. Add metadata filters + richer source attribution for Bedrock Knowledge Base retrieval.
3. Replace HTTP form submission adapter with Nova Act browser action sequence + screenshot approval.
4. Add OpenTelemetry spans per graph node and export to CloudWatch.
5. Add Bedrock Guardrails for transcript and justification safety.
