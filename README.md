# PriorAuth Agent (Amazon Nova Hackathon)

Voice-first prior authorization workflow that converts clinician speech into a structured PA submission package with a human review gate before portal submission.

## What is implemented now

- Voice intake parser that extracts patient and clinical context from transcript text.
- Clinical reasoning agent that maps ICD-10/CPT codes and performs policy-criteria checks.
- Payer policy retrieval agent backed by a local policy store (`knowledge_base/policies.json`).
- Browser automation adapter that submits to the local mock payer portal (`portal/app.py`).
- End-to-end orchestrator with trace steps:
  - Voice Intake
  - Eligibility Verification
  - Clinical Coding
  - Knowledge Retrieval
  - Medical Necessity Analysis
  - Form Population
  - Human Review
  - Portal Submission

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

### Enable Nova-backed medical justification

To generate prior-auth justification prose with Nova:

```bash
export USE_NOVA_REASONING=1
export USE_NOVA_JUSTIFICATION=1
export NOVA_JUSTIFICATION_MODEL_ID=amazon.nova-lite-v1:0
python main.py
```

`USE_NOVA_EXTENDED_THINKING=1` is enabled by default for the justification call and automatically falls back to standard inference if unsupported.

## Next integration steps for hackathon depth

1. Replace transcript parser with Nova 2 Sonic Bidi streaming + async tool calls.
2. Replace local retrieval with Bedrock Knowledge Bases + Nova Multimodal Embeddings.
3. Replace HTTP form submission adapter with Nova Act browser action sequence + screenshot approval.
4. Add OpenTelemetry spans per graph node and export to CloudWatch.
5. Add Bedrock Guardrails for transcript and justification safety.
