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

## Human-in-the-loop mode

Run without `--auto-approve` to stop at the approval gate:
```bash
python main.py
```

The workflow returns `next_action: "human_review_required"` and includes a review snapshot.

## Tests

```bash
python -m unittest -v test_nova.py
```

## Next integration steps for hackathon depth

1. Replace transcript parser with Nova 2 Sonic Bidi streaming + async tool calls.
2. Replace local retrieval with Bedrock Knowledge Bases + Nova Multimodal Embeddings.
3. Replace HTTP form submission adapter with Nova Act browser action sequence + screenshot approval.
4. Add OpenTelemetry spans per graph node and export to CloudWatch.
5. Add Bedrock Guardrails for transcript and justification safety.
