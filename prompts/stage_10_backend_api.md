# Stage 10 — FastAPI Backend

## Working protocol

Explain each new technical term in plain language with a concrete example. State assumptions, preserve unrelated user work, prefer readable documented code, show commands and results, and follow the gate policy in `prompts/README.md`.

## Objective

Expose the shared inference engine through safe, demo-ready prediction and robustness-comparison endpoints without duplicating preprocessing.

## Entry gate

This stage is conditional. Begin only after the Stage 12 core submission passes and at least 10 focused hours remain for both Stages 10 and 11. Otherwise record it as future work and stop.

## API contract

- `GET /health`: readiness, model ID/revision, head ID, device, and load status; never expose local paths.
- `POST /predict`: multipart field `file`; return `filename`, calibrated `pred`, `label`, `inference_ms`, and model version.
- `POST /compare`: multipart `file` plus validated `condition_id`; return transform metadata, clean result, transformed result, probability delta, and a transformed image preview data URL.

Verdicts use the frozen Stage 07 threshold and wording “Likely AI-generated” or “Likely authentic.” They must not claim certainty.

## Required work

1. Initialize exactly one `InferenceEngine` in FastAPI lifespan startup. `/health` is unready until loading succeeds; requests must never trigger lazy duplicate loads.
2. Read uploads with a 10 MB hard limit, verify content by decoding, correct EXIF orientation, and cap decoded size at 40 megapixels before expensive processing.
3. Accept static JPEG, PNG, and WebP only. Reject animation, decompression bombs, malformed content, unsupported formats, missing fields, and invalid condition IDs with clean 4xx responses.
4. Use the Stage 04 transform registry for `/compare`. Do not reimplement transforms in the backend. Encode the preview at a bounded size/quality.
5. Configure allowed origins from an environment variable defaulting only to the local Next.js development origin. Do not use wildcard CORS with credentials.
6. Serialize GPU inference through a bounded lock or queue to avoid concurrent VRAM spikes. Decode/validation failures must not enter that critical section.
7. Add structured request timing and sanitized error logs without retaining uploaded bytes.

## Tests

- Health before/after startup, response schemas, numeric agreement with CLI, threshold labels, and transform metadata.
- Spoofed MIME type, truncated file, oversized bytes, excessive dimensions, animated image, unknown condition, and malformed multipart.
- Concurrent prediction test proving a singleton model and bounded inference.
- CORS allow/reject behavior and absence of local paths/stack traces.

## Exit gate

Run API tests and local curl examples for all endpoints. Report startup time, warm inference latency, device, memory, and validation behavior, then continue to Stage 11 and rerun affected Stage 12 checks afterward.
