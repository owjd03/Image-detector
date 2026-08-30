# Stage 11 — Next.js Robustness Demo

## Working protocol

Explain each new technical term in plain language with a concrete example. State assumptions, preserve unrelated user work, prefer readable documented code, show commands and results, and follow the gate policy in `prompts/README.md`.

## Objective

Build a simple, polished single-page demo that compares clean and transformed predictions and communicates uncertainty honestly.

## Entry gate

Begin only after conditional Stage 10 passes and enough of its original 10-hour allowance remains to implement and verify this stage. Otherwise use the CLI and evaluation plots for the demo video.

## Required work

1. Replace the placeholder page with an accessible drag-and-drop/click file picker accepting JPEG, PNG, and WebP. Validate basic size/type client-side while treating backend validation as authoritative.
2. Show the selected image preview, filename, reset control, and a transform selector populated from a typed list matching backend condition IDs.
3. Provide two actions or one clear comparison flow that calls `/compare` and displays clean and transformed cards side by side. Each card must show preview, calibrated percentage, verdict, and inference time; also show probability delta and selected transform parameters.
4. Include idle, uploading, processing, success, and recoverable error states. Disable duplicate submissions and revoke object URLs during cleanup.
5. Use cautious language: “estimated likelihood” and “Likely …”; include a visible note that the detector can be wrong and should not be the sole basis for moderation or accusations.
6. Read the API base URL from a public environment variable. Centralize fetch logic and validate response shapes before rendering.
7. Keep styling responsive and presentation-ready without auth, accounts, history, analytics, database, or unrelated pages.

## Markdown report

Generate `report/frontend_report.md` with implemented states and features, accessibility checks, component/API-client/end-to-end results, production build result, real-backend demonstration evidence, commands used, and repository-relative screenshot links. If this conditional stage is skipped, Stage 12 must create the file with status `Skipped` and the documented reason.

## Tests

- Component tests for picker validation, transform selection, loading, reset, success, and backend error states.
- API-client tests for malformed/non-2xx responses.
- A mocked browser end-to-end flow that uploads an image and verifies both result cards and delta.
- Keyboard navigation, labelled controls, visible focus, useful alternative text, and mobile layout checks.
- Production type-check and build with no backend required.

## Exit gate

Run tests/build and demonstrate the page against the real local backend with licensed authentic and synthetic samples. Report results, confirm `report/frontend_report.md` was generated, and rerun the affected Stage 12 documentation, audit, and end-to-end checks.
