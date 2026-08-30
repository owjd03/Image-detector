# Stage 12 — Integration, Documentation, and Submission Package

## Working protocol

Explain each new technical term in plain language with a concrete example. State assumptions, preserve unrelated user work, prefer readable documented code, show commands and results, and follow the gate policy in `prompts/README.md`.

## Objective

Prove clean-machine reproducibility and prepare every judged deliverable without uploading, publishing, or sending anything externally on the user’s behalf.

## Required work

1. Follow the README from a clean environment: install, model download, dataset verification, manifest build, extraction, training, evaluation, CLI inference, backend startup, and frontend startup. Fix documentation or reproducibility defects found.
2. Expand the Stage 09 stub README. The challenge names five required README items — confirm each is present as its own section, not merely implied: **project overview**; **setup and installation instructions**; **steps to reproduce your results**; **a brief reflection on limitations and what you would improve given more time**; **team member contributions**. Add architecture, tool/library/model choices, credentials, commands, and dataset roles/licenses around them.
3. Embed the final clean-versus-transformed table, the clean-versus-robust-versus-consistency ablation, the graded-versus-held-out generalisation table, the fixed-FPR operating-point table, and selected plots. Clearly separate SID test from the non-scoring CIFAKE and WildFake demonstration results, disclose sample counts, and state the Stage 03 data tier so the numbers are honestly scoped.
4. Document the frozen model revision, consolidated `model/outputs/final/` runtime bundle, checkpoint SHA-256, calibration temperature, threshold, experiment fingerprint, and `<2B` compliance. Explain that CLIP downloads on first setup and is not committed.
5. Add a Mermaid architecture diagram showing shared inference logic used by CLI and FastAPI, plus the data/embedding/training/evaluation flow.
6. Draft submission assets in a documentation folder: Devpost description, technical summary, limitations/error-analysis note, demo-video shot list/script, likely judge Q&A, and a final checklist. The Devpost description must individually cover the five categories the challenge names: **how the solution addresses the problem statement**, **development tools used**, **models or APIs used**, **libraries and frameworks used**, and **datasets and assets used**. Do not upload the video or alter remote repositories without separate authorization.
7. Ensure demo assets are team-created or properly licensed and contain no unauthorized trademarks/copyrighted material. The demo video must be uploaded to YouTube with **public** visibility and linked in the Devpost description.
8. Prepare a manual-steps checklist for the user to execute — these are the actions that must not be taken on their behalf:
   - **Make the GitHub repository public.** The challenge requires a public repository; a private one fails the deliverable regardless of its contents.
   - Upload the demo video to YouTube as public and link it in Devpost.
   - Submit the Devpost description.
9. Audit Git for credentials, datasets, CLIP weights, caches, large binaries, local absolute paths, generated frontend output, and ignored error images. Record classifier checkpoint and results sizes.
10. Run all Python tests, frontend tests/build, CLI contract tests, API tests, and one real end-to-end browser flow. If Stages 10–11 were cut per Stage 00, say so explicitly in the README rather than leaving the omission to be discovered.

## Markdown report

Generate `report/submission_report.md` as the final report index and audit summary. Link `training_report.md`, `evaluation_report.md`, `error_analysis.md`, `cli_report.md`, `api_report.md`, and `frontend_report.md`; record deliverable status, reproducibility commands and results, Git audit findings, artifact sizes, conditional-stage decisions, remaining manual actions, and repository-relative links. Never fabricate a missing report: mark it `Pending` or `Skipped` with a reason. Before exiting Stage 12, confirm this file links every available stage report.

## Acceptance evidence

- The CLI works independently while the API/frontend are stopped.
- CLI and API predictions for the same decoded image agree within tolerance. *(Skip only if Stage 10 was cut.)*
- All six graded challenge transform families appear in report artifacts at every specified parameter.
- The held-out conditions, the fixed-FPR operating points, and the `mlp_consistency` ablation all appear in report artifacts.
- No WildFake data influenced training, selection, calibration, or thresholds. The Stage 09 WildFake `predictions.json` is a demonstration artifact only.
- Neither the held-out conditions nor CIFAKE influenced selection, calibration, or thresholds.
- The README states the Stage 03 data tier, the frozen checkpoint, temperature, threshold, and `<2B` compliance.
- Commands are copy/paste reproducible and expected runtimes/storage are documented.

## Exit gate

Stop with test/build results, audit findings, artifact links, remaining manual submission steps, the conditional-web time calculation, and the final checklist. Confirm the manual submission checklist with the user. If at least 10 focused hours remain afterward, Stages 10–11 may run; otherwise the core submission is complete.
