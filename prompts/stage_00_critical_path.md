# Stage 00 — Critical Path and Time Budget

## Working protocol

Explain each new technical term in plain language with a concrete example. State assumptions, preserve unrelated user work, prefer readable documented code, and show commands and results.

This stage is not a build step. It is the priority map that governs Stages 01–13 when time is short. Read it before starting any stage, and re-read it whenever a stage overruns its budget.

## Operating assumption

Under 72 hours remain before submission, on a single RTX 5080 with 32 GB system RAM. VRAM is unknown until detected with `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` or the Stage 01 doctor command. Plan for roughly 20 hours of focused build time plus overhead, not 72 hours of continuous work.

## What is actually graded

The challenge scores deliverables, not pipeline completeness. Three artifacts carry almost all the weight:

1. `model/scripts/predict.py` — the directory-to-JSON script the brief explicitly requires.
2. The root `README.md`.
3. The clean-versus-transformed robustness table.

Everything else is upside. A repository with those three finished beats a more ambitious pipeline that is half-built when time runs out.

## Must-ship path

Stages 01 → 02 → 03 (locked Tier A subset) → 04 → 05 → 06 → 07 → 08 (required tier) → 09 → 12.

These are a hard dependency chain and cannot be reordered: no embeddings before a pinned model, no heads before embeddings, no calibration before heads.

## Cut ladder

Cut in this order when time is short:

1. **Stage 13** (frequency-domain stretch) — skip outright at under 72 hours. Record the reasoning; see that stage.
2. **Stage 11** (Next.js demo) — no deliverable requires a web UI. The demo video can be a screen recording of the CLI and the evaluation plots.
3. **Stage 10** (FastAPI backend) — same reasoning. Cutting this also removes the CLI/API agreement check from Stage 12's acceptance evidence.

**Stage 08 is not on this ladder.** The Error Analysis Note is a required deliverable. Its *scope* is tiered inside that stage; its output always ships.

## Per-stage budget

These are elapsed budgets dominated by writing and debugging code, not guaranteed machine-time estimates. GPU throughput must be measured on the actual RTX machine before accepting an ETA; image decoding and transform encode/decode can dominate.

| Stage | Budget | Machine time | If it overruns |
|---|---|---|---|
| 02 model download | 0.5 h | ~5 min (network) | Check the HF cache path and network; do not fall back to an unpinned revision. |
| 03 datasets | 3 h | Measure network | Keep the locked Tier A quotas; use Tier C only if SID is inaccessible. |
| 04 manifests + transforms | 3 h | 10–20 min (CPU) | Ship the transform registry and manifests; defer near-duplicate review to a reported warning. |
| 05 extraction | 3 h | Measure first 1,000 images | Training-bank extraction takes priority; narrow report-only sweeps if the measured ETA overruns. |
| 06 head training | 2 h | Measure seed 42 | Run four seed-42 core experiments; run extra seeds only for the two finalists. |
| 07 calibration + eval | 3 h | ~10 min (CPU) | Core metric table and the clean-versus-transformed comparison first; bootstrap intervals and extra plots last. |
| 08 error analysis | 1 h | minutes | Required tier only. |
| 09 CLI | 2 h | seconds | This is graded. Do not cut it; cut something else. |
| 12 docs + submission | 3 h | — | Start this earlier rather than trimming it. |

### Two ways Stage 05 blows its budget

- **Too few workers.** A naive single-process decoder can leave the GPU idle. Tune workers conservatively within 32 GB RAM and accept only throughput measured by a 1,000-image preflight; do not rely on a theoretical images-per-second claim.
- **Serialized JPEG conditions.** Each `jpeg_qXX` condition is a real encode/decode round trip. Parallelize decode/transform work, but measure its memory cost and never allow workers to materialize the full sweep in RAM.

Because Stage 05 is the one long unattended run, use that window to draft Stage 12 documentation. Do not start conditional Stages 10–11 before the core Stage 12 gate passes.

## Ordering adjustments under time pressure

- **Write a stub README at the end of Stage 09, not Stage 12.** Twenty lines covering what the project is, how to install, and how to run `predict.py`. Stage 12 expands it. If the clock runs out mid-Stage-11, a working graded CLI with a short README still scores; neither one scoring is the failure mode to avoid.
- **Exit gates become report-and-continue.** Tier A is already locked. Report each stage and continue without waiting; stop only before the Stage 07 model freeze and at the Stage 12 submission checklist.
- Preserve every other invariant. Time pressure does not relax WildFake isolation, the frozen-CLIP rule, or the no-test-set-adaptation rule; violating those invalidates the results rather than saving time.

## Exit gate

Record under 72 hours, Tier A, conditional Stages 10–11, and skipped Stage 13 in the root README, then continue to Stage 01 remediation.
