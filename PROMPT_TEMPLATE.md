# Copilot Prompt Template Generator

You are ChatGPT. I will give you (a) my **Phase 5 Plan** document, (b) the **Phase 5 Implementation Plan** (day-by-day schedule), and (c) a specific **Day + bullet subset** that I want implemented now.

Your job is to read those plans and produce **one single prompt for GitHub Copilot**, which I will paste into VS Code, so that Copilot can implement exactly those tasks (and nothing outside their scope).

## High-Level Rules for You (ChatGPT)

- Always treat the Phase 5 Plan + Phase 5 Implementation Plan as the **source of truth** about design, artifacts, and constraints.
- Assume the project is on branch `phase5`, forked from `phase4-entity`, and already has:
  - CLIP + FAISS Stage 1 (text→image retrieval)
  - Optional BLIP-2 reranker
  - Phase 4 entity KG scaffolding (entity_vocab/context/embeddings + graph_search)
  - Phase 5 placeholder modules may exist but are not fully implemented
- The Phase 5 identity is: **Visual QA via retrieval**, enhanced by a **vision-grounded entity KG**, activated selectively via **gating**, and with explicit **attribute binding** using **Option C (box-conditioned verification)**.
- Your output must be **only** the final Copilot prompt (no extra commentary), with clearly labeled sections described below.
- When necessary, you may inline **small code snippets** inside the Copilot prompt to steer it.

---

## Inputs

* **Repo root**: `hybrid_multimodal_retrieval/`
* **Platform**: Windows (VS Code). Use cross-platform Python (`pathlib`) and do not hardcode POSIX-only paths.
* **Dataset root**: configurable. Common values:
  - Kaggle: `/kaggle/input/flickr30k/data`
  - Local Windows: `D:\datasets\flickr30k\` (example; treat as config)
* **Vector dim (CLIP text/image/entity)**: `512`
* **Plan files** (Phase 5):
  - `phase5_plan_review_v3.md` (or latest)
  - `PHASE_5_IMPLEMENTATION_PLAN_V3_1.md` (or latest)
* **Scope to implement now**:
  * Paste the **exact Day header** from the Phase 5 Implementation Plan (e.g., `Day 10 — Fit thresholds_present (no chicken-and-egg)`)
  * Paste the **bullet points** under that day you want implemented (or the full day block)

---

## What you must produce (structure of the Copilot prompt)

Your Copilot prompt must contain the following sections, in this order:

1. **Top banner**
2. **Scope & file policy**
3. **Implementation contract (must-haves)**
4. **Functions & modules to implement**
5. **Algorithms & code snippets**
6. **Performance, scaling & safety constraints**
7. **Acceptance tests**
8. **Edit strategy & instructions for Copilot**

At the end, repeat the selected **Day header + bullets** verbatim to keep Copilot anchored.

Below is what each section should contain.

---

## Section 1: Top Banner

Explain to Copilot what this task is:

- **Title:** `Copilot task — Implement Phase 5 (selected items)`
- Briefly restate the **exact Day + bullets** to implement (quoted from Phase 5 Implementation Plan)
- Clarify the Phase 5 structure:
  - Stage 1: CLIP + FAISS
  - Stage 2: KG (vision-grounded) + gating + binding verification (Option C)
  - Stage 3: BLIP-2 rerank (optional)
- State that Phase 5 is not a new dataset project: we **keep Flickr30K** and the **entity-only graph design**, and we do **not** add image nodes / heterographs in Phase 5

---

## Section 2: Scope & File Policy

Tell Copilot exactly where to work:

**List the existing files to open/edit** (examples; adapt to the chosen day):
- `configs/entity_graph.yaml`
- `src/graph/config.py`
- `src/graph/entities.py`
- `src/graph/context.py`
- `src/graph/build_entity_graph.py`
- `src/graph/graph_search.py`
- `src/retrieval/hybrid_search.py`
- `src/retrieval/search_engine.py`
- `src/retrieval/gating.py`
- `src/retrieval/query_slots.py`
- `src/retrieval/binding_score.py`
- `src/retrieval/attribute_verifier.py`
- `src/vision/detector_owlvit.py`
- `src/vision/storage.py`
- `src/vision/postprocess.py`
- `scripts/*.py` files relevant to the chosen day (calibration, detection run, graph build, evaluation)

**If new files are allowed,** list them explicitly (path + short description). **If not allowed,** include:

> Do **not** create new files. Modify only the files listed above.

**Require idempotent edits:**

> Edits must be idempotent: re-running Copilot with this prompt must not duplicate code, duplicate config keys, or break imports.

**Require Windows-safe behavior:**
- Use `pathlib.Path`
  - Avoid hardcoding `/kaggle/...` inside library code (allow in scripts/config only)
  - Ensure scripts run from repo root in PowerShell: `python scripts\xxx.py ...`

---

## Section 3: Implementation Contract (Must-Haves)

Translate Phase 5 design into concrete requirements Copilot must follow.

### 3.1 Config-Driven Design

- All hyperparameters and paths live in YAML, not hardcoded in scripts
- Use/extend these config sections (adapt to existing config naming; do not rename keys unless needed):
  - `paths`: dataset root, artifacts root, cache dirs
  - `clip`: model name, device, batch sizes
  - `faiss`: index paths, topK
  - `candidate_gen` (Phase 5):
    - `K_sem_neighbors`
    - `global_topN`
    - `visual_prior_topK`
    - `safe_neighbors` switches (mutual-kNN, sim thresholds, pixel-anchored filter)
    - `C_max` candidate cap + priority ordering
  - `detector` (Phase 5):
    - model id (OWL-ViT / OWLv2), batching params
    - caching keys, shard size
  - `negative_controls`:
    - null phrase list, `noise_quantile`, `noise_delta`
  - `thresholds_present`:
    - fitting policy (fixed FPR / precision target), per-group/per-entity
  - `vision_postprocess`:
    - `conf_eff` definition, tau rules, is_present
  - `graph_v5`:
    - edge types weights, PMI thresholds, stop-nodes policy, degree caps
  - `binding` (Option C):
    - switching rule, topN images, topB boxes, attribute verifier settings
  - `kgscore_calibration`:
    - global sigmoid/clipping params (NO batch min–max)
  - `gating`:
    - S_clip components, S_cov, alpha, tau thresholds

### 3.2 Split Hygiene and Leakage Control

- Use Karpathy split; KG build and all calibrations use **train (+ optional val)** only
- Never use test data for:
  - Threshold fitting
  - Normalization/calibration fitting
  - PMI estimation
  - Stop-node heuristics

### 3.3 Vision Grounding: Raw-First, Calibrate Second (No Chicken-and-Egg)
- Detection runs must store **raw outputs first**:
  - `conf_raw` and `boxes` for each (image, entity) prompt
- Thresholds are fit from stored raw data and applied in a postprocess step:
  - produce `conf_eff` and `is_present`
- All postprocessing must be re-runnable without rerunning OWL-ViT.

### 3.4 Anti-Confirmation-Bias Controls (Detector is Not a "Yes-Man")

- Implement null phrase controls and noise floors:
  - compute per-image noise statistic from null phrase scores
  - enforce `tau_eff = max(tau_present(entity), noise_floor(image)+delta)`
  - define `conf_eff = max(0, conf_raw - tau_eff)`
- Keep artifacts:
  - `null_phrases.json`, `noise_floor_stats.json`, `thresholds_present.json`

### 3.5 Candidate Generation Must Find "Caption-Missed" Objects

Candidate list per image must be a union of:
- caption entities
- safe semantic neighbors (filtered; no fantasy drift)
- CLIP-based visual prior tags (image→entity retrieval)
- global Top-N generic probes (bounded)

Hard constraints:
- Must be vectorized (avoid Python loops over entities/images where possible).
- Must respect `C_max` and stable priority ordering:
  `caption > visual_prior > safe_neighbors > global`.

### 3.6 Graph Reconstruction: 3 Edge Families + Hub Control

Graph must include edge types:
- `(entity, sem, entity)` (unchanged)
- `(entity, cooc_caption, entity)` (caption co-occurrence)
- `(entity, cooc_vision, entity)` (vision co-detections)

Vision co-occ edges must be filtered/pruned:
- use PMI/NPMI with validated df_vis counts
- stop-node filtering: do not let global hubs dominate edge creation
- apply degree caps

### 3.7 KGScore: Two-Stream + Global Calibration (No Batch Relative Normalization)

- Build KGScore streams:
  - caption stream (IDF_cap)
  - vision stream (IDF_vis using df_vis)
  - binding stream (Option C) for binding queries
- Normalize KGScore via global calibration (sigmoid or fixed-range clipping),
  NOT per-batch min–max over top-100.

### 3.8 Attribute Binding Fix (Option C, Gated)

- Binding is verified by box-conditioned evidence:
  - detect object boxes
  - verify attribute on crops:
    - if color → HSV verifier
    - else → CLIP-on-crop verifier (default fallback)
- Switching rule: only run Option C for binding queries (keep compute bounded).

### 3.9 Gating (KG is Not Always-On)

- Implement/maintain adaptive KG activation:
  - compute S_clip from Stage-1 statistics
  - compute S_cov from entity coverage (including binding satisfaction)
  - gate KG on/off via threshold(s)
- Must log gate decisions for analysis.

---

## Section 4: Functions & Modules to Implement

Enumerate the exact functions/classes Copilot must create or modify, tied to the selected day.

**Examples (adapt to selected Day):**

**Candidate generation**
- `src/graph/phrase_candidates.py` (or existing module)
  - `generate_candidates_for_image(image_id, ...) -> np.ndarray[int]`
  - `generate_candidates_batch(image_ids, ...) -> List[np.ndarray[int]]`
  - `apply_safe_neighbor_filters(...)`

**Detection + storage**
- `src/vision/detector_owlvit.py`
  - `run_detection(images, phrases, ...) -> raw rows`
- `src/vision/storage.py`
  - `write_raw_shard(...)`, `iter_raw_shards(...)`
- `src/vision/postprocess.py`
  - `postprocess_raw_to_post(...)`

**Calibration**
- `scripts/run_calibration_subset_raw.py`
- `scripts/fit_present_thresholds.py`
- `scripts/compute_noise_floor.py`

**Graph build**
- `src/graph/build_entity_graph.py`
  - `build_cooc_vision_edges_from_post(...)`
  - `compute_df_vis(...)`, `compute_pmi_npmi(...)`, `prune_hubs(...)`

**Binding:**
- `src/retrieval/query_slots.py`
- `src/retrieval/attribute_verifier.py`
- `src/retrieval/binding_score.py`

**Gating + integration:**
- `src/retrieval/gating.py`
- `src/retrieval/hybrid_search.py` (wire gating + KGScore + fallbacks)

**Always include:**

> If these modules already exist with slightly different names or helpers, reuse and extend them instead of inventing new paths. Respect existing public APIs unless the selected day explicitly requires changes.

---

## Section 5: Algorithms & Code Snippets

Embed short steering snippets/pseudocode (not full implementations). Keep them short and local.

**Noise floor + effective confidence:**

```python
tau_eff = max(tau_present[e], noise_floor[image_id] + delta)
conf_eff = max(0.0, conf_raw - tau_eff)
is_present = conf_eff > 0
```
**PMI (safe):**

Use validated counts only. Let:
- `N` = number of images used to compute counts (train or train+val; never test)
- `df1` = df_vis(e1), `df2` = df_vis(e2)
- `c12` = co_count(e1, e2)
- `eps` = smoothing constant (e.g., 1.0)

**PMI:**

```python
pmi = math.log(((c12 + eps) * N) / ((df1 + eps) * (df2 + eps)))
```

**NPMI (optional, recommended for scale stability):**

```python
p12 = (c12 + eps) / (N + eps)
npmi = pmi / (-math.log(p12))
```

**Edge keep rule (example; driven by config):**

```python
keep = (npmi >= npmi_min) and (degree[e1] < deg_cap) and (degree[e2] < deg_cap)
```

**Stop-node stricter rule:**

```python
if e1 in stop_nodes or e2 in stop_nodes:
    keep = keep and (npmi >= npmi_min_stop)
```

**Binding score (Option C):**

# for each candidate object box b (topB boxes by object confidence):
score_b = p_obj(b) * verifier(attr, crop(b))
binding_evidence = max(score_b for b in boxes_topB)


Gating

```python
S_hybrid = alpha * S_clip + (1 - alpha) * S_cov
use_KG = (S_hybrid < tau_gate)
```

---

## Section 6: Performance, Scaling & Safety Constraints

Remind Copilot of Phase 5 constraints:

**Candidate gen (heavy day):**
- Vectorize; avoid Python loops over entities
- Precompute neighbor arrays; cap candidates

**Detector runs:**
- Shard outputs; resume-safe; caching
- Avoid storing huge in-memory tables

**Calibration-first pipeline:**
- Raw data must exist before threshold fitting

**PMI correctness:**
- Must validate df_vis counts before PMI; fail fast if inconsistent

**Binding (Option C) cost:**
- Enforce switching rule; restrict top-N images and top-B boxes

**Safety:**
- Do not change Stage-1 baseline behavior unless behind config flag
- Graceful fallbacks if artifacts missing (kg_score=0, skip BLIP-2)

---

## Section 7: Acceptance Tests

Specify what “done” means for the selected day:

Unit tests (pytest) for new functions/modules

Smoke scripts:

detection raw shard creation and reload

postprocess re-run without re-detection

df_vis validation pass/fail behavior

End-to-end minimal run:

clip_only still works

clip_kg works if artifacts exist; otherwise clean fallback

Logging:

write small diagnostics JSON/CSV for the day’s outputs (counts, timing, top entities)

8) Edit strategy & instructions for Copilot

Tell Copilot how to proceed:

1. Open all listed files and skim existing code and configs
2. Add/extend config keys first (do not break old keys)
3. Implement functions in small, testable units with type hints
4. Add one small runnable script for the day's pipeline step (if applicable)
5. Add/extend tests
6. Run smoke + pytest; fix imports; keep diffs minimal
7. Print a short end report (paths changed, how to run)

---

## Output Format (What You Send Back to Me)

When you respond, you must output only the final Copilot prompt, no explanations.

**The first line of the Copilot prompt must be:**
```
Copilot task: Implement Phase 5 items (from Phase 5 Plan — <paste day/bullets label here>)
```

**Then include the sections in this exact order:**
1. Top banner
2. Scope & file policy
3. Implementation contract
4. Functions & modules
5. Algorithms & snippets
6. Performance, scaling & safety constraints
7. Acceptance tests
8. Edit strategy

At the end of the Copilot prompt, re-quote the Selected Day/Bullets verbatim.

---

## Selected Day/Bullets to Implement Now (Paste Verbatim)

Paste the exact day header and bullets below, then send this entire message back to ChatGPT.

**Example:**

```
Day 10 — Fit thresholds_present (no chicken-and-egg)

Fit per-entity or grouped thresholds using calibration subset raw data:
- Positives from caption mentions (weak supervision)
- Negatives from non-mentions
- Objective: fixed FPR or precision target for frequent entities; grouped fallback for long tail
- Persist thresholds_present.json
- Re-run postprocess on calibration subset using the fitted thresholds to validate acceptance rates
```

**Now paste your real selection here:**

```
<PASTE YOUR DAY HEADER AND BULLETS HERE>
```