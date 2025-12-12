You are ChatGPT. I will give you (a) my **Phase 5 Plan** document, (b) the **Phase 5 Implementation Plan** (day-by-day schedule), and (c) a specific **Day + bullet subset** that I want implemented now.

Your job is to read those plans and produce **one single prompt for GitHub Copilot**, which I will paste into VS Code, so that Copilot can implement exactly those tasks (and nothing outside their scope).

High-level rules for you (ChatGPT):

- Always treat the Phase 5 Plan + Phase 5 Implementation Plan as the **source of truth** about design, artifacts, and constraints.
- Assume the project is on branch `phase5`, forked from `phase4-entity`, and already has:
  - CLIP + FAISS Stage 1 (text→image retrieval),
  - optional BLIP-2 reranker,
  - Phase 4 entity KG scaffolding (entity_vocab/context/embeddings + graph_search),
  - Phase 5 placeholder modules may exist but are not fully implemented.
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

### 1) Top banner

Explain to Copilot what this task is:

- Title: `Copilot task — Implement Phase 5 (selected items)`
- Briefly restate the **exact Day + bullets** to implement (quoted from Phase 5 Implementation Plan).
- Clarify the Phase 5 structure:
  - Stage 1: CLIP + FAISS
  - Stage 2: KG (vision-grounded) + gating + binding verification (Option C)
  - Stage 3: BLIP-2 rerank (optional)
- State that Phase 5 is not a new dataset project: we **keep Flickr30K** and the **entity-only graph design**, and we do **not** add image nodes / heterographs in Phase 5.

---

### 2) Scope & file policy

Tell Copilot exactly where to work:

- List the **existing files** to open/edit (examples; adapt to the chosen day):
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

- If new files are allowed, list them explicitly (path + short description). If **not** allowed, include:
  > Do **not** create new files. Modify only the files listed above.

- Require idempotent edits:
  > Edits must be idempotent: re-running Copilot with this prompt must not duplicate code, duplicate config keys, or break imports.

- Require Windows-safe behavior:
  - Use `pathlib.Path`
  - Avoid hardcoding `/kaggle/...` inside library code (allow in scripts/config only)
  - Ensure scripts run from repo root in PowerShell: `python scripts\xxx.py ...`

---

### 3) Implementation contract (must-haves)

Translate Phase 5 design into concrete requirements Copilot must follow.

#### 3.1 Config-driven design
- All hyperparameters and paths live in YAML, not hardcoded in scripts.
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

#### 3.2 Split hygiene and leakage control
- Use Karpathy split; KG build and all calibrations use **train (+ optional val)** only.
- Never use test data for:
  - threshold fitting
  - normalization/calibration fitting
  - PMI estimation
  - stop-node heuristics

#### 3.3 Vision grounding: raw-first, calibrate second (no chicken-and-egg)
- Detection runs must store **raw outputs first**:
  - `conf_raw` and `boxes` for each (image, entity) prompt
- Thresholds are fit from stored raw data and applied in a postprocess step:
  - produce `conf_eff` and `is_present`
- All postprocessing must be re-runnable without rerunning OWL-ViT.

#### 3.4 Anti-confirmation-bias controls (detector is not a “yes-man”)
- Implement null phrase controls and noise floors:
  - compute per-image noise statistic from null phrase scores
  - enforce `tau_eff = max(tau_present(entity), noise_floor(image)+delta)`
  - define `conf_eff = max(0, conf_raw - tau_eff)`
- Keep artifacts:
  - `null_phrases.json`, `noise_floor_stats.json`, `thresholds_present.json`

#### 3.5 Candidate generation must find “caption-missed” objects
Candidate list per image must be a union of:
- caption entities
- safe semantic neighbors (filtered; no fantasy drift)
- CLIP-based visual prior tags (image→entity retrieval)
- global Top-N generic probes (bounded)

Hard constraints:
- Must be vectorized (avoid Python loops over entities/images where possible).
- Must respect `C_max` and stable priority ordering:
  `caption > visual_prior > safe_neighbors > global`.

#### 3.6 Graph reconstruction: 3 edge families + hub control
Graph must include edge types:
- `(entity, sem, entity)` (unchanged)
- `(entity, cooc_caption, entity)` (caption co-occurrence)
- `(entity, cooc_vision, entity)` (vision co-detections)

Vision co-occ edges must be filtered/pruned:
- use PMI/NPMI with validated df_vis counts
- stop-node filtering: do not let global hubs dominate edge creation
- apply degree caps

#### 3.7 KGScore: two-stream + global calibration (no batch relative normalization)
- Build KGScore streams:
  - caption stream (IDF_cap)
  - vision stream (IDF_vis using df_vis)
  - binding stream (Option C) for binding queries
- Normalize KGScore via global calibration (sigmoid or fixed-range clipping),
  NOT per-batch min–max over top-100.

#### 3.8 Attribute binding fix (Option C, gated)
- Binding is verified by box-conditioned evidence:
  - detect object boxes
  - verify attribute on crops:
    - if color → HSV verifier
    - else → CLIP-on-crop verifier (default fallback)
- Switching rule: only run Option C for binding queries (keep compute bounded).

#### 3.9 Gating (KG is not always-on)
- Implement/maintain adaptive KG activation:
  - compute S_clip from Stage-1 statistics
  - compute S_cov from entity coverage (including binding satisfaction)
  - gate KG on/off via threshold(s)
- Must log gate decisions for analysis.

---

### 4) Functions & modules to implement

Enumerate the exact functions/classes Copilot must create or modify, tied to the selected day.

Examples (adapt to selected Day):

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
  - `compute_df_vis(...)`, `compute_pmi(...)`, `prune_hubs(...)`

**Binding**
- `src/retrieval/query_slots.py`
- `src/retrieval/attribute_verifier.py`
- `src/retrieval/binding_score.py`

**Gating + integration**
- `src/retrieval/gating.py`
- `src/retrieval/hybrid_search.py` (wire gating + KGScore + fallbacks)

Always include:
> If these modules already exist with slightly different names or helpers, reuse and extend them instead of inventing new paths. Respect existing public APIs unless the selected day explicitly requires changes.

---

### 5) Algorithms & code snippets

Embed short steering snippets/pseudocode (not full implementations), for example:

**Noise floor + effective confidence**
```python
tau_eff = max(tau_present[e], noise_floor[image_id] + delta)
conf_eff = max(0.0, conf_raw - tau_eff)
is_present = conf_eff > 0