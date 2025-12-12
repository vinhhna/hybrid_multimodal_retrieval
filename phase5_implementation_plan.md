# Phase 5 Implementation Plan — v3.1 (Optimized for Heavy Days + Calibration-First + PMI Correctness + Binding Switch)

**Goal:** Implement Phase 5 on top of Phase 4, producing a **vision-grounded, anti-bias controlled, hub-pruned, two-stream KG** that is **selectively activated** and that **explicitly fixes attribute binding** via **Option C (box-conditioned verification)**.

**Core pipeline (unchanged):** CLIP + FAISS (Stage 1) → KG (Stage 2) → BLIP‑2 (Stage 3 re-rank, optional).  
**Dataset:** Flickr30K (Karpathy split).  
**Environment:** Kaggle-first (`/kaggle/working/hybrid_multimodal_retrieval`, data under `/kaggle/input/flickr30k/data/...`).  
**Days:** Start at **Day 1**.  
**Rule:** No test leakage — KG build, thresholds, calibration fit on **train** (+ optional val) only.

This version incorporates the engineering feedback:
- **Day 5:** Safe-neighbor logic must be **precomputed + vectorized**, avoiding Python loops.
- **Day 10:** Threshold fitting requires **raw detection data**; ensure calibration subset raw detections exist by Day 9.
- **Day 12:** PMI is critical; add explicit **df_vis correctness validations** before PMI.
- **Day 18:** Option C is complex; enforce a **switching rule** so crops run only for binding queries and only on top‑N images/boxes.

---

## End Deliverables (Checklist)

### Offline artifacts
- [ ] Raw detection shards (calibration subset): `data/vision/calib_raw_shards/*.parquet`  
  Columns: `image_id, entity_id, conf_raw, x1,y1,x2,y2, phrase_type, model_id`
- [ ] Raw detection shards (train + optional val): `data/vision/train_raw_shards/*.parquet`
- [ ] Postprocessed vision table (derived, can be recomputed): `data/vision/train_post_shards/*.parquet`  
  Adds: `noise_floor`, `tau_eff`, `conf_eff`, `is_present`
- [ ] Thresholds + controls:
  - [ ] `data/vision/thresholds_present.json`
  - [ ] `data/vision/null_phrases.json`
  - [ ] `data/vision/noise_floor_stats.json`
- [ ] Updated entity stores:
  - [ ] `data/entities/entity_context.json` (vision IDs, confs, boxes)
  - [ ] `data/entities/entity_meta.json` (`df_image_vision`, conf stats)
- [ ] Graph v5:
  - [ ] `data/graph/entity_graph_v5.pt` with edge types: `sem`, `cooc_caption`, `cooc_vision`
  - [ ] `data/graph/cooc_vision_pmi.*` + `data/graph/cooc_vision_edges.*`

### Online pipeline features
- [ ] Candidate generation v3.1: caption + global + CLIP visual prior + **safe neighbors (vectorized)**.
- [ ] Anti-bias vision evidence: per-entity/group thresholds + per-image noise floor + `conf_eff`.
- [ ] KGScore v3.1:
  - [ ] caption stream
  - [ ] vision stream (entity evidence)
  - [ ] vision stream (binding Option C for binding queries)
  - [ ] global calibration for KG (no per-batch min–max)
- [ ] Gating: `S_hybrid` with `S_cov` using `conf_eff` / binding satisfaction.

---

## Design Guardrails (Non-Negotiable)

1. **Raw-first grounding:** Always store `conf_raw` + boxes. Postprocess into `is_present` and `conf_eff` later.
2. **Vectorized safe neighbors:** Neighbor lists, mutual-kNN flags, and stop-node masks are precomputed arrays.
3. **PMI only after df_vis validation:** PMI is computed from the finalized presence table (postprocessed shards).
4. **Binding is gated:** Option C runs only if `is_binding_query(query)=True`, and only on top‑N images + top‑B boxes.

---

## Day-by-Day Plan

## Week 1 — Setup, Splits, Storage, and Precomputation

### Day 1: Phase 5 branch, config v3.1, and module scaffolding
**Tasks**
- [ ] Create branch (e.g., `phase5-v3.1`).
- [ ] Update `configs/entity_graph.yaml` with v3.1 keys:
  - candidates: `vis_prior_topK`, `tau_vis`, `C_max`
  - safe_neighbors: `tau_sim`, `use_mutual_knn`, `min_df_vis`, `require_in_vis_prior`
  - detector: model, batch sizes, caching paths
  - negative_controls: `null_phrases`, `noise_percentile`, `delta`
  - thresholds: grouped strategy + objectives
  - binding: `enable`, `topN_images`, `topB_boxes`, HSV ranges, CLIP fallback
  - PMI: smoothing, thresholds, caps, stop-node policy
- [ ] Create new modules (importable): `src/vision/*`, `src/graph/*`, `src/retrieval/*`.

**Deliverables**
- [ ] Config file updated; skeleton modules committed.

**Acceptance**
- [ ] Phase 4 modes still run (imports + `clip_only` smoke).

---

### Day 2: Karpathy split utilities + leakage enforcement
**Tasks**
- [ ] Implement split loader (train/val/test IDs).
- [ ] Add hard guardrails in scripts:
  - `--split train|val|test` required; build scripts forbid `test`.
- [ ] Write split manifests to `data/splits/`.

**Deliverables**
- [ ] `src/data/splits.py` + `data/splits/karpathy_{train,val,test}.json`

**Acceptance**
- [ ] Unit test: no overlap between splits; sizes match expected.

---

### Day 3: Detection storage schema + shard IO (raw vs postprocessed)
**Tasks**
- [ ] Define two tables:
  - raw detections (conf_raw + boxes)
  - postprocessed detections (adds noise_floor, tau_eff, conf_eff, is_present)
- [ ] Implement Parquet shard writer/reader (resume-safe, deterministic filenames).
- [ ] Standardize phrase types: `object|attribute|composed|null`.

**Deliverables**
- [ ] `src/vision/storage.py` + `data/vision/schema.md`

**Acceptance**
- [ ] Round-trip read/write passes; schema version field present.

---

### Day 4: Precompute indices & neighbor primitives (for Day 5 speed)
**Tasks**
- [ ] Build FAISS index over entity embeddings (for `E_vis(I)`).
- [ ] Precompute **semantic neighbor lists** for all entities:
  - `neighbors[e] = topK ids`
  - store as fixed-size int array `[N, K_neighbors]`
- [ ] Precompute **mutual-kNN mask**:
  - `mutual[e, j] = 1 if e in neighbors[neighbors[e,j]]`
- [ ] Persist:
  - `data/entities/entity_neighbors.npy`
  - `data/entities/entity_neighbors_mutual.npy` (bool)
  - `data/entities/entity_faiss.index`

**Deliverables**
- [ ] `src/graph/entity_index.py` + `scripts/precompute_entity_neighbors.py`
- [ ] Precomputed arrays on disk.

**Acceptance**
- [ ] No Python loops needed later except over images/batches; neighbor lookup is O(1) array indexing.

---

## Week 2 — Candidates (Fast), Detector Raw Runs, Noise Floors, Threshold Calibration

### Day 5: Candidate generation v3.1 (vectorized safe neighbors; no slow loops)
**Tasks**
- [ ] Implement candidate generation using **array ops**:
  - `E_cap(I)` from context (list)
  - `E_vis(I)` via FAISS topK
  - `E_global` precomputed list
  - safe neighbors computed as:
    - gather `neighbors[E_seed]` into a 2D array
    - apply masks (`tau_sim` prefiltered at neighbor-precompute time if possible; else mask by precomputed sims)
    - mask by `mutual` if enabled
    - optional pixel anchoring: keep only neighbors also in `E_vis(I)` (use boolean membership via bitset or sorted-merge)
- [ ] Implement budget cap `C_max` with priority:
  - caption > visual prior > safe neighbors > global
- [ ] Add performance test:
  - generate candidates for 5k images; record avg ms/image.

**Deliverables**
- [ ] `src/graph/phrase_candidates_v31.py`
- [ ] `tests/test_candidates_perf.py` (lightweight perf/assert thresholds)

**Acceptance**
- [ ] Candidate gen runs without Python loops over candidate entities.
- [ ] Average candidate gen time is stable and acceptable (target: sub‑10ms/image excluding disk IO).

---

### Day 6: OWL‑ViT adapter + cache keying + smoke test
**Tasks**
- [ ] Implement adapter + batching + caching.
- [ ] Ensure box format standardized to xyxy in image pixel coords.
- [ ] Smoke test 10 images × ~80 phrases.

**Deliverables**
- [ ] `src/vision/detector_owlvit.py` + `scripts/smoke_detector.py`

**Acceptance**
- [ ] Caching hit rate validated; outputs are finite and boxes valid.

---

### Day 7: Calibration subset raw detection run (required for Day 10)
**Tasks**
- [ ] Select **calibration subset** of train images (e.g., 2k–5k, deterministic seed).
- [ ] Run OWL‑ViT with candidates v3.1 and write **raw** shards:
  - `data/vision/calib_raw_shards/*.parquet`
- [ ] Record raw score diagnostics:
  - per-entity score histograms for frequent entities
  - fraction of images with any detection > 0.1, > 0.3, etc.

**Deliverables**
- [ ] `scripts/run_calibration_subset_raw.py`
- [ ] Calibration raw shards + `results/calib_raw_diagnostics.json`

**Acceptance**
- [ ] Calibration raw data exists and is sufficient to fit thresholds on Day 10.

---

### Day 8: Null phrases + per-image noise floors (on calibration subset)
**Tasks**
- [ ] Finalize `E_null` (10–30 stable phrases).
- [ ] Run OWL‑ViT on `E_null` for calibration subset images (raw).
- [ ] Compute `noise(I) = p95(conf_null)` and persist per-image noise floors for calibration subset.
- [ ] Persist diagnostics of noise distribution.

**Deliverables**
- [ ] `data/vision/null_phrases.json`
- [ ] `scripts/compute_noise_floor_calib.py`
- [ ] `data/vision/noise_floor_stats.json`

**Acceptance**
- [ ] Noise floor is non-degenerate; stable across reruns.

---

### Day 9: Postprocess calibration subset (compute tau_eff, conf_eff, is_present)
**Tasks**
- [ ] Implement postprocess:
  - `tau_eff(e,I) = max(tau_present_guess[e], noise(I)+delta)` (use provisional tau present guess, e.g., 0.0 for now)
  - output `conf_eff = max(0, conf_raw - tau_eff)` and `is_present`
- [ ] Ensure postprocess is an **overlay step** that can be re-run once thresholds are fitted.

**Deliverables**
- [ ] `src/vision/postprocess.py`
- [ ] `scripts/postprocess_shards.py` (calib subset)
- [ ] `data/vision/calib_post_shards/*.parquet` (provisional)

**Acceptance**
- [ ] Pipeline supports re-postprocessing without rerunning OWL‑ViT.

---

### Day 10: Fit `thresholds_present` using calibration subset raw data (no chicken-and-egg)
**Tasks**
- [ ] Fit per-entity or grouped thresholds using calibration subset:
  - positives from caption mentions (weak supervision)
  - negatives from non-mentions
- [ ] Objective: fixed FPR or precision target for frequent entities; grouped fallback for long tail.
- [ ] Persist `thresholds_present.json`.
- [ ] Re-run postprocess on calibration subset using the fitted thresholds to validate acceptance rates.

**Deliverables**
- [ ] `scripts/fit_present_thresholds.py`
- [ ] `data/vision/thresholds_present.json`
- [ ] `results/threshold_fit_report.md`

**Acceptance**
- [ ] Thresholds produce reasonable separation on calibration subset (report curves for top entities).

---

### Day 11: Full train raw detection run (resume-safe) + postprocess pass
**Tasks**
- [ ] Run OWL‑ViT for full train (and optional val) storing **raw** shards first:
  - `data/vision/train_raw_shards/*.parquet`
- [ ] Run postprocess using:
  - fitted `tau_present`
  - noise floors (compute noise per image via null phrases or cached null scores)
  - produce post shards:
    - `data/vision/train_post_shards/*.parquet`
- [ ] Update `entity_context.json` and `entity_meta.json` from post shards.

**Deliverables**
- [ ] Train raw shards + train post shards.
- [ ] Updated entity stores.

**Acceptance**
- [ ] `df_image_vision` matches counts from post shards.
- [ ] Boxes exist for object terms needed by Option C.

---

## Week 3 — PMI, `cooc_vision` Graph, and Graph Search (Correctness First)

### Day 12: df_vis correctness validation (mandatory) + co-count computation
**Tasks**
- [ ] Compute `df_vis(e)` from **post** shards (`is_present` or `conf_eff>0`).
- [ ] Validate df correctness:
  - invariant: `0 ≤ df_vis(e) ≤ |train_images|`
  - sum over entities consistent with shard aggregates
  - top entities look plausible (e.g., person/sky/grass high)
- [ ] Compute pair co-counts in a controlled way (sparse):
  - within-image pairs among entities present
  - use degree/budget caps per image if needed for compute
- [ ] Only after validation passes, proceed to PMI.

**Deliverables**
- [ ] `scripts/validate_df_vis.py` with a pass/fail report
- [ ] `data/graph/df_vis.parquet` + `data/graph/cooc_vis_counts.*`

**Acceptance**
- [ ] Validation report passes; no PMI computed if df checks fail.

---

### Day 13: PMI/NPMI computation + stop nodes list
**Tasks**
- [ ] Compute PMI/NPMI with smoothing ε using validated `df_vis` and co-counts.
- [ ] Create stop-node list:
  - global top‑N and/or df-based top‑M (saved).
- [ ] Save sparse PMI tables.

**Deliverables**
- [ ] `data/graph/cooc_vision_pmi.*`
- [ ] `data/graph/stop_nodes.json`

**Acceptance**
- [ ] PMI sanity checks: random pairs low PMI; meaningful pairs higher.

---

### Day 14: Build `cooc_vision` edges (PMI + degree caps + stop-node policy)
**Tasks**
- [ ] Construct edges:
  - filter by PMI/NPMI threshold
  - apply per-node top‑K cap
  - stricter PMI threshold when either endpoint is a stop node
- [ ] Persist `edge_index` + weights for PyG graph.

**Deliverables**
- [ ] `src/graph/build_cooc_vision.py`
- [ ] `data/graph/cooc_vision_edges.pt`

**Acceptance**
- [ ] Graph density within bounds; hubs capped.

---

### Day 15: Build entity_graph_v5 + update graph_search (3 edge types + hub penalty)
**Tasks**
- [ ] Build graph with edge types: `sem`, `cooc_caption`, `cooc_vision`.
- [ ] Update graph_search to:
  - support 3 type weights
  - optionally apply hub penalty during traversal
- [ ] Add unit tests (caps, determinism, time budget).

**Deliverables**
- [ ] `data/graph/entity_graph_v5.pt`
- [ ] Updated `src/graph/graph_search.py` + tests

**Acceptance**
- [ ] Graph loads; traversal runs; cooc_vision influences expansion when configured.

---

## Week 4 — Binding (Option C), KGScore, Calibration, Gating, Evaluation

### Day 16: Slot extraction + binding query classifier (must be reliable)
**Tasks**
- [ ] Implement `query_slots.py`:
  - extract `(object, attributes)` slots
  - classify `is_binding_query`
- [ ] Build a small labeled dev set (50–100 queries) for regression testing.

**Deliverables**
- [ ] `src/retrieval/query_slots.py` + `tests/test_query_slots.py`

**Acceptance**
- [ ] Slot extraction stable; binding classifier high precision (prefer fewer false positives).

---

### Day 17: Attribute verifier (HSV for colors + CLIP-on-crop fallback by default)
**Tasks**
- [ ] Implement `attribute_verifier.py`:
  - if `is_color(attr)`: HSV-based `p(attr|crop)`
  - else: **CLIP-on-crop** `p(attr|crop)` (default), batch-processed
- [ ] Add config switches:
  - enable/disable CLIP fallback
  - crop resize policy
- [ ] Performance test on 200 crops.

**Deliverables**
- [ ] `src/retrieval/attribute_verifier.py`
- [ ] `scripts/benchmark_attribute_verifier.py`

**Acceptance**
- [ ] Color works; CLIP fallback returns bounded scores and runs in batches.

---

### Day 18: Binding score (Option C) with strict switching rule (compute bounded)
**Tasks**
- [ ] Implement `binding_score.py`:
  - get object boxes from OWL‑ViT detections
  - restrict compute:
    - apply only when `is_binding_query=True`
    - only top‑N images (config `binding.topN_images`)
    - only top‑B boxes per object (config `binding.topB_boxes`)
- [ ] Compute:
  - `E(o+a,I) = max_box p(o|box) * p(a|crop)`
  - slot score and total binding score
- [ ] Add tests on synthetic cases + a tiny real sample.

**Deliverables**
- [ ] `src/retrieval/binding_score.py` + `tests/test_binding_score.py`

**Acceptance**
- [ ] Bounded runtime; switching rule enforced; correct ordering on toy contrasts.

---

### Day 19: Two-stream KGScore + global calibration for KG (no batch min–max)
**Tasks**
- [ ] Implement KG aggregation:
  - `score_cap(I)`
  - `score_vis(I)` = entity evidence OR binding score depending on query
- [ ] Fit global calibration on train/val:
  - sigmoid or percentile clipping (persist params)
- [ ] Replace any KG batch-minmax with calibrated mapping.

**Deliverables**
- [ ] `src/graph/kg_aggregation.py`
- [ ] `data/calibration/kg_{cap,vis}.json`

**Acceptance**
- [ ] Sparse-signal pathology prevented by calibration tests.

---

### Day 20: Integrate into HybridSearchEngine + gating updates + logging
**Tasks**
- [ ] Integrate KGScore v5 into fusion modes.
- [ ] Update gating:
  - `S_clip` signals
  - `S_cov` uses `conf_eff` presence and binding satisfaction (for binding queries)
- [ ] Add structured logs (CSV/JSONL) for gate decisions and evidence.

**Deliverables**
- [ ] Updated `src/retrieval/hybrid_search.py`, `src/retrieval/gating.py`
- [ ] `results/gating_logs.csv`

**Acceptance**
- [ ] End-to-end run for all modes; gated activation rate reasonable and tunable.

---

### Day 21: Evaluation + ablations + rescue case pack (final)
**Tasks**
- [ ] Run Flickr30K Karpathy evaluation:
  - `clip_only`, `hybrid`, `clip_kg_v5`, `full_v5`, gated variants
- [ ] Stress tests (Winoground / ARO).
- [ ] Ablations (minimum required):
  1) Candidate: visual prior on/off; safe neighbors on/off
  2) Anti-bias: global τ vs τ_present+noise+conf_eff
  3) Binding: bag-of-entities vs Option C
  4) Hub control: naive vs PMI+caps+stop
  5) Normalization: batch-minmax vs global calibration
  6) Static vs gated KG
- [ ] Generate rescue cases and case-study bundles with attribution.

**Deliverables**
- [ ] `results/phase5_flickr30k_karpathy.json`
- [ ] `results/phase5_stress_tests.json`
- [ ] `results/phase5_ablations.json`
- [ ] `results/phase5_rescue_cases/`

**Acceptance**
- [ ] Reproducible outputs with config snapshots and artifact version metadata.
- [ ] Clear attributed rescues demonstrating binding + anti-bias + cooc_vision contributions.

---

**End State:** Phase 5 v3.1 yields a robust, calibrated, and efficient vision-grounded KG with bounded-cost binding verification and correct PMI-based co-occurrence structure, enabling a rigorous “KG helps when CLIP fails” narrative.
