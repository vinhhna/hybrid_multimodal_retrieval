# Phase 5 Implementation Plan — Vision‑Grounded KG (v3) + Anti‑Bias + Binding‑Aware Scoring (Option C)

**Goal:** Implement Phase 5 on top of the Phase 4 system, producing a **vision-grounded, hub-controlled, two-stream KG** that is **selectively activated** and that **explicitly fixes attribute binding** via **box-conditioned verification (Option C)**.

**Core pipeline (unchanged):** CLIP + FAISS (Stage 1) → KG (Stage 2) → BLIP‑2 (Stage 3 re-rank, optional).  
**Dataset:** Flickr30K (Karpathy split).  
**Environment:** Kaggle-first (`/kaggle/working/hybrid_multimodal_retrieval`, data under `/kaggle/input/flickr30k/data/...`).  
**Days:** Start at **Day 1** (do not continue Phase 4 numbering).  
**Rule:** No test leakage — all KG construction, thresholds, and calibration fit on **train** (+ optional val) only.

---

## Deliverables Checklist (End State)

### Offline artifacts
- [ ] Vision detections (sharded): `data/vision/train_shards/*.parquet` (and optional val shards)  
  Columns: `image_id, entity_id, conf_raw, conf_eff, x1,y1,x2,y2, phrase_type, model_id`
- [ ] Thresholds & controls:
  - [ ] `data/vision/thresholds_present.json` (per-entity or grouped `τ_present`)
  - [ ] `data/vision/null_phrases.json` (negative-control phrase set)
  - [ ] `data/vision/noise_floor_stats.json` (noise distribution diagnostics)
- [ ] Updated entity stores:
  - [ ] `data/entities/entity_context.json` (adds vision fields + boxes)
  - [ ] `data/entities/entity_meta.json` (adds `df_image_vision` + confidence stats)
- [ ] Graph:
  - [ ] `data/graph/entity_graph_v5.pt` with edge types: `sem`, `cooc_caption`, `cooc_vision`
  - [ ] `data/graph/cooc_vision_pmi.*` + `data/graph/cooc_vision_edges.*`

### Online pipeline features
- [ ] Candidate generation v3: caption entities + global top‑N + CLIP visual prior + **safe neighbors**.
- [ ] Anti-confirmation-bias detection acceptance: per-entity/group thresholds + noise floor + `Conf_eff`.
- [ ] KGScore v3:
  - [ ] Caption stream
  - [ ] Vision stream (generic entity evidence)
  - [ ] Vision stream (binding-aware Option C for attribute binding)
  - [ ] Global calibration (no per-batch min–max for KG)
- [ ] Dynamic gating: `S_hybrid = α S_clip + (1−α) S_cov` with `S_cov` using vision evidence / binding satisfaction.
- [ ] Evaluation + ablations + rescue cases.

---

## Conventions (Phase 5 v3)

- **Presence decision:** `is_present_vision(e,I)` is based on per-entity/group `τ_present` **and** per-image noise floor (negative controls).
- **Effective confidence:** `Conf_eff(e,I)` is used downstream (aggregation, cooc_vision, coverage).
- **Binding:** for attribute-binding queries, the vision score uses **object boxes + crop-conditioned attribute verification** (Option C).  
  Colors default to HSV-based verification; non-color attributes optionally use CLIP-on-crop.

---

## Day-by-Day Plan

## Week 1 — Setup, Splits, Candidates, Detector Adapter

### Day 1: Phase 5 branch, configs, and module scaffolding
**Tasks**
- [ ] Create Phase 5 branch (e.g., `phase5-v3`).
- [ ] Add Phase 5 sections to `configs/entity_graph.yaml`:
  - detector, candidates, safe_neighbors, negative_controls, thresholds, cooc_vision, kg_aggregation, binding, gating.
- [ ] Create module skeletons (importable):
  - `src/vision/*`, `src/graph/*`, `src/retrieval/*` additions (binding/gating utilities).

**Deliverables**
- [ ] Updated config with Phase 5 v3 keys.
- [ ] Empty stubs committed without breaking Phase 4 modes.

**Acceptance**
- [ ] `pytest` and basic `clip_only` retrieval still run.

---

### Day 2: Karpathy split utilities + leakage guards
**Tasks**
- [ ] Implement a single source-of-truth split loader (train/val/test image IDs).
- [ ] Add hard checks:
  - KG build scripts never read test images/captions.
  - Calibration scripts never read test.
- [ ] Materialize split manifests under `data/splits/`.

**Deliverables**
- [ ] `src/data/splits.py` + `data/splits/karpathy_{train,val,test}.json`

**Acceptance**
- [ ] No overlap between splits (unit test).
- [ ] Sizes match expected (~29k/1k/1k).

---

### Day 3: Vision detection schema + storage utilities (Parquet shards)
**Tasks**
- [ ] Define canonical detection schema (including boxes and confs).
- [ ] Implement Parquet writer/reader utilities (sharded, resume-safe).
- [ ] Decide phrase type encoding (object vs attribute vs composed vs null).

**Deliverables**
- [ ] `src/vision/storage.py` + `data/vision/schema.md`

**Acceptance**
- [ ] Round-trip (write→read) preserves dtypes and row counts.

---

### Day 4: Entity index + CLIP visual prior retrieval
**Tasks**
- [ ] Build FAISS index over entity embeddings for fast top‑K entity retrieval.
- [ ] Implement: `retrieve_entities_for_image(image_id, topK, sim_floor)` using existing CLIP image embeddings.
- [ ] Add caching for repeated lookups.

**Deliverables**
- [ ] `data/entities/entity_faiss.index` (or equivalent)
- [ ] `src/graph/entity_index.py`

**Acceptance**
- [ ] Top‑K results deterministic; latency acceptable for batch use.

---

### Day 5: Candidate generation v3 + safe neighbor filtering (drift control)
**Tasks**
- [ ] Implement `src/graph/phrase_candidates_v3.py`:
  - caption entities
  - global top‑N
  - CLIP visual prior `E_vis(I)`
  - safe neighbors `Nbr_safe(I)` with configurable filters:
    - cosine ≥ `τ_sim`
    - mutual-kNN
    - `df_image_vision ≥ min_df_vis` (initially from caption df; later from vision df)
    - optional pixel anchoring (`n ∈ E_vis(I)`)
- [ ] Add `C_max` budget + priority order.
- [ ] Add unit tests for determinism and budget enforcement.

**Deliverables**
- [ ] Candidate generator module + tests.
- [ ] Debug script to print candidate-source breakdown for 100 images.

**Acceptance**
- [ ] Candidates include caption-omitted but visually plausible entities (spot check).
- [ ] Neighbor drift is visibly reduced (no obvious fantasy drift in small samples).

---

### Day 6: OWL‑ViT adapter + caching + smoke test
**Tasks**
- [ ] Implement `DetectorBase` + `DetectorOWLVit` adapter.
- [ ] Implement phrase batching and image batching (where possible).
- [ ] Add cache key: `(image_id, phrase_hash, model_id, preprocessing_version)`.
- [ ] Smoke test: 10 images × 50 phrases, verify boxes and scores.

**Deliverables**
- [ ] `src/vision/detector_base.py`, `src/vision/detector_owlvit.py`
- [ ] `scripts/smoke_detector.py`

**Acceptance**
- [ ] Outputs: valid boxes, finite confs; caching works on rerun.

---

## Week 2 — Grounding Pipeline + Anti‑Confirmation Bias Controls

### Day 7: Grounding runner (sharded, resume-safe) using candidates v3
**Tasks**
- [ ] Implement `scripts/phase5_run_grounding.py`:
  - loads split manifest
  - builds candidates per image
  - runs OWL‑ViT
  - writes Parquet shards
  - resume-safe shard completion markers
- [ ] Add per-shard summary logging.

**Deliverables**
- [ ] Sharded outputs for a small pilot shard set under `data/vision/pilot_shards/`.

**Acceptance**
- [ ] Can stop/restart without recomputing completed shards.

---

### Day 8: Null phrase set + noise-floor measurement (negative controls)
**Tasks**
- [ ] Define `E_null` (10–30 stable, visually diverse but irrelevant phrases).
- [ ] Run detector on `E_null` for pilot images and compute:
  - `noise(I) = p95(conf(null, I))`
- [ ] Persist `null_phrases.json` and noise diagnostics.

**Deliverables**
- [ ] `data/vision/null_phrases.json`
- [ ] `scripts/compute_noise_floor.py` + `data/vision/noise_floor_stats.json`

**Acceptance**
- [ ] Noise floor distribution looks non-degenerate (not all zeros, not all ones).

---

### Day 9: Effective confidence `Conf_eff` and acceptance rule implementation
**Tasks**
- [ ] Implement acceptance rule:
  - `τ_eff(e,I) = max(τ_present[e], noise(I)+δ)`
  - `Conf_eff = max(0, conf_raw − τ_eff)` (or sigmoid shaping)
- [ ] Update grounding pipeline to output both `conf_raw` and `conf_eff`.
- [ ] Ensure `is_present_vision` uses `conf_raw ≥ τ_eff`.

**Deliverables**
- [ ] Updated `src/vision/postprocess.py` (or equivalent) + pipeline integration.
- [ ] Pilot shards regenerated with `conf_eff`.

**Acceptance**
- [ ] False positives on null phrases do not produce positive `Conf_eff`.
- [ ] `Conf_eff` sparsity is reasonable (not all zero).

---

### Day 10: Per-entity / grouped `τ_present` calibration (train/val only)
**Tasks**
- [ ] Fit thresholds on train/val using weak supervision:
  - positives: images where entity appears in captions (weak, but usable)
  - negatives: random images without caption mention
- [ ] Choose calibration objective:
  - fixed FPR (recommended) or precision target for frequent entities
- [ ] For long-tail entities, back off to group thresholds (by df bucket).
- [ ] Persist `thresholds_present.json`.

**Deliverables**
- [ ] `scripts/fit_present_thresholds.py`
- [ ] `data/vision/thresholds_present.json`

**Acceptance**
- [ ] Thresholds are stable across reruns; frequent entities get sensible thresholds.

---

### Day 11: Full grounding run (train + optional val) + update entity_context/meta
**Tasks**
- [ ] Run grounding over train (and optional val) with:
  - candidates v3
  - per-entity/group thresholds
  - null-phrase noise floor
  - `Conf_eff` shaping
  - box retention for object terms
- [ ] Update:
  - `entity_context.json` (vision image IDs, confs, boxes)
  - `entity_meta.json` (`df_image_vision`, conf stats)

**Deliverables**
- [ ] `data/vision/train_shards/*.parquet` (+ optional `val_shards`)
- [ ] Updated `data/entities/entity_context.json`, `entity_meta.json`

**Acceptance**
- [ ] `df_image_vision` matches shard counts (sanity check).
- [ ] Spot-check a few entities: vision presence aligns with intuition.

---

## Week 3 — Graph v5 Build (`cooc_vision`), Graph Search Updates

### Day 12: Vision co-occurrence statistics + PMI/NPMI tables
**Tasks**
- [ ] Compute `df_vis(e)` and pair counts from `is_present_vision` (or `Conf_eff>0`).
- [ ] Compute PMI/NPMI with smoothing ε.
- [ ] Prepare stop-node list:
  - global top‑N and/or df-based top‑M.
- [ ] Persist sparse PMI structures.

**Deliverables**
- [ ] `data/graph/cooc_vision_counts.*`
- [ ] `data/graph/cooc_vision_pmi.*`
- [ ] `data/graph/stop_nodes.json`

**Acceptance**
- [ ] PMI behaves sensibly: common hubs have lower PMI with random nodes.

---

### Day 13: Build `cooc_vision` edges with hub control (PMI + degree caps + stop policy)
**Tasks**
- [ ] Construct `cooc_vision` edges:
  - keep edges with PMI/NPMI > threshold
  - apply per-node top‑K degree cap
  - apply stricter PMI threshold for stop-node endpoints
- [ ] Persist edge tensors.

**Deliverables**
- [ ] `src/graph/build_cooc_vision.py`
- [ ] `data/graph/cooc_vision_edges.pt` (or parquet)

**Acceptance**
- [ ] Graph is not explosively dense; hubs are capped.

---

### Day 14: Rebuild full entity graph v5 (sem + cooc_caption + cooc_vision)
**Tasks**
- [ ] Update graph builder to emit:
  - `("entity","sem","entity")`
  - `("entity","cooc_caption","entity")`
  - `("entity","cooc_vision","entity")`
- [ ] Version and store: `entity_graph_v5.pt` + metadata.

**Deliverables**
- [ ] `data/graph/entity_graph_v5.pt`
- [ ] `data/graph/entity_graph_v5_meta.json`

**Acceptance**
- [ ] Load graph successfully; edge types present with expected counts.

---

### Day 15: Update graph_search to support 3 edge types + optional hub penalty
**Tasks**
- [ ] Build adjacency lists for all edge types.
- [ ] Add config type weights:
  - `type_weight_sem`, `type_weight_cooc_caption`, `type_weight_cooc_vision`
- [ ] Add optional hub penalty multiplier during traversal (IDF-based).
- [ ] Add unit tests for traversal invariants (caps, time budget).

**Deliverables**
- [ ] Updated `src/graph/graph_search.py` + tests.

**Acceptance**
- [ ] Graph expansion favors co-occurrence when semantic weight is small (by config).

---

## Week 4 — Binding (Option C), Calibration, Gating, Evaluation

### Day 16: Query analysis — slot extraction + binding query classifier
**Tasks**
- [ ] Implement query slot extraction:
  - `(object, attributes)` pairs from dependency parse / heuristics.
- [ ] Implement binding query classifier:
  - returns `is_binding_query` and extracted slots.
- [ ] Define supported attribute types (start with colors).

**Deliverables**
- [ ] `src/retrieval/query_slots.py`

**Acceptance**
- [ ] On a test list of 50 queries, slots are reasonable and stable.

---

### Day 17: Crop pipeline + attribute verification (HSV first)
**Tasks**
- [ ] Implement crop extraction using OWL‑ViT boxes.
- [ ] Implement HSV-based color verification:
  - mapping color word → HSV ranges
  - produce `p(color | crop)` in [0,1]
- [ ] Batch crop processing for speed.

**Deliverables**
- [ ] `src/retrieval/attribute_verifier.py` (HSV)
- [ ] `scripts/debug_color_verifier.py` (visual sanity)

**Acceptance**
- [ ] Color verifier distinguishes red/green/blue on obvious cases.

---

### Day 18: Binding-aware vision score `score_vis,bind` (Option C) + switching rule
**Tasks**
- [ ] Implement binding evidence:
  - `E(o+a, I) = max_box p(o|box) * p(a|crop)`
- [ ] Implement slot satisfaction:
  - `score_vis,bind(I) = Σ_{slots} w * max_attr E(o+a, I)`
- [ ] Optional assignment consistency (only if needed).
- [ ] Switching rule:
  - binding query → `score_vis = score_vis,bind`
  - else → `score_vis = score_vis,ent`

**Deliverables**
- [ ] `src/retrieval/binding_score.py`
- [ ] Unit tests on synthetic boxes/crops.

**Acceptance**
- [ ] On constructed contrasts, “red shirt” ranks images with red shirts above red cars.

---

### Day 19: Two-stream KGScore + global calibration (no batch min–max for KG)
**Tasks**
- [ ] Implement KG aggregation:
  - `score_cap(I)`
  - `score_vis(I)` (ent or bind)
- [ ] Fit global calibration on train/val:
  - sigmoid calibration (preferred) or percentile clipping
- [ ] Persist calibration params.

**Deliverables**
- [ ] `src/graph/kg_aggregation.py`
- [ ] `scripts/fit_kg_calibration.py`
- [ ] `data/calibration/kg_{cap,vis}.json`

**Acceptance**
- [ ] Pathology prevented: one weak positive among zeros remains low after calibration.

---

### Day 20: Integrate into HybridSearchEngine + update gating (`S_cov`) + logging
**Tasks**
- [ ] Integrate v5 KGScore into fusion:
  - `FinalScore = w_clip*clip_norm + w_stage2*stage2_norm + w_kg*KGScore`
- [ ] Implement/extend gating:
  - `S_clip` from (max, margin, consistency)
  - `S_cov` using:
    - `Conf_eff` presence for concrete terms
    - binding slot satisfaction for binding queries
- [ ] Add structured per-query logs (CSV/JSONL) including:
  - gate decision, activation rate, top entities, slot scores.

**Deliverables**
- [ ] Updated `src/retrieval/hybrid_search.py`
- [ ] `src/retrieval/gating.py`
- [ ] `results/gating_logs.csv`

**Acceptance**
- [ ] All modes run end-to-end on a small query set; gate toggles KG sensibly.

---

### Day 21: Evaluation + ablations + rescue case pack
**Tasks**
- [ ] Run Flickr30K Karpathy evaluation (test captions) for:
  - `clip_only`, `hybrid`, `clip_kg_v5`, `full_v5`
  - gated variants
- [ ] Run stress tests (Winoground and/or ARO).
- [ ] Run required ablations (minimum):
  1) Candidate: with/without visual prior; safe neighbors on/off
  2) Anti-bias: global τ only vs (τ_present + noise floor + Conf_eff)
  3) Binding: bag-of-entities vs Option C binding
  4) Hub control: naive `cooc_vision` vs PMI+caps+stop policy
  5) Normalization: per-batch vs global calibration
  6) Static KG vs gated KG (activation rate + latency)
- [ ] Generate rescue cases and case-study bundles:
  - query, before/after ranks, key entities/edges, slot evidence.

**Deliverables**
- [ ] `results/phase5_flickr30k_karpathy.json`
- [ ] `results/phase5_stress_tests.json`
- [ ] `results/phase5_ablations.json`
- [ ] `results/phase5_rescue_cases/` (bundles)

**Acceptance**
- [ ] Reproducible results with saved configs and artifact versions.
- [ ] Clear “KG helps when CLIP fails” examples with attributable mechanisms (cooc_vision / binding / gating).

---

## Notes (Kaggle Practicalities)
- Prefer sharded Parquet for detections; each shard independent for resume safety.
- Cache OWL‑ViT outputs aggressively (phrase hashes + model id).
- Keep binding verification bounded:
  - apply Option C only to top‑N candidates (e.g., 50–200 images)
  - limit boxes per object (e.g., top 1–3).
- Log artifact versions + config snapshots for reproducibility.

---

**End State:** Phase 5 v3 yields a **vision-grounded**, **anti-bias controlled**, **hub-pruned** KG, with **binding-aware scoring (Option C)** and **adaptive activation**, enabling rigorous improvements on compositional/binding failures and credible rescue case studies.
