# Phase 4 Implementation Plan — Knowledge Graph (LightRAG-Multimodal)

**Timeline:** 3 weeks (21 days)  
**Goal:** Build and integrate a multimodal knowledge graph that augments retrieval with LightRAG-style guided expansion and provides structured context for Phase 5.

---

## 8. Implementation Steps & Timeline (1–3 weeks)

### Week 1 — Design, Scaffolding, and Minimal Graph Builder

#### Day 1–2: Finalize schema & configs (minimal multimodal: image, caption)

**Define node/edge types (minimal):**
* **Nodes:** 
  * Image, Caption (Region kept as a stub only; not implemented this phase)
* **Edges:**
  * **Semantic:** top-k_sem neighbors by cosine (image↔image, caption↔caption)
  * **Co-occurrence:** caption↔image (paired in dataset), caption↔caption (same image)

**CLIP-space alignment:**
* Ensure all nodes carry CLIP embeddings:
  * **Image:** CLIP image embedding (L2-normalized)
  * **Caption:** CLIP text embedding (L2-normalized)

**Config file (centralized YAML):**
Store these keys and defaults in `configs/phase4_graph.yaml` (or merged into `configs/default.yaml`).
```yaml
graph:
  k_sem: 16
  degree_cap: 16
  edge_dtype: fp16

retriever:
  K_seed: 10
  B: 20              # beam size
  H: 2               # max hops
  decay: 0.85
  N_max: 200         # max collected nodes
  T_cap_ms: 150      # time cap in milliseconds
  eps_gain: 1e-3     # marginal gain threshold

fusion:
  default: "rank_fusion"
  weighted_w:
    w1: 0.7          # CLIP weight
    w2: 0.2          # Stage-2 weight (when strong agreement)
    w3: 0.1          # KG weight

enrichment:
  enabled: true
  top_terms: 5
```

**Acceptance:**
* The YAML config loads successfully through the existing loader; printing it (or converting to dict) shows all defaults and any runtime overrides applied.

---

#### Day 3–4: Implement PyG containers and serialization utilities

**Data containers:**
* Use PyG `HeteroData` or two `Data` objects (one per type) with explicit cross-links:
  * `HeteroData.x_dict`: `{"image": Tensor[N_img, d], "caption": Tensor[N_cap, d]}`
  * `HeteroData.edge_index_dict`:
    * `("image", "sem_sim", "image")`: edge_index_ii, edge_weight_ii
    * `("caption", "sem_sim", "caption")`: edge_index_cc, edge_weight_cc
    * `("caption", "paired_with", "image")`: edge_index_ci (weight=1.0)
    * `("caption", "cooccur", "caption")`: edge_index_cc_co (optional; weight=1.0)
  * Maintain maps:
    * `nid_maps`: `{("image", image_id) → node_idx, ("caption", caption_id) → node_idx}`
    * `id_maps`: reverse maps for result decoding

**Builders:**
* `build_semantic_edges(type, X, k_sem, degree_cap)`: chunked approximate k-NN over L2-normalized CLIP features; returns `(edge_index, edge_weight in fp16)`
* `build_cooccurrence_edges(dataset)`: construct caption↔image, caption↔caption (same image)

**Serialization:**
* `save_graph(hetero, path_dir)`: saves tensors to `path_dir/{x_*.pt, eidx_*.pt, w_*.pt, maps.json}`
* `load_graph(path_dir)`: loads tensors & maps; validates shapes and dtypes

**Acceptance:**
* Running a tiny slice (e.g., 100 images) produces a saved graph, reloads without errors, and edge degrees are ≤ degree_cap.

---

#### Day 5–7: Minimal graph search scaffolding & enrichment interface

**Encode query:**
* `encode_query(q)`: returns CLIP embedding (text or image path input), L2-normalized

**Seed selection (shared CLIP space):**
* `seed_nodes(emb, K_seed)`: ANN over {image, caption} feature stores (FAISS or in-memory index for the prototype); returns seed lists with scores
* Always include paired captions for any seeded images

**Enrichment (toggleable):**
* `enrich_query(q, seeds, graph, top_terms=5)`: collect top captions adjacent to seeds (by semantic/co-occurrence) → tokenize → select top n-grams unigram/bigram by tf-idf or frequency → join into "enriched query" string
* If disabled, return original query

**Acceptance:**
* Unit tests on 5 synthetic queries verify seed selection returns K_seed items, enrichment returns a non-empty string (when enabled), and runtime < 5 ms/query (excluding ANN build).

---

### Week 2 — Full Graph Build & Guided Retrieval (Shallow Multi-hop)

#### Day 8–9: Batch-build full graph (Flickr30K)

**Chunked k-NN:**
* Process captions and images separately in chunks (e.g., 8k vectors/chunk) to build semantic edges; keep top-k_sem and enforce degree caps

**Co-occurrence edges:**
* Add caption↔image edges from dataset pairs; add caption↔caption edges for captions belonging to the same image (optional)

**Save artifacts:**
* `data/graph/x_image.pt`, `x_caption.pt`, `eidx_image_sem.pt`, `w_image_sem.pt`, `eidx_caption_sem.pt`, `w_caption_sem.pt`, `eidx_caption_image_paired.pt`, `maps.json`

**Acceptance:**
* Build time logged; memory footprint recorded; spot-check degree distributions; reload test passes.

---

#### Day 10–11: Implement guided expansion (LightRAG-style)

**Frontier & beam:**
* Data structure: a min-heap or priority queue keyed by `score(node)`; maintain visited sets per node type
* **Score update:**
  * Initialize seeds with score=1.0 (or normalized CLIP similarity)
  * For edge (u→v), `score_v_candidate = score_u × (decay^hop) × (edge_weight × type_weight)`
    * `type_weight`: sem=1.0, cooc=0.7

**Multi-hop:**
* For hop in 1..H:
  * For each node in current beam (size B), expand up to its top `expansion_cap` neighbors (e.g., ≤ degree_cap)
  * Add/update scores for neighbors; if already visited, keep the max score
* Maintain collected set; stop if `|collected| ≥ N_max` or `elapsed > T_cap_ms`

**Candidate extraction:**
* Collect image nodes from visited/collected; compute a "graph score" per image = max score encountered for that image (or aggregate by sum/max)

**Acceptance:**
* On 25-query smoke set, expansion completes under T_cap_ms with `|collected| ≤ N_max`; returns a non-empty candidate list per query.

---

#### Day 12: KG-based re-ranking + fusion plumbing

**KG score:**
* **Option A (fast):** shortest-path approximation via hop count from seeds (if edge_index is unweighted for sem/cooc); `kg_score = 1 / (1 + min_hops)`
* **Option B (richer):** approximate personalized PageRank (few power iterations over the subgraph only); kg_score in [0,1]

**Normalize & combine:**
* `clip_norm`: min-max over candidate list (avoid div-by-zero)
* (Optional) `stage2_norm`: only if Stage-2 is enabled and correlation with CLIP is strong; otherwise skip
* `kg_norm`: min-max normalized
* `final_score = w1*clip_norm + w2*stage2_norm + w3*kg_norm`, defaults w1=0.7, w2=0.0/0.2, w3=0.1
* **Fallback:** if `|ρ(clip, stage2)| < 0.15` → `rank_fusion(clip_rank, kg_rank[, stage2_rank])` using RRF

**Acceptance:**
* Deterministic ranking for fixed seed/random seeds; tests confirm `final_score` monotonic with each component when others are constant.

---

#### Day 13–14: Integrate Query Contextualization and add Graph mode to evaluator

**Enrichment integration:**
* If `enrichment.enabled`, call `enrich_query()` once before CLIP seeding; re-encode enriched text; proceed with seed selection

**Evaluator additions:**
* Add mode "Graph-augmented":
  * Step 1: (optional) enrich → encode → seed
  * Step 2: guided expansion (B, H, decay, caps)
  * Step 3: candidate finalization + re-ranking
  * Output: ranked image IDs
* Metrics: Recall@1/5/10, MRR, nDCG@10; latency mean/median; signed deltas vs CLIP-only

**Acceptance:**
* On 25-query smoke test: Graph mode runs end-to-end; logs include seed size, hops, collected nodes, and time breakdowns (seed/expansion/rerank).

---

### Week 3 — Hardening, Tuning, and Reporting

#### Day 15–16: Latency & agreement guardrails; ablations

**Guardrails:**
* Early exit: if `elapsed > T_cap_ms` at any step → return CLIP ranking
* Stage-2 gating (if used): skip Stage-2 when `|ρ(clip, stage2)| < 0.05`; use rank_fusion when `|ρ| < 0.15`

**Ablations (25-query set):**
* Enrichment ON/OFF; KG re-rank ON/OFF; B∈{10,20}, H∈{1,2}, K_seed∈{5,10,15}; decay∈{0.8,0.85,0.9}; w3∈{0.05,0.1,0.2}

**Acceptance:**
* Record a table (or JSON) of metrics and latency for each toggle; pick a default that meets success criteria (no Recall@10 drop, R@1/MRR ≥ Hybrid, Δlatency ≤ +250 ms over CLIP-only median).

---

#### Day 17: Robustness & edge cases

**Determinism:** 
* Fix seeds with FAST_SEED for smoke tests; ensure numpy/pytorch RNG seeding where applicable

**Empty/short captions:** 
* Enrichment should degrade gracefully

**Disconnected components:** 
* If seeds land in sparse regions, fallback maintains CLIP ranking

**Memory checks:** 
* Verify degree caps; ensure edge tensors are fp16 where safe

**Acceptance:**
* All tests pass; no crashes on missing neighbors; memory within budget.

---

#### Day 18–19: Context synthesizer stub & example payloads

**`synthesize_context(query, results, graph, cfg)`:**
* **texts:** gather top captions for the top-N images (truncate to ~120 chars)
* **image_refs:** absolute file paths or base64 URIs (configurable; default paths)
* **metadata:** image_ids, caption_ids (top hits), edges_summary (counts per edge type in subgraph)

**Save 2–3 example payloads for qualitative inspection**

**Acceptance:**
* JSON payloads validate; evaluator prints a short summary snippet per mode.

---

#### Day 20–21: Final evaluation run & short report

**Run the 25-query smoke test across:**
* CLIP-only, Hybrid (fixed), Graph (no enrichment), Graph (+enrichment)

**Report (short, structured notes or JSON):**
* Metrics per mode; Δ vs CLIP-only; latency budget adherence
* What helped most (e.g., KG re-rank + enrichment + B=20, H=2)
* Defaults to use going forward (cfg snapshot)

**Acceptance:**
* Graph mode meets success criteria or is clearly justified with trade-offs and next steps (e.g., keep KG re-rank on; keep enrichment off by default if it adds latency with negligible gain).

---

## Deliverables Checklist

### Design & Scaffolding (Week 1)
- [ ] Node types: image, caption (region stub optional)
- [ ] Edge types: semantic, co-occurrence
- [ ] Config YAML (`configs/phase4_graph.yaml`) created with all parameters
- [ ] PyG containers implemented
- [ ] Serialization/deserialization working
- [ ] Enrichment interface implemented

### Graph Build & Retrieval (Week 2)
- [ ] Graph serialization/load working on full dataset
- [ ] Degree-capped semantic & co-occurrence edges
- [ ] Seed selection implemented
- [ ] Enrichment (toggle) working
- [ ] Guided expansion (H≤2, B=20) implemented
- [ ] KG re-rank implemented

### Evaluation & Context (Week 3)
- [ ] Evaluator supports Graph mode
- [ ] Metrics & latency logged
- [ ] Ablation results for key knobs
- [ ] Context stub produces `{texts, image_refs, metadata}`
- [ ] Sample payloads saved (2-3 examples)
- [ ] Final defaults chosen and recorded in YAML configs

---

**Ready to implement!** This detailed day-by-day plan ensures steady progress toward a working knowledge graph retrieval system with proper evaluation and context preparation for Phase 5.
