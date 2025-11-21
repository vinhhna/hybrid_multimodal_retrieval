# Phase 4 Implementation Plan — Entity-Centric Knowledge Graph

**Timeline:** ~3 weeks (21 days)  
**Goal:** Implement an **entity-centric knowledge graph** with **query enrichment** and **graph expansion** integrated into the existing CLIP/Hybrid retrieval system.

This plan replaces the older image/caption-node design and follows the updated Phase 4 Plan.

---

## 1. Prereqs & Setup (Day 0)

- Confirm the existing codebase is working:
  - CLIP-only search
  - Hybrid (CLIP + BLIP-2) search
  - Evaluation scripts for R@K and MRR
- Add new config sections:
  - `entity_graph`: paths, thresholds (min_df, k_sem, degree_cap, etc.)
  - `query_enrichment`: K_seed_raw, M_enrich, templates, etc.
  - `graph_search`: K_seed, H_max, B, T_cap_ms, fusion weights
- Create module skeletons:
  - `src/graph/entities.py`
  - `src/graph/build_entity_graph.py`
  - `src/graph/graph_search.py`
  - `src/graph/context.py`
  - `src/graph/config.py` (optional helpers)

---

## 2. Week 1 — Entity Extraction & Graph Construction

### Day 1–2: Entity vocabulary & context building

**Tasks:**

- Implement `build_entity_vocabulary(dataset, cfg)`:
  - Iterate over all captions.
  - Run noun-phrase / entity extraction (e.g. via spaCy or a small POS tagger).
  - Normalize strings (lowercase, strip punctuation, optional lemmatization).
  - Count frequencies per entity (image-level and corpus-level).
  - Filter by `min_df` (e.g. ≥ 5) to avoid ultra-rare entities.

- Build context mappings:
  - `entity_to_images: Dict[entity_name, Set[image_id]]`
  - `entity_to_captions: Dict[entity_name, Set[caption_id]]`

**Artifacts:**

- `data/entities/entity_vocab.json` (`entity_name` → stats, IDs)
- `data/entities/entity_context.json` (`entity_id` → {image_ids, caption_ids})  

**Acceptance:**

- Vocabulary size printed (reasonable number).
- Sample entries manually inspected for quality.

---

### Day 3–4: Entity embeddings & metadata

**Tasks:**

- Assign integer IDs to entities: `entity_name → entity_id`.
- Encode each entity name with CLIP text encoder:
  - Optionally use a template like `"a photo of {entity_name}"`.
  - L2-normalize embeddings.
- Save:
  - `entity_embeddings.pt` (tensor `[N_entities, d_model]`)
  - `entity_meta.json` (`entity_id → {name, df, cf}`)

**Acceptance:**

- No NaNs/Infs in embeddings.
- Basic stats printed: mean norm, distribution checks.

---

### Day 5–7: Build entity graph (semantic + co-occurrence edges)

**Tasks:**

- **Semantic edges:**
  - Use FAISS or batched dot products to compute top-`k_sem` neighbors per entity.
  - Edge weight = cosine similarity.
  - Degree-cap to avoid hubs.

- **Co-occurrence edges:**
  - For each image (or caption), get its entities.
  - For each pair `(e_i, e_j)`:
    - Add/accumulate a co-occurrence weight.
  - Normalize weights (optional: PMI or scaled counts).
  - Degree-cap.

- Pack into PyTorch Geometric `HeteroData`:
  - `data["entity"].x = entity_embeddings`
  - `("entity", "sem", "entity").edge_index / edge_weight`
  - `("entity", "cooc", "entity").edge_index / edge_weight`

- Implement `save_entity_graph(data, path)` / `load_entity_graph(path)`.

**Artifacts:**

- `data/graph/entity_graph.pt` (PyG HeteroData)
- `data/graph/entity_meta.json`, `entity_context.json`

**Acceptance:**

- Quick size/memory report.
- Degree distribution sanity-check.
- Load/reload smoke test.

---

## 3. Week 2 — Query Enrichment & Graph Search

### Day 8–9: Query enrichment (mandatory pipeline)

**Tasks:**

- Implement `enrich_query(query, dataset, encoders, entity_context, cfg)`:

  1. Encode original query with CLIP → `q0`.
  2. CLIP search over captions/images → top-`K_seed_raw`.
  3. Collect candidate entities from those seeds (via `entity_context`).
  4. Score entities by:
     - frequency in seeds
     - similarity between entity embedding and `q0`
  5. Select top-`M_enrich` entities.
  6. Build enriched text:
     - For text query: `f"{query}. Related: {e1}, {e2}, ..."`
     - For image query: `"photo of e1, e2, e3, ..."`
  7. Encode enriched text with CLIP → `q_enriched`.

- Integrate into search pipeline:
  - Graph mode **always** calls `enrich_query` first.
  - Baseline modes can bypass for ablations.

**Acceptance:**

- Unit tests on a few example queries.
- Log enriched text and entities for inspection.

---

### Day 10–12: Graph expansion (multi-hop LightRAG-style)

**Tasks:**

- Implement `graph_search(query, graph, encoders, cfg)`:

  1. Call `enrich_query` → `q_enriched`.
  2. Compute similarity between `q_enriched` and all entity embeddings.
  3. Select top-`K_seed` entity seeds.
  4. Initialize a priority queue (max-heap) of frontier nodes:
     - Key: `score(node)`
     - Seed scores start from similarity to `q_enriched`.
  5. Expand up to:
     - `H_max` hops (default 2),
     - `B` nodes processed,
     - or `T_cap_ms` runtime.

- Scoring rule:
  - For each edge `u → v` at hop `h`:

    \
    score_v += score_u * decay**h * edge_weight * type_weight
    \

  - `decay ≈ 0.85`
  - `type_weight = 1.0` for semantic, `0.7` for co-occurrence.

- Maintain:
  - `visited` set to avoid loops.
  - `entity_scores` dict keyed by `entity_id`.

- After expansion:
  - Convert `entity_scores` → `image_kg_scores` using `entity_context`.

**Acceptance:**

- Smoke test: for a few queries, print:
  - Seed entities
  - Expanded entities
  - Top images by KG score.
- Log runtime to verify within budget.

---

### Day 13–14: Score fusion & integration with existing pipeline

**Tasks:**

- Integrate KG scores into the main retrieval function:
  - For each candidate image, gather:
    - `clip_score` (existing Stage 1)
    - `stage2_score` (optional BLIP-2 Stage 2)
    - `kg_score` (from entity graph)

- Implement normalization + fusion:
  - Per-query min–max or softmax normalization.
  - Weighted sum:

    \
    final = w1 * clip + w2 * stage2 + w3 * kg
    \

  - Default configs for:
    - KG-only (no BLIP-2)
    - Full hybrid.

- Ensure **fallback paths**:
  - If KG not built/loaded → fall back to Hybrid.
  - If BLIP-2 disabled → adjust weights accordingly.

**Acceptance:**

- End-to-end search runs in:
  - CLIP-only
  - Hybrid (no KG)
  - KG+CLIP
  - Full hybrid modes.

---

## 4. Week 3 — Context, Evaluation & Tuning

### Day 15–16: Context synthesizer (entity-based)

**Tasks:**

- Implement `synthesize_context(query, ranked_images, entity_scores, graph, cfg)`:

  1. For top-`K_ctx` images:
     - Gather contributing entities with highest weights.
  2. Build text snippets per image:
     - `"Image {id}: entities = e1, e2, e3"`
     - Optionally attach one caption.
  3. Package as:

     ```python
     {
       "texts": [...],
       "image_refs": [{"image_id": ..., "path": ...}, ...],
       "metadata": {
         "top_entities": [...],
         "graph_stats": {...}
       }
     }
     ```

**Acceptance:**

- Save 2–3 example contexts to disk.
- Manually inspect for interpretability.

---

### Day 17–19: Evaluation & ablations

**Tasks:**

- Extend the evaluator:
  - Add **Graph mode** and **Full hybrid mode**.
  - Measure:
    - R@1, R@5, R@10
    - MRR
    - Latency per query.

- Run comparisons:
  1. CLIP-only baseline
  2. Hybrid baseline (CLIP + BLIP-2, no KG)
  3. CLIP + KG + enrichment
  4. CLIP + BLIP-2 + KG + enrichment

- Ablations:
  - Disable enrichment (holds H_max, KG) to see its impact.
  - Set `H_max = 0` (no expansion) vs `H_max = 2` (expansion).
  - Try a few different fusion weight sets `(w1, w2, w3)`.

**Acceptance:**

- Table of metrics saved (e.g. `results/phase4_eval.json` + markdown summary).
- Chosen defaults recorded in YAML configs with rationale.

---

### Day 20–21: Cleanup & documentation

**Tasks:**

- Clean code, type hints, docstrings.
- Write a short developer README section:
  - How to build the entity graph.
  - How to run graph-based search.
  - How to reproduce evaluation.
- Tag Phase 4 completion in git.

**Acceptance:**

- Code passes lint/format checks (if any).
- Plan items checked off.
- Ready for Phase 5 LLM integration.

---

**Result:** At the end of this implementation plan, the project has:
- An **entity-centric KG** built from Flickr30K captions.
- **Mandatory query enrichment** in the KG pipeline.
- A **graph expansion** module that propagates relevance along entity relations.
- Full integration with the existing CLIP/Hybrid system and evaluation framework.
