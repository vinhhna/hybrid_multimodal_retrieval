# Phase 4 Plan — Entity-Centric Knowledge Graph (LightRAG-Multimodal)

**Goal:** Replace the earlier image/caption node graph with an **entity-centric knowledge graph**, integrate it into the existing CLIP/Hybrid pipeline, and **mandatorily use query enrichment + graph expansion** (multi-hop reasoning) to improve retrieval quality while staying within latency limits.

This plan already incorporates instructor feedback:
- Use **graph expansion** (if *a = b* and *b = c* ⇒ *a* is related to *c*) via controlled multi-hop traversal.
- **Always use query enrichment** in the KG pipeline (not only as a toggle).
- Switch to **Entity nodes**; images are stored in a separate index/DB, and entities keep references (image IDs, caption IDs) as their context.

---

## 0. Scope & Success Criteria

### Scope (Phase 4 only)

- Build an **entity-centric multimodal knowledge graph** on Flickr30K using PyTorch Geometric.
- Implement a **graph retriever** (LightRAG-style):
  - Seed selection in CLIP space.
  - Guided **multi-hop expansion** (graph expansion) with scoring & stopping rules.
- Implement **mandatory query enrichment** prior to graph retrieval.
- Produce a **KG-based relevance signal** and combine it with:
  - CLIP score (Stage 1),
  - BLIP-2 cross-encoder score (optional Stage 2).
- Implement a **context synthesizer** that returns entity-based text + image references for Phase 5.
- Keep **CLIP-only** and **Hybrid (CLIP + BLIP-2)** as baselines and fallbacks.

### Success Criteria

- Accuracy:
  - Entity-KG-augmented retrieval **matches or exceeds** Hybrid on R@1 / MRR on the 25-query smoke test.
  - Recall@10 does **not degrade** by more than 2–3% compared to Hybrid.
- Latency:
  - Additional KG+enrichment work adds **≤ 250 ms** median over CLIP-only.
  - Hard caps: if KG or BLIP-2 overruns limits, system cleanly falls back to cheaper modes.
- Engineering:
  - Clean, well-documented APIs:
    ```python
    graph = build_entity_graph(dataset, encoders, cfg)
    save_entity_graph(graph, path)
    graph = load_entity_graph(path)

    results = graph_search(query, graph, encoders, cfg)
    context = synthesize_context(query, results, graph, cfg)

    metrics = evaluate_graph_mode(dataset, graph, cfg)
    ```

---

## 1. Entity-Centric Graph Design

### 1.1 Node types

We drop separate *Image* and *Caption* nodes. The graph now has a single **Entity** node type.

- **Entity node** (core node type):
  - `entity_id: int`
  - `name: str` — normalized surface form (e.g., `"brown dog"`, `"soccer ball"`, `"city street"`)
  - `embedding: Tensor[d_model]` — CLIP **text** embedding of `name` (L2-normalized)
  - `context` (stored off-graph, but indexed by `entity_id`):
    - `image_ids: List[str]` — IDs of images where this entity appears
    - `caption_ids: List[str]` — IDs of captions where it appears
    - (optional) positions / frequencies
  - `stats`:
    - `df` (document/image frequency)
    - `cf` (total count in all captions)

Images themselves are **not nodes**. They live in:
- The existing **FAISS indices** (image / caption embeddings).
- A simple **image metadata DB** (JSON, SQLite, or in-memory dict) mapping `image_id → path, size, etc.`

### 1.2 Entity extraction

For Phase 4 we keep extraction simple and deterministic:

- Input: all Flickr30K captions for each image.
- Pipeline (offline):
  1. **Tokenization + POS tagging / noun-phrase chunking** (e.g. spaCy or a small model).
  2. Extract **candidate entities**:
     - Noun phrases (`"a man"`, `"red car"`, `"football stadium"`).
     - High-information nouns/adjectives (filter stopwords).
  3. Normalize:
     - Lowercase, strip punctuation.
     - Optional lemmatization.
  4. Deduplicate into a global **vocabulary of entities**:
     - Keep entities with frequency `min_df` (e.g. ≥ 5).
  5. For each entity:
     - Compute CLIP text embedding of `"entity_name"` (or a short template like `"a photo of {entity_name}"`).
     - Build `image_ids` and `caption_ids` lists where it appears.

This yields ~several thousand entity nodes, each connected to many images via context.

### 1.3 Edge types

Edges operate **between entities only**:

1. **Semantic similarity edges** (`entity_sem`):
   - For each entity, find top-k semantic neighbors in CLIP text embedding space.
   - Edge weight = cosine similarity.
   - Apply degree-capping: each node keeps at most `k_sem` outgoing semantic edges.

2. **Co-occurrence edges** (`entity_cooc`):
   - For each image (or caption), take all entities that appear in it.
   - Connect all pairs `(e_i, e_j)` from that set with a co-occurrence edge.
   - Edge weight:
     - Either fixed (1.0), or
     - Proportional to PMI / co-occurrence count.
   - Apply `degree_cap` to prevent hubs.

This structure supports **graph expansion**: if entity *A* frequently co-occurs with *B*, and *B* with *C*, then following edges (*A → B → C*) allows us to discover *C* even if the query initially only hit *A*.

### 1.4 Storage & format

- Use **PyTorch Geometric** `HeteroData` with a single node type:
  - `data["entity"].x` — shape `[N_entities, d_model]`
  - `data["entity", "sem", "entity"].edge_index`
  - `data["entity", "sem", "entity"].edge_weight`
  - `data["entity", "cooc", "entity"].edge_index`
  - `data["entity", "cooc", "entity"].edge_weight`
- External maps / DB:
  - `entity_meta.json`: mapping `entity_id → {name, df, cf}`
  - `entity_context.json` or DB: `entity_id → {image_ids, caption_ids}`
  - `image_db.json`: `image_id → {path, caption_ids, clip_emb_index}`

---

## 2. Query Enrichment (Mandatory)

Query enrichment is now a **required step** in the KG pipeline (can be toggled off only for baselines/ablations).

### 2.1 Enrichment strategy

1. **Initial CLIP retrieval:**
   - Encode raw query (text or image) with CLIP → `q0`.
   - Run a **fast CLIP search** over captions/images to get top-`K_seed_raw` seeds.
2. **Entity extraction from seeds:**
   - Collect candidate entities from:
     - Captions of top images.
     - Context entities already attached to those images.
   - Score them (frequency, tf-idf, similarity of entity embedding to `q0`).
   - Pick top-`M_enrich` entity names.
3. **Build enriched query text:**
   - For text query:
     - Concatenate:  
       `"{original_query}. Related: entity1, entity2, entity3..."`.
   - For image query:
     - Construct a synthetic description from top entities:  
       `"photo of entity1, entity2, entity3..."`.
4. **Re-encode with CLIP:**
   - Encode enriched text with CLIP → `q_enriched`.
   - Use `q_enriched` for:
     - Entity seeding in the graph.
     - Re-running CLIP Stage 1 (if we want enriched CLIP scores).

### 2.2 When used

- **Default Phase 4 pipeline:** *always* enriched (`q_enriched`) for graph search.
- Baselines:
  - Graph search **without** enrichment (for ablation).
  - Hybrid (no KG) for comparison.

---

## 3. Graph Retrieval & Expansion (LightRAG-style)

### 3.1 Seeding entities

Input: `q_enriched`.

1. Compute similarity between `q_enriched` and all entity embeddings (via FAISS or batched dot product).
2. Select top-`K_seed` entity nodes as **seeds**.
3. Optionally ensure diversity:
   - De-duplicate entities with near-identical names.
   - Prefer seeds that connect to many images (high `df`).

### 3.2 Graph expansion (multi-hop / “if a=b and b=c then a=c”)

We implement **controlled multi-hop expansion** to realize the instructor’s “a=b, b=c ⇒ a=c” idea.

- Parameters:
  - `H_max` — max hops (default 2; allow 3 for experiments).
  - `B` — beam size (default 20).
  - `T_cap_ms` — time cap for expansion.
- Data structures:
  - Priority queue / max-heap of frontier nodes keyed by `score(node)`.
  - `visited` set of entity IDs.
  - Accumulator of **collected entities** with their final scores.

**Scoring rule (per edge traversal):**

For an expansion from `u` to `v` at hop `h`:

\[
score_v \mathrel{+}= score_u 	imes decay^h 	imes edge\_weight 	imes type\_weight
\]

- `decay` ≈ 0.85
- `type_weight`:
  - semantic edge: 1.0
  - co-occurrence edge: 0.7

This means:
- If *query → entity A* is high scoring,
- And *A* is strongly connected to *B*,
- And *B* is strongly connected to *C*,
- Then *C* also receives a non-trivial score even when not directly similar to the query — exactly the **graph expansion / transitivity** behavior requested.

**When to use graph expansion:**

- **Used by default** for KG mode:
  - For descriptive, multi-object queries (`"a kid playing football in a stadium"`).
  - For abstract queries where direct CLIP similarity may be weaker, but entities form a meaningful chain.
- **Limited / curtailed** when:
  - Time cap exceeded (`T_cap_ms`): stop expansion and use collected entities so far.
  - Hop limit reached (`H_max`).
  - Beam exhausted (`B` frontier nodes processed).

### 3.3 From entities back to images

Once we have a scored set of entities:

1. For each entity, access its `image_ids` context.
2. Compute an image KG score as an aggregation over its entities:
   - e.g. `image_kg_score[img] = sum(score_entity / sqrt(df_entity))`.
3. Optionally normalize per-image by number of contributing entities.

This gives a **KG-derived image relevance score** that captures multi-hop semantic and co-occurrence structure.

---

## 4. Score Fusion & Modes

For each image candidate we can have up to three signals:

- `clip_score` — from CLIP Stage 1 (possibly enriched query).
- `stage2_score` — from BLIP-2 cross-encoder (only for a subset).
- `kg_score` — from entity graph expansion.

### 4.1 Normalization

- Normalize each score into `[0, 1]`:
  - Min–max over candidate set, or
  - Softmax-based normalization.
- Handle missing `stage2_score` (if BLIP-2 not run) by:
  - Setting `w2 = 0` in that mode, or
  - Imputing a neutral value (e.g. mean).

### 4.2 Fusion strategies

1. **Weighted sum (default):**

\[
final = w_1 \cdot clip + w_2 \cdot stage2 + w_3 \cdot kg
\]

- Starting defaults:
  - Graph-only experiment: `w1=0.7, w2=0.0, w3=0.3`
  - Full hybrid: `w1=0.6, w2=0.2, w3=0.2`

2. **Rank fusion (RRF) as fallback** when signals disagree strongly:
   - Compute rank for each image in each list.
   - Use RRF: `score = sum(1 / (k + rank))`.

We will record final chosen defaults in a YAML config.

---

## 5. Context Synthesizer (Entity-based)

Given `(query, ranked_images, entity_scores, graph)` produce a **Phase 5 friendly** context:

```python
{
  "texts": [
    # short summaries / snippets formed from entities and captions
  ],
  "image_refs": [
    {"image_id": ..., "path": ...},
    ...
  ],
  "metadata": {
    "top_entities": [...],
    "entity_to_images": {...},
    "graph_stats": {...}
  }
}
```

### 5.1 Building textual context

For the top-K images:

1. Collect all entities that contributed to their `kg_score`.
2. For each image, keep top-`M` entities (highest contribution).
3. Build small, human-readable snippets:
   - `"Image {id}: entities = dog, ball, grass field, child"`
4. Optionally attach 1–2 original captions as additional context.

### 5.2 Image references

- For each selected image, return:
  - `image_id`
  - Path or URI (existing dataset path)
  - Optional thumbnail or precomputed features.

This context will be fed into a multimodal LLM in Phase 5.

---

## 6. Evaluation Plan (Phase 4)

We will evaluate four modes:

1. **CLIP-only baseline**
2. **Hybrid baseline (CLIP + BLIP-2, no KG)**
3. **KG + query enrichment, no BLIP-2** (CLIP + KG)
4. **Full hybrid: CLIP + BLIP-2 + KG + enrichment**

For each mode:

- Metrics:
  - R@1, R@5, R@10
  - MRR
  - Latency (median, p90) per query.
- Ablations:
  - With vs without **query enrichment**.
  - With vs without **graph expansion (H=0 vs H=2)**.
  - Different fusion weights `(w1, w2, w3)`.

We’ll select defaults that:
- Improve R@1 and/or MRR vs Hybrid baseline.
- Keep latency within the defined budget.

---

**Outcome:** After Phase 4, the system has an **entity-centric, LightRAG-style knowledge graph** tightly integrated into the existing CLIP/Hybrid pipeline, with **mandatory query enrichment** and **explicit graph expansion** as requested by the instructor. Images are retrieved via entity context rather than being graph nodes themselves, making the design more flexible for future cross-dataset extensions.
