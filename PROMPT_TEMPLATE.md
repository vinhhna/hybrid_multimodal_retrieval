You are ChatGPT. I will give you (a) my **Phase_4_Plan** document, (b) the **Phase 4 Implementation Plan** (21-day schedule), and (c) a specific **Day + bullet subset** that I want implemented now. Your job is to read those plans and produce **one single prompt for GitHub Copilot**, which I will paste into VS Code, so that Copilot can implement exactly those tasks (and nothing outside their scope).

High-level rules for you (ChatGPT):

- Always treat the Phase_4_Plan + Implementation_Plan as the **source of truth** about design and constraints.
- Assume the project is on branch `phase4-entity` and already has: CLIP-only + Hybrid search, BLIP‑2 reranker, and `build_entity_vocabulary.py` working on full Flickr30K.
- Your output must be **only** the final Copilot prompt (no extra commentary), with clearly labeled sections described below.
- When necessary, you may inline **small code snippets** inside the Copilot prompt to steer it.

### Inputs

* **Repo root**: `hybrid_multimodal_retrieval/`
* **Dataset paths**: images at `/kaggle/input/flickr30k/data/images`, captions at `/kaggle/input/flickr30k/data/results.csv`, full dataset root at `/kaggle/input/flickr30k/data`
* **Vector dim (CLIP text/image/entity)**: `512`
* **Plan files**: `PHASE_4_PLAN.md`, `IMPLEMENTATION_PLAN.md` (already summarized above; I will paste relevant parts as needed)
* **Scope to implement now**:
  * Paste the **exact Day header** from Implementation_Plan (e.g. `Week 2 — Query enrichment & graph search, Day 10–12: Graph expansion (multi-hop LightRAG-style)`)
  * Paste the **bullet points** under that day you want implemented (or the full day block)

### What you must produce (structure of the Copilot prompt)

Your Copilot prompt must contain the following sections, in this order:

1. **Top banner**
2. **Scope & file policy**
3. **Implementation contract (must-haves)**
4. **Functions & modules to implement**
5. **Algorithms & code snippets**
6. **Performance & safety constraints**
7. **Acceptance tests**
8. **Edit strategy & instructions for Copilot**

At the end, repeat the selected **Day header + bullets** verbatim to keep Copilot anchored.

Below is what each section should contain.

---

#### 1. Top banner

Explain to Copilot what this task is:

- Title: `Copilot task — Implement Phase 4 (selected items)`
- Briefly restate the **exact Day + bullets** to implement (quoted from Implementation_Plan).
- Clarify that Phase 4 uses the **entity‑centric graph design** from Phase_4_Plan (entities as only node type, images/captions off‑graph as context).

#### 2. Scope & file policy

Tell Copilot exactly where to work:

- List **existing files** to open/edit, with paths relative to `hybrid_multimodal_retrieval/` (examples; adapt as needed for the chosen day):

  - `src/graph/entities.py`
  - `src/graph/build_entity_graph.py`
  - `src/graph/graph_search.py`
  - `src/graph/context.py`
  - `scripts/build_entity_vocabulary.py`
  - `scripts/build_entity_graph.py`
  - `scripts/evaluate_phase4.py` (or similar evaluation script)
  - `config/entity_graph.yaml`

- If new files are allowed, list them explicitly (path + short description). If **not** allowed, include in the Copilot prompt:

  > Do **not** create new files. Modify only the files listed above.

- Require idempotent edits:

  > Edits must be idempotent: re-running Copilot with this prompt should not duplicate code, duplicate config entries, or break imports.

#### 3. Implementation contract (must-haves)

Here you translate the design from the plans into concrete requirements Copilot must follow. Include bullets for at least:

- **Config-driven design**

  - All hyperparameters and file paths live in YAML (or existing config helpers), not hard-coded in scripts.
  - Use these config sections and keys (adapt names if existing code already defines them):

    - `entity_graph`: `min_df`, `k_sem`, `degree_cap`, paths to `entity_vocab.json`, `entity_context.json`, `entity_embeddings.pt`, `entity_graph.pt`.
    - `query_enrichment`: `K_seed_raw`, `M_enrich`, enrichment templates, mode toggles for text/image.
    - `graph_search`: `K_seed`, `H_max`, `B`, `T_cap_ms`, `N_max`, `decay`, edge-type weights (`type_weight_sem`, `type_weight_cooc`), fusion weights.
    - `fusion`: `(w_clip, w_kg, w_blip2)` or equivalent.

  - Print the relevant config at runtime for transparency.

- **CLIP space alignment**

  - Images, captions, and **entities** all use the same CLIP model and embedding dimension.
  - Embeddings must be `float32`, L2-normalized, and NaN/Inf-free.
  - Entity embeddings are saved as `entity_embeddings.pt` (or `.npy`) with shape `[N_entities, d_model]`.

- **Entity-only graph schema (PyG)**

  - Graph is stored as `torch_geometric.data.HeteroData` with a single node type `"entity"`:

    - `data["entity"].x` — CLIP text embeddings for each entity.
    - `data["entity","sem","entity"].edge_index` / `edge_weight` — semantic edges (k-NN in embedding space).
    - `data["entity","cooc","entity"].edge_index` / `edge_weight` — co-occurrence edges (from `entity_context`).

  - Keep off-graph JSON metadata:

    - `entity_meta.json`: `entity_id → {name, df_caption, df_image, cf}`.
    - `entity_context.json`: `entity_id → {image_ids, caption_ids}`.
    - `image_db.json` (if present): `image_id → {path, caption_ids, clip_emb_index}`.

- **Semantic & co-occurrence edges**

  - Semantic edges: build via chunked k-NN over normalized entity embeddings, with `k_sem` neighbors, degree capping, and optional symmetricization.
  - Co-occurrence edges: connect entity pairs that co-occur in the same image/caption; weight proportional to count/PMI; apply `degree_cap` to avoid hub explosion.

- **Mandatory query enrichment**

  - Graph mode must **always** call `enrich_query` first (except for controlled ablations).
  - `enrich_query(query, dataset, encoders, entity_context, cfg)`:

    - Encode original query with CLIP → `q0`.
    - CLIP search over captions/images → top `K_seed_raw`.
    - Collect and score candidate entities using both frequency in seeds and similarity of entity embeddings to `q0`.
    - Select top `M_enrich` entities.
    - Build enriched text:
      - text query: `"{query}. Related: e1, e2, ..."`
      - image query: `"photo of e1, e2, e3, ..."`
    - Return both `q_enriched` and the list of enriched entities.

  - Graph search must seed entities based on `q_enriched`.

- **Graph expansion & scoring (LightRAG-style)**

  - Implement `graph_search(query, graph, encoders, cfg)`:

    - Call `enrich_query` → `q_enriched`.
    - Compute similarity between `q_enriched` and entity embeddings; pick top `K_seed` seed entities.
    - Initialize a max-heap frontier keyed by `score(node)`.
    - Expand up to `H_max` hops, at most `B` processed nodes, or until `T_cap_ms` deadline.

  - Scoring rule for each edge `u → v` at hop `h`:

    ```python
    score_v += score_u * (decay ** h) * edge_weight * type_weight
    ```

    - `decay ≈ 0.85`.
    - `type_weight` is `type_weight_sem` for semantic edges and `type_weight_cooc` for co-occurrence edges.

  - Maintain a dictionary of per-entity scores and track which entities/images are discovered at each hop.

- **Entity → image aggregation & fusion**

  - Map entity scores to image scores using `entity_context`:

    - For each `image_id`, aggregate scores from entities that appear in that image (e.g. sum over entities, optionally scaled by `1/sqrt(df)` or similar to downweight very common entities).

  - Combine:
    - CLIP similarity scores (`clip_score`),
    - KG-derived image scores (`image_kg_score`),
    - BLIP‑2 cross-encoder scores (when enabled),
    using normalized weighted sum:

    ```python
    score_final = w_clip * clip_norm + w_kg * kg_norm + w_blip2 * blip2_norm
    ```

  - Use simple per-query normalization (min–max or softmax) before combining.

- **Context synthesis API**

  - `synthesize_context(query, results, graph, cfg)` should take:
    - Original query + enriched form,
    - Ranked images and their entities,
    - Graph neighborhood information,
    and produce a small, human-readable JSON/dict with:
    - Top entities with scores,
    - Short textual explanation or “reasoning” chain: how entities/edges lead to images,
    - Pointers to image_ids / paths for debugging.

- **Determinism & reproducibility**

  - No random seeds unless explicitly configured; if used, must be set via config and logged.
  - Scripts must be idempotent: re-running should overwrite artifacts, not create duplicates.

#### 4. Functions & modules to implement

In this section, enumerate the exact functions/classes Copilot must create or modify, tying them to the plan. Examples (adapt depending on Day/bullets):

- `src/graph/build_entity_graph.py`:
  - `build_entity_graph(entity_embeddings, entity_context, cfg) -> HeteroData`
  - `build_semantic_edges(...) -> Tuple[edge_index, edge_weight]`
  - `build_cooccurrence_edges(...) -> Tuple[edge_index, edge_weight]`
  - `save_entity_graph(data, path) -> None`
  - `load_entity_graph(path) -> HeteroData`

- `src/graph/graph_search.py`:
  - `enrich_query(query, dataset, encoders, entity_context, cfg) -> EnrichmentResult`
  - `graph_search(query, graph, encoders, cfg) -> GraphSearchResult`
  - Optional helpers: `seed_entities(...)`, `expand_frontier(...)`, etc.

- `src/graph/context.py`:
  - `synthesize_context(query, results, graph, cfg) -> Dict[str, Any]`

- `scripts/build_entity_graph.py`:
  - CLI entrypoint that loads configs, calls `build_entity_graph`, and writes artifacts.

- `scripts/evaluate_phase4.py` (or similar):
  - Wraps evaluation for the different search modes (baseline vs KG).

In the Copilot prompt, always say:

> If these modules already exist with slightly different names or helpers, reuse and extend them instead of inventing new paths. Respect existing public APIs unless the plan explicitly says to change them.

#### 5. Algorithms & code snippets

Embed short, self-contained snippets or pseudocode to steer Copilot, such as:

- L2 normalization helper for embeddings.
- Beam/priority-queue loop for graph expansion with `decay` and `type_weight`.
- Entity → image aggregation from `entity_scores` to `image_scores`.
- Per-query min–max normalization and weighted fusion.
- Degree capping and edge pruning.
- Simple timing guard that enforces `T_cap_ms`.

These snippets should be small and self-explanatory, not full implementations.

#### 6. Performance & safety constraints

Remind Copilot of constraints from the plans:

- Graph search parameters:
  - `H_max` default 2 (optionally 3 for experiments),
  - `B ≤ 20`,
  - `N_max ≤ 200` collected entities,
  - `T_cap_ms ≤ 150` for graph work.

- Latency budget:
  - KG + enrichment adds **≤ 250 ms** median over CLIP-only, with hard fallbacks if exceeded.
  - BLIP‑2 may be disabled or truncated (fewer images) if latency is too high.

- Memory constraints:
  - Use chunked k-NN; never build dense `[N_entities, N_entities]` matrices.
  - Store edge weights in `float16` when safe; keep node embeddings as `float32`.
  - Avoid loading duplicate copies of large tensors.

Also reiterate:

- Do not break existing CLIP-only or Hybrid baselines. If something must change, keep old behavior behind a config flag.

#### 7. Acceptance tests

Describe what must be true for the implementation to be “done”:

- **Graph build sanity checks** on a small subset (e.g., 100 entities):
  - Non-empty edge sets for semantic and co-occurrence edges.
  - Reasonable degree distribution (no single node dominates after capping).

- **Search smoke test** (≈25 queries):
  - Run in 4 modes:
    1. CLIP-only
    2. Hybrid (CLIP + BLIP‑2, no KG)
    3. CLIP + KG (with query enrichment)
    4. CLIP + BLIP‑2 + KG (full Phase 4)
  - Record R@1, R@5, R@10, MRR, latency stats (median, p90) per mode.

- **Metrics artifact**:
  - Save a JSON like `results/phase4_eval.json` plus a short Markdown summary.
  - Log key config values used in the run (including fusion weights and graph_search config).

- **Context examples**:
  - Save 2–3 example context JSONs from `synthesize_context` and manually inspect for interpretability.

- **Idempotence**:
  - Re-running build/eval scripts should overwrite outputs cleanly and produce identical metrics (within numerical noise).

#### 8. Edit strategy & instructions for Copilot

Finally, tell Copilot how to approach the edits:

- Step 1: Open all listed files and skim existing code and docstrings.
- Step 2: Plan changes in comments (e.g., `# TODO(phase4-entity): ...`) before writing heavy code.
- Step 3: Implement functions in small, testable units.
- Step 4: Plug functions into scripts/entrypoints (e.g., `scripts/build_entity_graph.py`, evaluation script).
- Step 5: Run quick sanity checks (unit tests or small manual runs) and adjust if necessary.
- Step 6: Keep imports tidy and avoid circular dependencies.

---

### Output format (what you send back to me)

When you respond, you must output **only** the final Copilot prompt, no explanations.

- The first line of the Copilot prompt must be:

  ```text
  Copilot task: Implement Phase 4 items (from Phase_4_Plan — <paste day/bullets label here>)
  ```

- Then include the sections in this exact order:
  1. Top banner
  2. Scope & file policy
  3. Implementation contract
  4. Functions & modules
  5. Algorithms & snippets
  6. Performance & safety constraints
  7. Acceptance tests
  8. Edit strategy

At the end of the Copilot prompt, re-quote the **Selected Day/Bullets** verbatim.

---

**Selected Day/Bullets to implement now (paste verbatim):**

<PASTE THE EXACT DAY HEADER AND/OR BULLETS HERE>
