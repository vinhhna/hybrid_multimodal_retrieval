**TEMPLATE YOU SEND TO CHATGPT (fill in the placeholders before sending):**

---

You are ChatGPT. I will give you (a) my **Phase_4_Plan** document and (b) the **exact day / bullet points** I want implemented **now**. Your job: **generate one single, copy-pasteable prompt for GitHub Copilot** that will make it write (or modify) code to implement *only* those specified items, exactly as written in the plan. The Copilot prompt you return **must be so explicit that Copilot can’t do it wrong**. You may include small **code snippets** inside the Copilot prompt to steer it.

### Inputs

* **Repo root**: `hybrid_multimodal_retrieval/`
* **Dataset paths**: images at `/kaggle/input/flickr30k/data/images`, captions CSV at `/kaggle/input/flickr30k/data/results.csv`, full data set path at `/kaggle/input/flickr30k/data`
* **Vector dim**: `512`
* **Plan file**: Phase_4_Plan (attached file), Implementation_Plan (attached file)
* **Scope to implement now**:

  * Paste **exact Day header** (e.g., “Week 2 — Graph Build & Retrieval, Day 10–11: Implement guided expansion…”)
  * Paste **the bullet points** under that day you want implemented (or the entire day block)

### Your task (ChatGPT):

Produce **only** a single **Copilot prompt** (no explanations), with the following structure and constraints:

1. **Top banner**

   * Title: “Copilot task — Implement Phase 4 (selected items)”
   * Restate the **precise day/bullets** to implement (quote them).

2. **Change scope & file policy**

   * List the **exact files** to edit (paths are relative to `hybrid_multimodal_retrieval/`).
   * If creation of new files is **not allowed**, say: “**Do not create new files. Modify only the files listed.**”
   * If allowed, enumerate the new files by exact path and minimal contents to create.
   * Require **idempotent** edits (re-running Copilot shouldn’t duplicate code).

3. **Implementation contract (must-haves)**
   For each bullet you quoted, map to concrete code tasks with **function names, signatures, data types, and algorithms**. Include:

   * **Config keys & defaults** (e.g., `graph.k_sem=16`, `retriever.K_seed=10`, `retriever.B=20`, `retriever.H=2`, `retriever.decay=0.85`, `retriever.N_max=200`, `retriever.T_cap_ms=150`, `fusion.default="rank_fusion"`, `fusion.weighted_w=(w1=0.7, w2=0.2, w3=0.1)`, `enrichment.enabled=True`, `enrichment.top_terms=5`).
   * **CLIP-space alignment**: ensure **image/caption** embeddings are L2-normalized and stored consistently.
   * **Semantic edges** builder (chunked k-NN over normalized vectors, degree cap).
   * **Co-occurrence edges** builder (caption↔image pairs; optional caption↔caption per image).
   * **PyG containers** (`HeteroData` recommended): specify `x_dict`, `edge_index_dict`, `edge_weight` dtypes (fp16 where safe).
   * **Seed selection** in CLIP space, **query enrichment** (toggleable), **guided expansion** (H≤2, beam B, decay), **stopping rules** (N_max, T_cap_ms, ε).
   * **KG re-ranking** (shortest-path or small PPR within the subgraph), then **fusion** (rank_fusion by default; weighted only if agreement strong).
   * **Determinism**: honor a `FAST_SEED` when provided.
   * **Logging**: concise prints for seeds, hops, collected counts, and timings; no progress spam.

4. **Exact function & class specifications**
   Provide concrete headers to implement (or extend), including modules and docstrings. Example (adjust names/paths to the repo):

   * `src/graph/build.py`

     * `build_semantic_edges(X: np.ndarray, k_sem: int, degree_cap: int) -> Tuple[Tensor, Tensor]`
     * `build_cooccurrence_edges(dataset: Flickr30KDataset) -> Tuple[Tensor, Tensor]`
   * `src/graph/store.py`

     * `save_graph(g: HeteroData, out_dir: str) -> None`
     * `load_graph(out_dir: str) -> HeteroData`
   * `src/graph/search.py`

     * `encode_query(q: Union[str, np.ndarray], mode: Literal["text","image"]) -> np.ndarray`
     * `seed_nodes(q_emb: np.ndarray, K_seed: int) -> List[Seed]`
     * `enrich_query(q_text: str, seeds: List[Seed], g: HeteroData, top_terms: int) -> str`
     * `guided_expand(seeds: List[Seed], g: HeteroData, cfg: Dict) -> SubgraphResult`
     * `kg_rerank(cands: List[Candidate], g: HeteroData, cfg: Dict) -> List[Candidate]`
   * `scripts/evaluate_accuracy.py` (Graph mode integration)

     * `evaluate_graph_mode(...) -> Dict[str, Any]`
       If these modules already exist under different paths/names, **use those**; do **not invent** new APIs—adapt to existing project structure.

5. **Algorithms & snippets (inside the Copilot prompt)**
   Include short **illustrative snippets** to force intent, e.g.:

   * Min-max normalization with epsilon; L2 normalization; reciprocal-rank fusion (RRF with k=60); beam expansion pseudocode; shortest-path hop scoring; simple PPR (few power iterations on the collected subgraph only).
   * Example of **degree capping** after k-NN.
   * Example **timer** blocks to collect seed/expand/rerank timings.
   * Example **Spearman gating** for Stage-2 (skip or reduce weight when |ρ| < 0.15).

6. **Performance & safety constraints**

   * Respect caps (`H≤2`, `B≤20`, `N_max≤200`, `T_cap_ms≤150`).
   * Use `float16` for `edge_weight` where safe; avoid allocating full dense matrices.
   * Early exit to CLIP ranking if caps are hit.
   * No network calls; no heavy dependencies; keep imports consistent with repo.

7. **Acceptance tests (must pass)**

   * **Build tiny graph** (e.g., 100 images) → save/load round-trip with shapes/dtypes verified.
   * **Search smoke test** (25 queries):

     * Logs show seeds, hops, collected nodes, and time breakdown.
     * Returns non-empty ranked results for every query within caps.
   * **Metrics** printed: R@1/5/10, MRR, nDCG@10, mean/median latency; JSON saved with these fields only.
   * **Idempotence**: running the task twice does not duplicate code or config.

8. **Editing instructions**

   * List exact files to open and where to insert code (search anchors like function names or comments).
   * If refactors are needed, specify minimal, surgical edits.
   * Preserve public signatures unless explicitly told otherwise.

9. **Output format**

   * Return **only** the final **Copilot prompt** (no explanations).
   * The Copilot prompt must begin with:
     `Copilot task: Implement Phase 4 items (from Phase_4_Plan — <paste day/bullets label here>)`
   * Then include all sections above in the order: **Scope → Files → Implementation → Functions → Snippets → Constraints → Acceptance → Edit steps**.

---

**Selected Day/Bullets to implement now (paste verbatim):**
<PASTE THE EXACT DAY HEADER AND/OR BULLETS HERE>
