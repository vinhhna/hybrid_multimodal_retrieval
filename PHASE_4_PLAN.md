# Phase 4 Plan — Knowledge Graph (LightRAG-Multimodal)

**Goal**: Build and integrate a multimodal knowledge graph that augments retrieval and provides rich, structured context for answer generation. This plan aligns with the instructor's pipeline and includes the minimal fixes identified earlier.

---

## 0. Scope & Success Criteria

### Scope (Phase 4 only):

* Graph building (nodes, edges, features) using PyTorch Geometric (preferred) or NetworkX for prototyping.
* Graph retriever (LightRAG-style): seeded subgraph selection + guided expansion (walk/cluster) with scoring and stopping rules.
* Query contextualization (simple enrichment) before retrieval.
* KG-based re-ranking signal combined with CLIP (+ optional BLIP-2).
* Context synthesizer stub that returns text + image payloads (file paths or base64) for Phase 5.
* Toggleable integration; CLIP-only and Hybrid baselines remain available.

### Success criteria:

* **Accuracy**: Graph-augmented retrieval matches or beats Hybrid on a top-rank metric (R@1 or MRR) without hurting Recall@10 (25-query smoke test).
* **Latency**: Graph retrieval adds ≤ 250 ms median over CLIP-only (target end-to-end ≤ 2.0 s in later phases).
* **Robustness**: Early exit and fallback to CLIP-only if limits are exceeded.
* **Completeness**: Context synthesizer stub produces `{texts, image_refs}` payloads for Phase 5.

---

## 1. Data & Encoders (Embedding Alignment)

### Encoders (now):

* **Text/Image**: CLIP (ViT-B/32) embeddings (already available).
* **Optional Stage-2**: BLIP-2 match scores (orientation-safe with Phase 3 fixes).

### Embedding alignment (required):

* Store a **CLIP-space vector for every node type used**:
  * **image**: CLIP image embedding.
  * **caption**: CLIP text embedding.
* **(Optional stub) region**: pooled patch/region embedding mapped to CLIP space; keep interface only for now.
* All seeding and similarity scoring operate in this **shared CLIP space**.

### Artifacts:

* `E_text[id]`, `E_image[id]` (L2-normalized).
* Optional `S_hybrid[q, img]` (used only if helpful).

---

## 2. Graph Builder (Multimodal Nodes/Edges — Minimal)

### 2.1 Node schema

* **Image node**: one per image; features = CLIP image embedding; attrs: `image_id`, `path`, `size`.
* **Caption node**: one per caption; features = CLIP text embedding; attrs: `caption_id`, `image_id`, `text`.
* **(Optional stub) Region node**: interface only; not required to ship in Phase 4.

### 2.2 Edge types

* **Semantic (similarity)**: top-k_sem neighbors by cosine within image↔image and caption↔caption, plus cross-type caption↔image links (paired items).
* **Co-occurrence**: caption↔image (paired in dataset), caption↔caption (same image).
* Degree cap per node (e.g., `k_sem ≤ 16`) to avoid hubs.

### 2.3 Storage

* **PyG Data/HeteroData**: `x` (or `x_dict`), `edge_index` (or per-type), `edge_weight` (fp16 where possible).
* **Maps**: `node_id → {type, image_id, caption_id}`, `image_id → path`.
* **Serialization**: `torch.save()`; quantize `edge_weight` to fp16.

---

## 3. Graph Retriever (Shallow Multi-hop + Efficient Local Traversal)

### 3.1 Query contextualization (simple enrichment)

* Encode the user query (text or image) with CLIP.
* From initial CLIP seeds (top-K_seed images/captions), gather a few neighbor captions/terms (e.g., top 5) and compose an **enriched text string** (e.g., "mèo đang ăn; thức ăn cho mèo; chén thức ăn"). Use this enriched string for the fast CLIP pass (toggleable).

### 3.2 Seeding (K-best in CLIP space)

* Select top-K_seed (default 10) nodes (images and/or captions) nearest to the query in CLIP space.
* Always include paired caption nodes for seeded images.

### 3.3 Guided expansion (LightRAG-style, shallow)

* **Multi-hop reasoning**: expand up to `H = 2` hops (support ≤ 3 later) using a beam of size `B = 20`.
* **Edge scoring**: `score(u→v) = parent_score × decay^hop × edge_weight`, with `decay ≈ 0.85`.
* **Edge-type weights**: semantic = 1.0, co-occurrence = 0.7 (multiply into `edge_weight` before decay).
* **Efficient traversal limits**: `N_max ≤ 200` collected nodes, `T_cap ≤ 150 ms`; avoid revisits.

### 3.4 Stopping / fallback

* Stop when `N_max` or `T_cap` reached or marginal gain < ε; if limits hit, reuse CLIP ranking (fallback).

---

## 4. Candidate Finalization (KG-based Re-ranking)

* Collect candidate **image nodes** from the explored subgraph.
* Compute a **KG proximity score** per candidate (lightweight):
  * **Option A (fast)**: shortest-path length from any seed (convert to `1/(1+dist)`).
  * **Option B (richer)**: personalized PageRank from seeds (use a few power iterations).
* Normalize signals and combine (orientation-safe):
  * `final = w1*clip_norm + w2*stage2_norm + w3*kg_norm`
  * defaults: `w1 = 0.7`, `w2 = 0.2` (only when agreement strong), `w3 = 0.1`
  * Keep **rank_fusion** as fallback when agreement between signals is weak.
* Output top-k images for evaluation.

---

## 5. Context Synthesizer (Stub for Phase 5)

**Input**: `(query, ranked_results, subgraph)`

**Output** (dict):
```python
{
    'texts': [],        # top captions/snippets (e.g., 5)
    'image_refs': [],   # top image paths or base64 URIs (e.g., 4)
    'metadata': {}      # {image_ids, caption_ids, edges_summary}
}
```

No LLM call in Phase 4; just packaging for later.

---

## 6. Latency & Accuracy Guardrails

* **Early exit**: if `|ρ(CLIP, BLIP-2)| < 0.05` (or Stage-2 unavailable), skip Stage-2.
* **Graph caps**: `H ≤ 2`, `B ≤ 20`, `N_max ≤ 200`, `T_cap ≤ 150 ms`; fallback to CLIP ranking if exceeded.
* **Fusion defaults**: prefer `rank_fusion`; apply weighted fusion only when signal agreement is strong; use conservative weights (e.g., 0.7/0.3 for Stage-1/Stage-2).
* **Metrics**: Recall@1/5/10, MRR, nDCG@10; latency mean/median.

---

## 7. Evaluation Plan

* **Dataset**: Flickr30K. Use 25-query smoke test for rapid iteration; optional 500-query set for deeper check.
* **Methods compared**: CLIP-only, Hybrid (fixed), Graph-augmented (with/without enrichment; with/without KG re-rank).
* **Reporting**: per-method summary + signed deltas vs CLIP-only.
* **Ablations**: 
  - (i) KG re-ranking on/off
  - (ii) enrichment on/off
  - (iii) K_seed, B, H, decay
  - (iv) edge-type weights
  - (v) Stage-2 on/off

---

## 8. Implementation Steps & Timeline (1–3 weeks)

### Week 1 — Design & Scaffolding

1. Finalize minimal node/edge schema (image, caption; region stub interface only).
2. Implement semantic k-NN and co-occurrence edges (+ degree caps).
3. Build PyG containers and serialization utils.
4. Add config hooks: K_seed, enrichment on/off, B, H, decay, N_max, T_cap, fusion policy.

**Deliverables**: schema docstrings, PyG builder, small saved sample graph.

### Week 2 — Graph Build & Retrieval

1. Batch-build graph over full dataset (chunked k-NN; cap degrees).
2. Implement seeded multi-hop expansion with beam, scoring, and limits.
3. Implement **KG-based re-ranking** and integrate with rank_fusion/weighted fusion (agreement-aware).

**Deliverables**: graph tensors on disk; `graph_search(query)` returning ranked images; time/memory logs.

### Week 3 — Context & Evaluation

1. Add **query contextualization** step (enrichment) with toggle.
2. Add context synthesizer stub (texts + image_refs).
3. Evaluate Graph mode vs baselines; tune thresholds/weights; short report.

**Deliverables**: results JSON, brief README snippet; example context payloads for 2–3 queries.

---

## 9. Risk & Mitigation

* **Latency creep**: enforce N_max/T_cap; prefer rank_fusion; reduce K_seed if slow.
* **Stage-2 noise**: keep correlation gating; allow Stage-2 off for graph experiments.
* **Memory growth**: degree caps; fp16 edge_weight; optional feature quantization.
* **Scope control**: concept/region nodes remain stubs; revisit only if needed.

---

## 10. Checklists

### Design
- [ ] Node types: image, caption (region stub optional)
- [ ] Edge types: semantic, co-occurrence
- [ ] Degree caps/thresholds set; CLIP-space alignment stated

### Build
- [ ] k-NN edges (chunked), co-occurrence edges
- [ ] PyG tensors saved (x, edge_index, edge_weight)
- [ ] Maps: node_id ↔ (image_id/caption_id), image_id → path

### Retriever
- [ ] Query contextualization toggle implemented
- [ ] Seed selection (K_seed) in CLIP space
- [ ] Beam expansion (B, H, decay) with limits (N_max, T_cap)
- [ ] KG-based re-ranking implemented (shortest-path or PPR)
- [ ] Fusion policy: rank_fusion default; weighted when agreement strong

### Context & Eval
- [ ] Context stub returns {texts, image_refs, metadata}
- [ ] Evaluator has Graph mode + ablations
- [ ] Metrics + latency logged; JSON saved

---

## 11. Stable Interfaces

### Build
```python
graph = build_multimodal_graph(encoders, cfg) -> PyG Data/HeteroData
save_graph(graph, path)
graph = load_graph(path)
```

### Retrieve
```python
results = graph_search(query, graph, encoders, cfg) -> List[(image_id, score)]
context = synthesize_context(query, results, graph, cfg) -> Dict
```

### Evaluate (Phase 4)
```python
evaluate_graph_mode(dataset, graph, cfg) -> metrics_dict
```

---

**Ready to implement.** Keep CLIP-only/Hybrid as fallback modes while iterating. After stabilizing the graph retriever and context stub, Phase 5 can plug in the multimodal LLM directly.
