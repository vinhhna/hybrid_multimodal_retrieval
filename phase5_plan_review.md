# Phase 5 – Vision-Grounded Entity Extraction & Graph Upgrade  
**Final Review Version**

---

## 5.1. Scope and Objective

Phase 5 upgrades the existing **caption-only, entity-centric knowledge graph (KG)** into a **vision-grounded KG** for Flickr30K.

- **Task stays the same:**  
  Natural-language question → ranked images (QA via retrieval).
- **What changes:**  
  Entities and co-occurrence edges are now supported by **direct visual evidence** from an open-vocabulary detector, and the KG signal is integrated via a **balanced caption/vision scoring scheme** and a **dynamic KG activation gate**.

Phase 5 is positioned as **future work** on top of the Phase 4 system (CLIP + FAISS, BLIP-2 re-ranker, caption-derived entity graph, graph_search, and hybrid fusion).

---

## 5.2. Motivation (Limitations of Phase 4)

Phase 4 builds:

- an entity vocabulary and embeddings from captions,
- an entity-only KG with:
  - semantic edges (`sem`) from embedding k-NN,
  - caption co-occurrence edges (`cooc_caption`),
- and uses this KG for query enrichment and graph search.

Limitations:

- **Caption sparsity:** captions often omit background and secondary objects (“My cute pet” over a rich scene).
- **Caption-only co-occurrence:** co-occ edges reflect what humans *mention*, not necessarily what is *visible*.
- **Frequency imbalance:** common entities like `person`, `sky`, `tree` appear in many images but not always in captions.
- **KG signal scaling:** naïve use of document frequency can over-penalize visual signals if df_image_vision is much larger than df_image_caption.

Phase 5 explicitly addresses these issues.

---

## 5.3. High-Level Design

Phase 5 introduces three main changes:

1. **Vision-grounded entity evidence**  
   - Use an open-vocabulary detector (e.g., OWL-ViT / Grounding DINO) to confirm entities directly from pixels.
   - Store both **binary presence** and **continuous vision confidence** per entity–image pair.

2. **Vision-aware co-occurrence edges**  
   - Add a second co-occurrence edge type (`cooc_vision`) based on visual co-presence.
   - Keep `cooc_caption` and `cooc_vision` separate for query-time weighting.

3. **Two-stream entity→image aggregation**  
   - Compute separate **caption** and **vision** KG scores per image.
   - Normalize each stream independently, then fuse with tunable weights.
   - Integrate this KG score with CLIP and BLIP-2 in the hybrid retrieval pipeline.
   - Optionally gate KG activation per query via a **hybrid query performance predictor**.

---

## 5.4. Vision-Grounded Entity Extraction

### 5.4.1. Candidate Entity Phrases per Image

Brute-forcing ~12k vocabulary entries per image is intractable; using only caption entities creates an “echo chamber.” We design a **hybrid candidate list** per image \(I\):

1. **Caption-derived candidates**
   - Start from entities already associated with \(I\) in `entity_context` based on captions.
   - For each such entity, retrieve its top-k semantic neighbors in entity embedding space (e.g., k = 20).
   - Merge and deduplicate.

2. **Global baseline list (Top-N frequent entities)**
   - Precompute a **Global Top-N** list of high-frequency entities (e.g., N = 50; typical: `person`, `man`, `woman`, `child`, `dog`, `car`, `grass`, `tree`, `sky`, `street`, `building`, etc.).
   - Add this list to every image’s candidate set.

3. **Final candidate set**
   - `candidates(I) = (caption_entities(I) ∪ semantic_neighbors) ∪ global_top_N`.

This guarantees coverage of both **foreground** entities and **background / setting** entities even when captions are vague.

---

### 5.4.2. Open-Vocabulary Detection with Vision Confidence

For each image \(I\) and `candidates(I)`:

- Run the open-vocabulary detector with:
  - `image = I`
  - `text_queries = candidate entity names`.

For each entity \(e\):

- Collect all detector outputs (boxes, scores) for query \(e\).
- Define:

  - `vision_confidence[e, I] = max detection score for e in I` (0 if no detection).
  - `is_present_vision(e, I) = (vision_confidence[e, I] ≥ τ_present)` for some threshold \(τ_\text{present}\) (e.g., 0.25–0.3).

We **retain both**:

- a fast binary presence flag, and  
- a continuous confidence value to weight evidence downstream.

Bounding boxes `(x1, y1, x2, y2)` can also be stored for potential spatial reasoning but are not required for the core Phase 5 design.

---

### 5.4.3. Updating `entity_context` and `entity_meta`

We extend the Phase 4 schemas; we **separate caption-based and vision-based statistics.**

#### `entity_context[entity_id]`

Caption side (Phase 4):

- `image_ids`: images where the entity appears in captions.
- `caption_ids`: associated caption indices, etc.

Vision side (Phase 5 additions):

- `vision_image_ids`: images where `is_present_vision(e, I)` is true.
- `vision_confidence[image_id]`: stored `vision_confidence[e, I]`.
- (Optional) `vision_detections[image_id]`: list of `{box, score}`.

#### `entity_meta[entity_id]`

- `df_image_caption`: number of images whose captions mention the entity.
- `df_image_vision`: number of images where the entity is visually detected.
- Optional summary stats of `vision_confidence` (mean, max).

We **do not merge** these document frequencies; they serve as independent normalization factors for caption and vision streams.

---

## 5.5. Graph Reconstruction (Multi-Edge Entity Graph)

The Phase 4 entity graph is extended but not fundamentally changed in structure.

### 5.5.1. Edge Types

1. **Semantic edges (unchanged)**  
   - Type: `("entity", "sem", "entity")`  
   - Construction: k-NN in CLIP entity embedding space (symmetrized, degree-capped).  
   - Purpose: propagate relevance along semantic similarity.

2. **Caption co-occurrence edges**  
   - Type: `("entity", "cooc_caption", "entity")`  
   - Construction: entities co-occurring in captions / caption-based image context.  
   - Edge weight: function of caption co-occurrence count, normalized and frequency-corrected (e.g., PMI-like or normalized by `df_image_caption`).

3. **Vision co-occurrence edges (new)**  
   - Type: `("entity", "cooc_vision", "entity")`  
   - Construction: entities co-detected in the same image:

     - both `is_present_vision(e1, I)` and `is_present_vision(e2, I)` true.

   - Edge weight: function of visual co-occurrence count across images, normalized by `df_image_vision` to down-weight ubiquitous entities and optionally modulated by average joint `vision_confidence`.  
   - Future extension (optional): incorporate spatial affinity (IoU / relative distance between boxes) to emphasize interacting entities.

We **keep `cooc_caption` and `cooc_vision` as distinct edge types**. This allows us to adjust type weights per query (e.g., rely more on `cooc_vision` for concrete object-centric queries, more on `cooc_caption` for abstract themes).

### 5.5.2. Type Weights in Graph Search

Graph search is updated to support three edge families:

- `type_weight_sem`
- `type_weight_cooc_caption`
- `type_weight_cooc_vision`

These are read from `configs/entity_graph.yaml` and can later be modulated by query features (e.g., concrete vs abstract).

---

## 5.6. Entity → Image Aggregation (Two-Stream Caption/Vision Scoring)

Graph search produces an entity relevance score \(S_e\) for each visited entity \(e\). Phase 5 defines a **two-stream** mapping from entity scores to per-image KG scores.

Let:

- \(\mathcal{I}\): candidate image set for the query (e.g., Stage-1 / Stage-2 images).
- \(E(I)\): entities linked to image \(I\) via caption and/or vision context.

We compute a **caption stream** and a **vision stream** and then fuse them.

### 5.6.1. Caption-Based Stream

For each entity \(e\) and image \(I\):

- Caption presence: `is_present_caption(e, I)` from caption context.
- Caption df: \(df_{\text{cap}}(e) = df\_image\_caption[e]\).

Define a caption IDF term:

\[
\text{IDF}_\text{cap}(e) = \frac{1}{\sqrt{df_\text{cap}(e) + \varepsilon}}
\]

Caption score for image \(I\):

\[
\text{score}_\text{cap}(I) = \sum_{e \in E(I)} S_e \cdot \mathbf{1}[\text{is\_present\_caption}(e, I)] \cdot \text{IDF}_\text{cap}(e)
\]

### 5.6.2. Vision-Based Stream

For vision:

- Vision df: \(df_{\text{vis}}(e) = df\_image\_vision[e]\).
- Vision confidence per image: \(\text{Conf}_\text{vis}(e,I) = \text{vision\_confidence}[e, I]\).

Define a vision IDF term:

\[
\text{IDF}_\text{vis}(e) = \frac{1}{\sqrt{df_\text{vis}(e) + \varepsilon}}
\]

Vision score for image \(I\):

\[
\text{score}_\text{vis}(I) = \sum_{e \in E(I)} S_e \cdot \text{Conf}_\text{vis}(e,I) \cdot \text{IDF}_\text{vis}(e)
\]

This formulation:

- boosts entities with high graph relevance and high detection confidence,
- down-weights visually ubiquitous entities via `df_image_vision`.

### 5.6.3. Stream Normalization and KGScore

To avoid one stream dominating due to scale, we **normalize each stream per query** over the candidate image set \(\mathcal{I}\):

\[
\widehat{\text{score}_\text{cap}}(I) = \text{Norm}_\mathcal{I}\big(\text{score}_\text{cap}(I)\big)
\]
\[
\widehat{\text{score}_\text{vis}}(I) = \text{Norm}_\mathcal{I}\big(\text{score}_\text{vis}(I)\big)
\]

where `Norm` can be min–max or z-score.

The final **KGScore** for image \(I\):

\[
\text{KGScore}(I) = w_\text{cap} \cdot \widehat{\text{score}_\text{cap}}(I)
                  + w_\text{vis} \cdot \widehat{\text{score}_\text{vis}}(I)
\]

with `w_cap` and `w_vis` configured (and potentially query-dependent in future work).

### 5.6.4. Fusion with CLIP and BLIP-2

`KGScore(I)` plugs into the existing hybrid fusion:

\[
\text{FinalScore}(I) =
  w_\text{clip} \cdot \widehat{\text{clip}}(I)
+ w_\text{stage2} \cdot \widehat{\text{stage2}}(I)
+ w_\text{kg} \cdot \widehat{\text{KGScore}}(I)
\]

where each component is normalized across candidate images and the weights depend on the selected mode (`clip_only`, `hybrid`, `clip_kg`, `full`).

---

## 5.7. Dynamic KG Activation via Hybrid Query Performance Prediction (QPP)

Phase 5 also introduces an **adaptive KG activation** mechanism, so KG is only used when CLIP is likely struggling.

### 5.7.1. CLIP Behaviour Signals

After Stage 1 CLIP retrieval (top-K images, scores \(s_1 \ge \ldots \ge s_K\), embeddings \(v_1..v_K\)):

1. **Max score:** \(S_\text{max} = s_1\) – absolute similarity of the top result.  
2. **Margin:** \(S_\text{margin} = s_1 - s_2\) – decisiveness between top-1 and top-2.  
3. **Visual consistency:**  

   \[
   S_\text{cons} = \text{mean}_{i \neq j} \cos(v_i, v_j), \quad i,j \in \{1..5\}
   \]

   – coherence among top images.

Normalize to \([0,1]\) and define:

\[
S_\text{clip} = w_\text{max}\hat S_\text{max} + w_\text{margin}\hat S_\text{margin} + w_\text{cons}\hat S_\text{cons}
\]

### 5.7.2. Vision-Grounded Semantic Coverage

From query enrichment, we have top query entities \(E_q\).

For each \(e \in E_q\) and top-K images \(\{I_1..I_K\}\):

- Use a **synonym safety net**: extend \(e\) to `{e} ∪ neighbors(e)` where neighbors are high-similarity entities in the graph.
- For **concrete** entities: presence is decided via visual detections (or neighbors) above a confidence threshold.  
- For **abstract** entities: presence can fall back to caption match only.

Coverage per entity:

\[
\text{coverage}(e) = \frac{\#\{ I_k \text{ where } e \text{ (or neighbor) is present} \}}{K}
\]

Semantic coverage:

\[
S_\text{cov} = \frac{1}{|E_q|} \sum_{e \in E_q} \text{coverage}(e)
\]

### 5.7.3. Hybrid Confidence and Gating

Hybrid confidence:

\[
S_\text{hybrid} = \alpha S_\text{clip} + (1-\alpha) S_\text{cov}
\]

Gating rule (conceptual):

- If \(S_\text{hybrid} \ge \tau\):  
  CLIP is reliable → use `clip_only` / `hybrid` (low or zero `w_kg`).
- If \(S_\text{hybrid} < \tau\):  
  CLIP uncertain / semantically misaligned → activate KG:
  - run graph_search over `sem`, `cooc_caption`, `cooc_vision`,
  - compute `KGScore` via two-stream aggregation,
  - use `clip_kg` / `full` mode with higher `w_kg`.

A stricter 2-D gate (e.g., separate thresholds on `S_clip` and `S_cov`) is also possible.

---

## 5.8. Evaluation & Analysis

To validate the design, we evaluate across three layers plus ablations.

### 5.8.1. Standard Benchmark – Flickr30K (Karpathy Split)

- **Dataset:** Flickr30K with Karpathy split (≈29k train / 1k val / 1k test).  
- **Task:** each test caption is a query; its paired image is the ground-truth relevant image.  
- **Metric:** Recall@K (K ∈ {1, 5, 10}).  
- **Compared modes:** `clip_only`, `hybrid`, `clip_kg`, `full`.  
- **Goal:** `clip_kg` / `full` maintain or improve R@K versus `clip_only` / `hybrid`.

KG is built only from train (+ optionally val) to avoid test leakage.

### 5.8.2. Stress-Test Benchmarks – Compositional Failures

- **Datasets:** Winoground and/or relevant ARO subsets (relation/order).  
- **Metric:** group score (both captions matched to the correct images in each group).  
- **Compared methods:** CLIP baseline vs CLIP+KG modes.  
- **Goal:** show KG-based reasoning yields higher group scores on compositional/relational examples where CLIP is known to fail.

### 5.8.3. Phase 5 Grounding Benchmark – Flickr30K Entities (Future Work)

- **Dataset:** Flickr30K Entities (bounding boxes linked to phrases).  
- **Task:** when our KG claims entity \(e\) is present in image \(I\), check overlap with annotated regions.  
- **Metric:** phrase localization recall (e.g., Recall@K or recall at IoU ≥ threshold).  
- **Goal:** demonstrate that vision-based entity presence is accurate and that the KG is truly vision-grounded.

### 5.8.4. Ablation & Component Analysis

1. **Static vs Adaptive KG**

   - Compare always-on KG vs gated KG (based on \(S_\text{hybrid}\)).  
   - Report R@K, KG activation rate, and approximate latency.  
   - Desired outcome: adaptive KG achieves similar/better R@K with lower KG usage.

2. **Concrete vs Abstract Query Breakdown**

   - Partition queries into **concrete** (object/scene-centric) vs **abstract** (themes/feelings).  
   - Compare caption-only KG, vision-only KG, full two-stream KG, and `clip_only`.  
   - Hypothesis: vision-grounded edges and the vision stream benefit concrete queries; caption edges benefit abstract queries.

3. **KG “Rescue Rate” and Case Studies**

   - Count cases where the correct image moves from outside Top-10 under `clip_only` to inside Top-5 under KG modes.  
   - Present qualitative examples highlighting:
     - the query,  
     - before/after rankings,  
     - key entities and edges involved.

These analyses verify not only that Phase 5 improves performance in the right scenarios, but also **why** it works and how the KG, vision grounding, and dynamic gate contribute.

---

This document is the consolidated **Phase 5 Plan (review version)** for the project.
