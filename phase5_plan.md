# Phase 5 – Vision-Grounded Entity Extraction & Graph Upgrade

**Final Review Version (v3: Robustness + Binding-Aware Scoring)**

---

## 5.1 Scope and Objective

Phase 5 upgrades the existing **caption-only, entity-centric knowledge graph (KG)** into a **vision-grounded KG** for Flickr30K.

- **Task stays the same:** Natural-language question → ranked images (QA via retrieval).  
- **Core pipeline remains:** CLIP + FAISS (Stage 1) → KG reasoning (Stage 2) → BLIP‑2 re-ranking (Stage 3, optional).  
- **What changes in Phase 5:**  
  1) entities are **visually verified** via an open-vocabulary detector,  
  2) candidate generation is made **robust against caption omission and semantic drift**,  
  3) the KG includes a **vision co-occurrence** edge family with hub control,  
  4) KGScore becomes **two-stream (caption/vision)** with **globally calibrated mapping**, and  
  5) **attribute binding** is handled explicitly with **box-conditioned (Option C) scoring** when needed.

---

## 5.2. Motivation (Limitations of Phase 4)

Phase 4 builds:

- an entity vocabulary and embeddings from captions,
- an entity-only KG with:
  - semantic edges (`sem`) from embedding k-NN,
  - caption co-occurrence edges (`cooc_caption`),
- and uses this KG for query enrichment and graph search.

Key limitations addressed in Phase 5:

- **Caption sparsity:** captions omit background and secondary objects.
- **Caption-only co-occurrence:** edges reflect what humans mention, not what is visible.
- **Candidate echo chamber:** candidate entities derived only from captions can never be detected.
- **Silent failure risk #1 (confirmation bias):** detector can “confirm” hallucinated candidates if thresholds are too permissive.
- **Silent failure risk #2 (bag-of-entities):** summing entity scores cannot fix attribute binding (“red shirt” vs “red car”).
- **Silent failure risk #3 (semantic drift):** naive top-k neighbor expansion can drift into unrelated/fantasy concepts.

---

## 5.3. High-Level Design (Phase 5 Additions)

Phase 5 introduces six concrete upgrades:

1. **Vision-grounded entity evidence**  
   - Use an open-vocabulary detector (e.g., OWL‑ViT / OWLv2 / Grounding DINO) to confirm entities directly from pixels.
   - Store both **binary presence** and **continuous confidence**, plus **boxes** for object terms.

2. **Robust candidate generation (caption + global + CLIP visual prior + safe neighbors)**  
   - Retain caption entities and neighbors for efficiency, but add a **Stage‑0 visual prior** (CLIP image→entity retrieval).
   - Apply **safe neighbor filtering** to control semantic drift.

3. **Detector calibration to avoid “yes-man” confirmation bias**  
   - Use **per-entity (or grouped) thresholds**, **negative-control noise floors**, and a shaped confidence `Conf_eff`.

4. **Graph reconstruction with three edge families**  
   - `sem`, `cooc_caption`, and **new** `cooc_vision` (vision co-occurrence).

5. **Hub-controlled `cooc_vision` construction**  
   - PMI/NPMI filtering + degree caps + stop-node policy to prevent dense, uninformative hubs.

6. **Two-stream KGScore + binding-aware vision scoring (Option C)**  
   - Caption stream: as before (caption presence + IDF).
   - Vision stream:
     - generic: entity confidence evidence,
     - binding-aware: **box-conditioned attribute verification** for attribute-binding queries.

---

## 5.4. Vision-Grounded Entity Extraction

### 5.4.1. Candidate Entity Phrases per Image (Robust Candidate Set)

We construct a hybrid candidate list per image \(I\) that explicitly addresses caption omission and semantic drift.

**(1) Caption-derived candidates**  
Start from caption-linked entities \(E_{cap}(I)\) (Phase 4 `entity_context`).

**(2) Stage‑0 visual prior (fast, CLIP-based tagging)**  
To surface salient objects not mentioned by captions:

- Encode the image \(I\) into CLIP embedding \(v_I\) (already present in the retrieval pipeline).
- Retrieve the top-\(K_{vis}\) nearest entity embeddings \(\{x_e\}\) by cosine similarity:
  \[
  E_{vis}(I) = \text{TopK}_{e} \, \langle v_I, x_e \rangle
  \]
- Filter by similarity floor \(	au_{vis}\) and cap by budget.

**(3) Global baseline list (Top‑N frequent entities)**  
Add a Global Top‑N list of high-frequency scene primitives \(E_{global}\) to probe background/setting entities.

**(4) Safe semantic neighbors (drift-controlled expansion)**  
Instead of “Top‑20 neighbors unconditionally,” we define:

- Start from candidate neighbor set \(Nbr_k(e)\) for each \(e \in E_{cap}(I)\) (or optionally \(e \in E_{cap}(I) \cup E_{vis}(I)\)).
- Keep a neighbor \(n\) only if it passes **safe expansion filters** (configurable):

  **Filter set (recommended default):**
  - Similarity threshold: \(\cos(x_e, x_n) \ge \tau_{sim}\)
  - Mutual-kNN: \(e \in kNN(n)\) (reduces asymmetrical drift)
  - Vision support: \(df_{vis}(n) \ge min\_df\_vis\) (must be visually grounded in train)
  - Pixel anchoring (optional but strong): \(n \in E_{vis}(I)\) (neighbor must also be supported by the image visual prior)

This yields a drift-controlled neighbor set \(Nbr_{safe}(I)\).

**(5) Final candidate set**
\[
\text{candidates}(I) =
E_{cap}(I) \cup E_{vis}(I) \cup E_{global} \cup Nbr_{safe}(I)
\]

**Efficiency controls**
- Cap total candidates per image to \(C_{max}\).
- If over budget, keep by priority: caption entities > visual prior > safe neighbors > global list.

---

### 5.4.2. Open-Vocabulary Detection (OWL‑ViT) with Confidence + Boxes

For each image \(I\) and `candidates(I)`:

- Run the open-vocabulary detector with `image = I`, `text_queries = candidates(I)`.

For each entity \(e\):

- Gather all detection outputs (boxes, scores) for phrase \(e\).
- Define:

  - `vision_confidence[e, I] = max detection score for e in I` (0 if none).
  - `is_present_vision(e, I) = (vision_confidence[e, I] ≥ τ_present[e])`.

We store:

- binary presence for fast checks,
- continuous confidence for weighting evidence,
- and **bounding boxes** for object terms (required for binding-aware scoring in Section 5.6).

---

### 5.4.3. Anti-Confirmation Bias Controls (Detector is not a “Yes-Man”)

To prevent the detector from systematically confirming CLIP/caption hallucinations:

**(1) Per-entity / grouped thresholds**  
Use `τ_present[e]` (or group-level thresholds) calibrated on train/val to control false positives.

**(2) Negative-control noise floor per image**  
For each image \(I\), run a small, fixed set of **null/irrelevant phrases** \(E_{null}\) (e.g., 10–30):

- \(noise(I) = p95(\{conf(p,I) : p \in E_{null}\})\)
- Accept detection only if:
  \[
  conf(e,I) \ge \max(\tau_{present}[e],\ noise(I)+\delta)
  \]

**(3) Shaped confidence for downstream use**  
Define an “effective confidence” used everywhere downstream (aggregation, cooc_vision, coverage):

- \(Conf_{eff}(e,I)=\max(0, conf(e,I)-\tau_{eff}(e,I))\) where \(\tau_{eff}=\max(\tau_{present}[e], noise(I)+\delta)\)
- (Optional) use a sigmoid shaping: \(Conf_{eff}=\sigma((conf-\tau_{eff})/T)\)

**(4) Optional stability gate for rare/abstract entities**  
For entities with low \(df_{vis}(e)\), require detection stability across 2 lightweight augmentations (resize/flip) before accepting.

These controls reduce the chance that Phase 5 becomes a “confirmation loop” over biased candidates.

---

### 5.4.4. Updating `entity_context` and `entity_meta`

We extend Phase 4 schemas and keep caption and vision statistics separate.

#### `entity_context[entity_id]` additions
- `vision_image_ids`
- `vision_confidence[image_id]` (raw conf)
- `vision_conf_eff[image_id]` (optional but recommended; derived using Section 5.4.3)
- `vision_detections[image_id]` (optional): list of `{box, score}`

#### `entity_meta[entity_id]` additions
- `df_image_caption`
- `df_image_vision`
- optional confidence stats (mean, p90) over `vision_conf_eff` or raw conf

---

## 5.5. Graph Reconstruction (Multi-Edge Entity Graph)

### 5.5.1. Edge Types

1. **Semantic edges (`sem`)**  
   - k-NN in entity embedding space (degree-capped).

2. **Caption co-occurrence edges (`cooc_caption`)**  
   - co-occurrence from caption/image context; PMI-like or normalized weights.

3. **Vision co-occurrence edges (`cooc_vision`) with hub control**  
   - base signal: co-presence in the same image using `is_present_vision` (or better: `Conf_eff>0`).
   - compute PMI/NPMI from vision co-detection stats.
   - keep edges only if PMI/NPMI exceeds threshold.
   - apply degree caps per node.
   - stop-node policy: entities in Global Top‑N (or df-based top-M) require higher PMI to form edges.

We keep `cooc_caption` and `cooc_vision` separate for query-time weighting.

---

### 5.5.2. Graph Search Type Weights

Graph search supports:
- `type_weight_sem`
- `type_weight_cooc_caption`
- `type_weight_cooc_vision`

Optional hub penalty during traversal:
- down-weight expansion into high-df nodes (e.g., via IDF-based multiplier).

---

## 5.6. Entity → Image Aggregation (Two-Stream KGScore + Binding-Aware Vision Scoring)

Graph search outputs entity relevance scores \(S_e\). Phase 5 computes a per-image KGScore as a fusion of:

- **Caption stream**: robust for abstract/thematic cues.
- **Vision stream**: robust for concrete objects and attribute binding.

### 5.6.1. Caption Stream (unchanged)

\[
\text{IDF}_\text{cap}(e) = \frac{1}{\sqrt{df_\text{cap}(e) + \varepsilon}}
\]
\[
\text{score}_\text{cap}(I) =
\sum_{e} S_e \cdot \mathbf{1}[\text{is\_present\_caption}(e, I)] \cdot \text{IDF}_\text{cap}(e)
\]

---

### 5.6.2. Vision Stream (Generic Entity Evidence)

For non-binding queries, the vision stream can aggregate entity evidence:

\[
\text{IDF}_\text{vis}(e) = \frac{1}{\sqrt{df_\text{vis}(e) + \varepsilon}}
\]
\[
\text{score}_\text{vis,ent}(I) =
\sum_{e} S_e \cdot Conf_{eff}(e,I) \cdot \text{IDF}_\text{vis}(e)
\]

where \(Conf_{eff}\) is defined in Section 5.4.3.

---

### 5.6.3. Vision Stream (Binding-Aware, Option C — Box-Conditioned Verification)

For attribute-binding queries (e.g., “red shirt, green pants”), summing entities is insufficient. We introduce a binding-aware vision score computed from **object boxes**.

**Step A: Query slot extraction**  
Extract slot constraints of the form:
- \( (o, A) \) where \(o\) is an object (“shirt”) and \(A\) is a set of attributes (“red”).

Example:
- (shirt, {red})
- (pants, {green})

**Step B: Object box candidates**  
From OWL‑ViT detections, obtain top-\(B\) boxes \(B_o(I)\) for each object \(o\).

**Step C: Attribute verification inside the crop**  
For each crop \(crop(I,b)\), compute attribute evidence \(p(a|crop)\) using:
- **Color attributes:** fast HSV/histogram classifier (recommended default)
- **Non-color attributes:** CLIP-on-crop similarity to text prompt (optional extension)

Define binding evidence:
\[
E(o{+}a, I) = \max_{b \in B_o(I)} \Big(p(o|b) \cdot p(a|crop(I,b))\Big)
\]

**Step D: Slot satisfaction score**  
\[
\text{score}_{vis,bind}(I) = \sum_{(o,A)} w_{o,A} \cdot \max_{a \in A} E(o{+}a, I)
\]

**Step E (optional): assignment consistency**  
If multiple slots compete for the same box types, apply a small bipartite matching to prevent one region satisfying multiple slots.

**Switching rule**  
If the query contains binding slots, use `score_vis = score_vis,bind`. Otherwise use `score_vis = score_vis,ent`.

---

### 5.6.4. Stable Stream Mapping (Global Calibration) and KGScore

Phase 5 uses **global calibration** (fit on train/val, no test leakage):

\[
\widehat{\text{score}}_\text{cap}(I) = \sigma\left(\frac{\text{score}_\text{cap}(I) - \mu_\text{cap}}{\sigma_\text{cap}}\right)
\qquad
\widehat{\text{score}}_\text{vis}(I) = \sigma\left(\frac{\text{score}_\text{vis}(I) - \mu_\text{vis}}{\sigma_\text{vis}}\right)
\]

Then:
\[
\text{KGScore}(I) =
w_\text{cap} \cdot \widehat{\text{score}}_\text{cap}(I)
+ w_\text{vis} \cdot \widehat{\text{score}}_\text{vis}(I)
\]

---

### 5.6.5. Fusion with CLIP and BLIP‑2

\[
\text{FinalScore}(I) =
  w_\text{clip} \cdot \widehat{\text{clip}}(I)
+ w_\text{stage2} \cdot \widehat{\text{stage2}}(I)
+ w_\text{kg} \cdot \text{KGScore}(I)
\]

---

## 5.7. Dynamic KG Activation (Hybrid Confidence Gate)

KG is activated only when CLIP is uncertain or semantically misaligned.

### 5.7.1. CLIP confidence signals
- \(S_{max}\), \(S_{margin}\), \(S_{cons}\) → normalized → \(S_{clip}\)

### 5.7.2. Semantic coverage \(S_{cov}\)
From query enrichment, obtain \(E_q\). For top‑K images:

- presence for concrete terms uses `Conf_eff > 0` (vision) and/or binding slot satisfaction where relevant,
- abstract terms may fall back to caption presence.

### 5.7.3. Hybrid confidence and gating
\[
S_{hybrid} = \alpha S_{clip} + (1-\alpha) S_{cov}
\]

If \(S_{hybrid} \ge \tau\): baseline retrieval (little/no KG).  
If \(S_{hybrid} < \tau\): activate KG (graph_search + KGScore).

---

## 5.8. Evaluation & Analysis

### 5.8.1. Standard benchmark: Flickr30K (Karpathy split)
- R@1/R@5/R@10 across modes: `clip_only`, `hybrid`, `clip_kg`, `full`, plus gated variants.

### 5.8.2. Stress tests: compositional failures
- Winoground and/or ARO relational/compositional subsets (group score).

### 5.8.3. Grounding benchmark: Flickr30K Entities (future work)
- phrase localization recall with IoU thresholds.

### 5.8.4. Ablations & component analysis (updated)
1. **Static vs adaptive KG** (always-on vs gated; activation rate + latency).
2. **Candidate set ablation**
   - without visual prior vs with visual prior,
   - safe neighbors on/off (drift impact).
3. **Anti-confirmation bias ablation**
   - global τ only vs per-entity τ + noise floor + Conf_eff shaping.
4. **Binding ablation (required)**
   - bag-of-entities vision score (`score_vis,ent`) vs **Option C binding score** (`score_vis,bind`)
   - report on attribute-binding query subset (constructed from Flickr30K + hard negatives; and/or ARO).
5. **Hub control ablation**
   - naive `cooc_vision` vs PMI/NPMI + degree caps + stop-node policy.
6. **Normalization ablation**
   - per-query min–max vs global calibration.

### 5.8.5. Rescue rate + case studies
- “Rescue” = ground-truth outside top‑10 in `clip_only` but inside top‑5 in KG/gated mode.
- For each rescue, report:
  - which entities/edges contributed,
  - whether improvement came from vision co-occurrence vs binding-aware scoring vs gating.

---

This v3 document incorporates the three “silent failure” mitigations:
- **Anti-confirmation bias** controls for detector evidence,
- **Binding-aware Option C** scoring to address attribute binding,
- **Safe neighbor expansion** to prevent semantic drift contamination.
