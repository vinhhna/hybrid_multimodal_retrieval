IMMEDIATE PLAN: Short-term technical plan to address critical gaps
===============================================================

Purpose
-------
This document lists a set of immediate, high-impact technical tasks you should implement next in the project. It focuses on four urgent gaps you identified:

1. Replace rule-based heuristics for query classification with an LLM-driven approach.
2. Design and implement LLM routing with three options (All-in-one, Router, Planner) and decide a first MVP.
3. Turn Questions into usable assets (evaluation / training / prompts / few-shot examples).
4. Urgently start using images in the pipeline (minimal working integration and progressive improvements).

Checklist (what I'll cover)
---------------------------
- Goal for each problem
- Concrete design options and recommended MVP
- Files/modules to add or change
- Tests / validation to add
- Short timeline (1-week sprint) and risks

1) Replace heuristics with an LLM for processing input queries
-------------------------------------------------------------
Goal
: Use an LLM to map a raw natural-language query into structured intent + parameters (the role currently played by `gqa_nl_parser.py`), improving coverage and reducing brittle regex rules.

Why
: LLMs generalize well to varied phrasing and reduce manual pattern maintenance. They can also return structured JSON that is easier to use downstream.

MVP recommendation
: Add an LLM-backed parser fallback (or primary) with a prompt that asks for a JSON response with fields: {"query_type": <one of 9 types or 'unknown'>, "params": {...}, "confidence": <0..1>, "explain": "string"}.

Implementation notes
- New module: `src/llm_parser.py` with an adapter class `LLMParser`.
- `LLMParser.parse(query: str) -> ParseResultLike` where ParseResultLike matches the current parser return shape (so `QueryInterface` can accept it).
- Keep `gqa_nl_parser.py` as a fast local fallback or for unit-testing; progressively migrate to LLM as acceptable.
- Prompt template must include: (1) the 9 canonical query types; (2) exact output schema; (3) a few 5–12 in-repo examples (few-shot) pulled from Questions (see section 3).
- Prefer models: local LLM (Llama 2 / Mistral / Falcon) if privacy/offline required, or OpenAI Chat* if accessible. Abstract the model behind an interface (adapter pattern) so you can swap providers.

Files to add / change (MVP)
- Add: `src/llm_parser.py` (adapter + prompt templates)
- Modify: `src/gqa_query_interface.py` to call `LLMParser` first (or switch based on a flag `--parser=llm|rules`) and to gracefully handle malformed JSON.
- Add: `tests/test_llm_parser.py` (mock the LLM adapter) to assert JSON structure and fallback behavior.

Validation / tests
- Unit test for JSON schema compliance (happy path + malformed LLM output handling).
- Smoke integration: run `QueryInterface` with `--parser llm` and an offline mocked LLM to validate pipeline.

Risks
- LLM hallucination or inconsistent output shapes — mitigate by schema enforcement and verification logic.
- Cost/latency if using a remote API — mitigate via caching and micro-batching.

2) LLM routing: three design options and recommended MVP
-------------------------------------------------------
Problem statement
: You want to route a query to the appropriate reasoning/solver. There are three designs:
  A) All-in-one LLM
  B) LLM Router (classify & dispatch to 9 query handlers)
  C) LLM Planner (planner that issues sub-queries and composes answers / tool-calls)

Pros/cons summary
- All-in-one LLM
  - Pros: simplest to implement; LLM handles everything end-to-end (parse, plan, answer). Good for prototypes.
  - Cons: Expensive at scale; hard to ground on graph data; opaque reasoning; poor reproducibility.

- LLM Router
  - Pros: Middle ground — LLM classifies intent and parameters; you still use the deterministic KG solvers for execution (best of both worlds).
  - Cons: Still depends on LLM classification quality; requires robust schema enforcement.

- LLM Planner (tool-calling)
  - Pros: Most powerful. LLM reasons, decomposes queries, calls graph solvers, composes multi-hop answers. Best for complex reasoning.
  - Cons: More engineering (tool API design, retry, tool security), higher latency and complexity.

MVP recommendation
: Implement LLM Router first (option B). It gives immediate quality gains while keeping deterministic, debuggable execution in the existing `GQA_Reasoning_Engine`.

Implementation plan for Router MVP
- Add module: `src/llm_router.py` that exposes `route(query: str) -> RoutingResult` where RoutingResult contains: `target_type` (one of 9), `params` (normalized), `confidence`, `explain`.
- Integrate with `QueryInterface.query()` to accept either router output or fallback to rule parser.
- Add a lightweight `RouterEvaluator` script to compute classification accuracy on a labeled set of Questions.

Files to add / change
- Add: `src/llm_router.py` (router adapter + prompt)
- Add: `scripts/router_eval.py` - run batch queries and measure top-1/type accuracy & confusion matrix
- Modify: `src/gqa_query_interface.py` to allow `--router=llm|rules` and to call router before execution
- Add: `tests/test_router.py` with mocked LLM responses

Validation / tests
- Create small labeled dataset (see section 3). Evaluate precision/recall; iterate with prompt engineering.

Risks
- Misclassification still possible — keep a fallback path to rule-based parser and/or human-in-loop.

3) Use Questions as evaluation / training data
--------------------------------------------
Goal
: Turn Questions (your example queries) into structured assets for evaluation, few-shot examples, and potential fine-tuning or retrieval-augmented prompts.

Actions
- Curate a dataset `data/questions.jsonl` with fields: `id`, `query`, `gold_type`, `gold_params`, `gold_answer_id(s)` (if available), `notes`.
- Use them for:
  - Router training/eval (classification label = gold_type)
  - LLM parser few-shot examples (inject 4–12 examples into prompts)
  - Create an evaluation harness `scripts/evaluate_retrieval.py` to compute Recall@k, MRR, accuracy on sample queries.
- Use LLMs to augment Questions: generate paraphrases (few per query) to increase robustness; verify paraphrases via a vetting script or human review.

Files to add / change
- Add: `data/questions.jsonl` (start with 200–1,000 hand-labeled examples) — commit the metadata only; keep full labeled sets externally if sensitive.
- Add: `scripts/generate_paraphrases.py` (uses LLM to create paraphrases) — outputs candidates to `data/` with human vetting flags.
- Add: `scripts/router_eval.py` (see earlier) and `scripts/evaluate_retrieval.py`.

Validation / tests
- Run `scripts/router_eval.py` with `--limit 200` and produce a confusion matrix and list of failed cases for prompt improvements.

4) Urgently start using images inside the project
------------------------------------------------
Problem
: Current project is purely graph/text-based; you must integrate images urgently but don't know how to start.

Priority constraint
: Start small and practical — minimal integration that yields immediate value; then iterate to richer multimodal reasoning.

Immediate minimal options (pick one MVP)
- MVP A (fastest, smallest changes): Use CLIP (OpenAI CLIP / OpenCLIP) to compute global image embeddings and index them. For each returned graph result, return the image embedding distance as an additional signal to re-rank. This quickly stitches images into retrieval without changing KG.
- MVP B (richer): Use a vision-language captioner (BLIP/BLIP-2) to produce text captions for each image and insert captions as additional instance-level chunks into the KG. The NL parser and reasoning engine will then naturally access that text.
- MVP C (object-level, higher fidelity): Extract object crops using bounding boxes from sceneGraphs, run a multimodal model (CLIP/DETR/ViT/Segmenter) to get per-object embeddings or captions and attach them to instance nodes in the graph. This requires more computation but gives the most precise grounding.

MVP recommendation (urgent)
: Implement MVP A + B in two steps: first implement CLIP embeddings and index (quick), then add BLIP captioning to enrich KG nodes (next day). This gives immediate image-signal for ranking and a path to grounded text.

Concrete implementation: step-by-step
1. Image discovery
   - Where images live: if you have local image files named by image_id, point a config to that folder. If not, add `data/image_manifest.csv` with `image_id,image_path` mapping.
2. CLIP embedding service (fast)
   - Add module: `src/image_embeddings.py` using `open_clip` or `sentence-transformers` (clip) to compute 512–1024-d vectors.
   - Add script: `scripts/index_image_embeddings.py --images data/image_manifest.csv --out output/clip_index.faiss` that computes embeddings in batches and stores FAISS (or hnswlib) plus a mapping `image_id -> vector_id`.
   - During query: after KG-based candidate list is produced, compute query-text embedding (CLIP text encoder) and run ANN search to retrieve nearest images or re-rank candidate images by a combined score (KG_score * alpha + clip_score * (1-alpha)).

3. BLIP captioning to enrich KG (next step)
   - Add module `src/image_caption.py` using Hugging Face `microsoft/` or `Salesforce/blip-` models to produce captions per image or per crop.
   - Run `scripts/add_captions_to_graph.py --graph experiments/sample_10k/gqa_lightrag.gpickle --manifest data/image_manifest.csv --out experiments/sample_10k/gqa_lightrag_with_captions.gpickle` which will add a `caption` attribute or a new node type `caption_chunk` linked to instance/image nodes.

Files to add / change (images)
- Add: `src/image_embeddings.py` (CLIP adapter)
- Add: `src/image_caption.py` (optional BLIP wrapper)
- Add: `scripts/index_image_embeddings.py`, `scripts/add_captions_to_graph.py`
- Add: `data/image_manifest.csv` (image_id,image_path)
- Add tests: `tests/test_image_indexing.py` (mock images or tiny real images) and `tests/test_caption_integration.py`.

Validation / tests
- Small smoke run on 100 images: build embeddings, add to FAISS, run a combined query that uses KG and CLIP re-ranking. Verify top-5 images are returned and pipeline completes in reasonable time.

Operational concerns
- Storage for embeddings and indices; batch processing to avoid OOM.
- GPU acceleration recommended for BLIP / CLIP encoding but CPU-only CLIP exists (slower).
- Caching and reproducible seeds.

Security, cost & privacy
- If you use hosted APIs (OpenAI), confirm data privacy and costs. For images, prefer local models where possible.

7-day immediate timeline (detailed)
---------------------------------
Day 0 (today, quick, 0.5–1 day)
- Add `IMMEDIATE_PLAN.md` (this file) and create `data/questions.jsonl` skeleton with 50 example questions.
- Stub `src/llm_parser.py` and `src/llm_router.py` skeletons (no network calls) and add unit-test harness with mocked responses.

Day 1 (1 day)
- Implement LLM Router MVP using a configurable LLM adapter (mockable). Integrate into `gqa_query_interface.py` with flag `--router`.
- Add `scripts/router_eval.py` and a small labeled dataset to `data/questions.jsonl` and run evaluation.

Day 2 (1 day)
- Implement `src/image_embeddings.py` using CLIP and `scripts/index_image_embeddings.py`. Create `data/image_manifest.csv` for 100 sample images.
- Smoke test: compute embeddings for 100 images and store index.

Day 3 (1 day)
- Integrate CLIP re-ranking into `QueryInterface.query()` as an optional post-ranker with weight `--alpha`.
- Add tests for combined KG+CLIP ranking.

Day 4–5 (2 days)
- Add BLIP captioning integration to enrich KG nodes and re-run builder to add captions for sample graphs.
- Update parser few-shot examples to include caption-derived examples.

Day 6–7 (2 days)
- Iteratively tune prompts, run `router_eval.py` and `evaluate_retrieval.py`, fix failure modes, add more Questions/paraphrases.
- Finalize tests and small CI workflow to run smoke tests.

Minimum viable deliverables (end of week)
- `src/llm_parser.py` and `src/llm_router.py` (router used by default)
- `scripts/router_eval.py` and `data/questions.jsonl` (>=200 examples preferred; start with 50)
- `src/image_embeddings.py` and `scripts/index_image_embeddings.py` plus `data/image_manifest.csv` (100-sample)
- Integration in `src/gqa_query_interface.py` for optional re-ranking
- Unit tests for parser/router and image indexing

Practical prompts & schema examples (copyable)
----------------------------------------------
Example: router prompt (short form) — ask model to return only JSON with this exact schema:

```json
{
  "query_type": "entity_search|statistical_knowledge|similarity_search|relational_path|negative_constraints|comparative|hierarchical|anomaly_detection|visual_attribute_constraint|unknown",
  "params": { /* typed parameters depending on query_type */ },
  "confidence": 0.0,
  "explain": "short human-readable explanation of mapping"
}
```

Use a few-shot list of 6–12 examples (use `data/questions.jsonl`) appended to the prompt.

Quick terminal commands (copy-paste)
-----------------------------------
Install minimal extras for image work and testing (use a virtualenv):

```bash
pip install -r requirements.txt
pip install open-clip-torch faiss-cpu sentence-transformers # or equivalent
pip install transformers accelerate ftfy # for BLIP/vision models
```

Index a small image set (example):

```bash
python scripts/index_image_embeddings.py --manifest data/image_manifest.csv --out output/clip_index.faiss --batch 32
```

Run the router evaluation (example):

```bash
python scripts/router_eval.py --questions data/questions.jsonl --router llm --limit 200
```

Notes, risks and acceptance criteria
-----------------------------------
- Acceptance: router classification accuracy > 85% on an initial labeled set; image re-ranking improves top-5 recall in human spot-checks; integration tests pass.
- Risks: LLM hallucinations and inconsistent JSON; mitigate with strict validation, schema enforcement, and safe fallback to rules.
- If a hosted LLM is used, add budget notes and monitoring for latency/cost.

Next immediate action I will take if you want me to proceed
---------------------------------------------------------
- Add the skeleton modules (`src/llm_parser.py`, `src/llm_router.py`, `src/image_embeddings.py`) and the `data/questions.jsonl` skeleton + tests; run the unit tests locally (mocked LLM) and report back with failing cases.


---
Generated on 2026-01-13

